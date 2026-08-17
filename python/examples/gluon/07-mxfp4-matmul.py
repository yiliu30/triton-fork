"""
Pure MXFP4 x MXFP4 2CTA GEMM
=============================

A focused learning example for the high-performance Blackwell
2CTA warp-specialized MXFP4 GEMM path. The implementation keeps
the TMA loads, packed MXFP4 operands, swizzled MX scales,
warp-specialized pipeline, and fixed performance configuration
from the general block-scaled example, while removing mixed
formats, cuBLAS, and autotuning branches.
"""

import torch

import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch, upcast_from_mxfp_torch

from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    clc,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
    mbarrier,
    tma,
)

# ---------------------------------------------------------------------------
# Tile scheduler
# ---------------------------------------------------------------------------


@gluon.jit
def _planar_snake(lin_idx, m_tiles, n_tiles, minor_dim: gl.constexpr, tile_width: gl.constexpr):
    major_size = n_tiles if minor_dim == 0 else m_tiles
    minor_size = m_tiles if minor_dim == 0 else n_tiles

    full_minor_tiles = minor_size // tile_width
    full_minor_size = full_minor_tiles * tile_width
    full_elements = full_minor_tiles * tile_width * major_size

    minor_tile_idx = lin_idx // (tile_width * major_size)

    full_minor_within = lin_idx % tile_width
    full_major_within = (lin_idx // tile_width) % major_size
    full_minor = minor_tile_idx * tile_width + full_minor_within
    full_major = gl.where((minor_tile_idx % 2) == 0, full_major_within, major_size - 1 - full_major_within)

    partial_width = minor_size - full_minor_size
    partial_width = gl.where(partial_width > 0, partial_width, 1)
    partial_lin = lin_idx - full_elements
    partial_minor_within = partial_lin % partial_width
    partial_major_within = (partial_lin // partial_width) % major_size
    partial_minor = minor_tile_idx * tile_width + partial_minor_within
    partial_major = gl.where((minor_tile_idx % 2) == 0, partial_major_within, major_size - 1 - partial_major_within)

    in_full_tile = lin_idx < full_elements
    minor = gl.where(in_full_tile, full_minor, partial_minor)
    major = gl.where(in_full_tile, full_major, partial_major)

    if minor_dim == 0:
        return minor, major
    return major, minor


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


@gluon.constexpr_function
def get_split_dim(cga_layout, dim):
    return 1 << sum(b[dim] != 0 for b in cga_layout)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def quantize_and_pack_mxfp4(values):
    """Quantize float values to MXFP4 and pack two E2M1 values per byte.

    ``downcast_to_mxfp_torch`` is the repository's reference microscaling
    implementation. It computes one E8M0 scale per block of 32 values,
    applies the scale before E2M1 rounding, and returns the packed uint8
    representation consumed by the GEMM kernel.
    """
    if values.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("values must be float16, bfloat16, or float32")
    if values.ndim != 2 or values.shape[1] % 32 != 0:
        raise ValueError("values must be 2D with K divisible by the MX block size 32")

    packed, scales = downcast_to_mxfp_torch(values, torch.uint8, axis=1)
    dequantized = upcast_from_mxfp_torch(packed, scales, torch.float32, axis=1)
    return packed, scales, dequantized


def random_mxfp4_tensor(MN, K):
    """Create float input, then quantize, pack, and dequantize it for reference."""
    assert K % 32 == 0, "MXFP4 scales cover blocks of 32 values along K"
    source = torch.randn((MN, K), device="cuda", dtype=torch.float32)
    return quantize_and_pack_mxfp4(source)


def align_to(a, b):
    # Return next multiple of `b` greater than or equal to `a`.
    return triton.cdiv(a, b) * b


def swizzle_scales_packed_block(scales: torch.Tensor):
    # When the scale tensor is not an even multiple of [128, 4], we need to pad
    # the scale tensor so it can use the packed block format.
    PAD_MN = align_to(scales.shape[0], 128) - scales.shape[0]
    PAD_K = align_to(scales.shape[1], 4) - scales.shape[1]
    scales = torch.nn.functional.pad(scales, (0, PAD_K, 0, PAD_MN))

    MN, SCALE_K = scales.shape[0], scales.shape[1]
    REP_MN = MN // 128
    REP_K = SCALE_K // 4
    scales = scales.reshape(REP_MN, 4, 32, REP_K, 4)
    scales = scales.permute(0, 3, 2, 1, 4)
    return scales.contiguous()


# ---------------------------------------------------------------------------
# Fixed MXFP4 configuration and TMA setup
# ---------------------------------------------------------------------------

SCHEDULER_CLC = gl.constexpr("cluster_launch_control")
SCHEDULER_SPS = gl.constexpr("static_persistent")

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 256
EPILOGUE_BLOCK_N = 64
NUM_BUFFERS = 5
NUM_ACC_BUFFERS = 1
GRID_MINOR_DIM = 0
GRID_TILE_WIDTH = 8
NUM_CTAS = 2


def mma_scaled_tma_set_block_size_hook(nargs):
    """Set the exact TMA shapes/layouts used by the MXFP4-only kernel."""
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    epilogue_n = nargs["EPILOGUE_BLOCK_N"]
    cga_layout = nargs["CGA_LAYOUT"]

    a_block = [block_m, block_k // 2]
    b_block = [block_n, block_k // 2]
    c_block = [block_m, epilogue_n]
    cga = tuple(tuple(x) for x in cga_layout) if cga_layout else None

    nargs["a_desc"].block_shape = a_block
    nargs["b_desc"].block_shape = b_block
    nargs["c_desc"].block_shape = c_block
    nargs["a_desc"].layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.uint8, cga_layout=cga)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(b_block, gl.uint8, cga_layout=cga)

    c_dtype = getattr(gl, str(nargs["c_desc"].base.dtype).split('.')[1])
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(c_block, c_dtype, cga_layout=cga)

    rep_m = block_m // 128
    rep_n = block_n // 128
    rep_k = block_k // (32 * 4)
    nargs["a_scale_desc"].block_shape = [1, rep_m, rep_k, 2, 256]
    nargs["b_scale_desc"].block_shape = [1, rep_n, rep_k, 2, 256]

    if cga_layout:
        nargs["a_scale_desc"].layout = gl.NVMMASharedLayout(
            swizzle_byte_width=0, element_bitwidth=8, rank=5, cga_layout=[[0, 1, 0, 0, 0]])
        nargs["b_scale_desc"].layout = gl.NVMMASharedLayout(
            swizzle_byte_width=0, element_bitwidth=8, rank=5, cga_layout=[[0, 0, 0, 0, 0]])
    else:
        no_swizzle = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
        nargs["a_scale_desc"].layout = no_swizzle
        nargs["b_scale_desc"].layout = no_swizzle


@gluon.jit
def unswizzle_scales_shared_memory(smem, BLOCK_MN: gl.constexpr, BLOCK_K: gl.constexpr, VEC_SIZE: gl.constexpr):
    smem = smem.reshape((smem.shape[1], smem.shape[2], 32, 4, 4))
    smem = smem.permute((0, 3, 2, 1, 4))
    return smem.reshape((BLOCK_MN, BLOCK_K // VEC_SIZE))


@gluon.jit
def async_mma_scaled_impl(a_smem, b_smem, a_scale_smem, b_scale_smem, acc_tmem, mma_bar, use_acc, pred):
    BLOCK_M: gl.constexpr = a_smem.shape[0]
    BLOCK_N: gl.constexpr = b_smem.shape[0]
    BLOCK_K: gl.constexpr = a_smem.shape[1] * 2
    VEC_SIZE: gl.constexpr = 32

    a_scale = unswizzle_scales_shared_memory(a_scale_smem, BLOCK_M, BLOCK_K, VEC_SIZE)
    b_scale = unswizzle_scales_shared_memory(b_scale_smem, BLOCK_N, BLOCK_K, VEC_SIZE)

    two_ctas: gl.constexpr = acc_tmem.type.layout.two_ctas
    a_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=[[1, 0]] if two_ctas else [])
    b_scale_layout: gl.constexpr = TensorMemoryScalesLayout(cga_layout=[[0, 0]] if two_ctas else [])
    a_scale_tmem = allocate_tensor_memory(a_scale.dtype, a_scale.type.shape, a_scale_layout)
    b_scale_tmem = allocate_tensor_memory(b_scale.dtype, b_scale.type.shape, b_scale_layout)
    tcgen05_copy(a_scale, a_scale_tmem)
    tcgen05_copy(b_scale, b_scale_tmem)

    tcgen05_mma_scaled(a_smem, b_smem.permute((1, 0)), acc_tmem, a_scale_tmem, b_scale_tmem, "e2m1", "e2m1",
                       use_acc=use_acc, pred=pred, multicast=True, mbarriers=[mma_bar])


# This helper function computes all the load indexing and issues the async loads
# based on the current `pid_m`, `pid_n`, and `k` indices. The compiler will run
# loop-invariant code motion to hoist code that does not depend on `k`, like
# `pid_m * BLOCK_M`, outside of the inner loop, so we can safely abstract the
# load indexing without performance loss.
#
# Encapsulating the load indexing logic will help keep our pipelined kernel code
# clean, as pipelining can get messy.
@gluon.jit
def issue_loads(producer, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs,
                b_scale_bufs, bars, pred):
    A_ELEM_PER_BYTE: gl.constexpr = 2
    B_ELEM_PER_BYTE: gl.constexpr = 2
    BLOCK_M: gl.constexpr = a_desc.block_shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_shape[0]
    BLOCK_K: gl.constexpr = a_desc.block_shape[1] * A_ELEM_PER_BYTE
    REP_M: gl.constexpr = a_scale_desc.block_shape[1]
    REP_N: gl.constexpr = b_scale_desc.block_shape[1]
    A_REP_K: gl.constexpr = a_scale_desc.block_shape[2]
    B_REP_K: gl.constexpr = b_scale_desc.block_shape[2]

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    off_m_a_scale = pid_m * REP_M
    off_n_b_scale = pid_n * REP_N
    off_k_a = k // A_ELEM_PER_BYTE
    off_k_b = k // B_ELEM_PER_BYTE
    off_k_a_scale = (k // BLOCK_K) * A_REP_K
    off_k_b_scale = (k // BLOCK_K) * B_REP_K

    index = producer.index
    bar = bars.index(index)
    mbarrier.expect(
        bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta + a_scale_desc.nbytes_per_cta + b_scale_desc.nbytes_per_cta,
        pred)
    tma.async_load(a_desc, [off_m, off_k_a], bar, a_bufs.index(index), pred, multicast=True)
    tma.async_load(b_desc, [off_n, off_k_b], bar, b_bufs.index(index), pred, multicast=True)
    tma.async_load(a_scale_desc, [0, off_m_a_scale, off_k_a_scale, 0, 0], bar, a_scale_bufs.index(index), pred,
                   multicast=True)
    tma.async_load(b_scale_desc, [0, off_n_b_scale, off_k_b_scale, 0, 0], bar, b_scale_bufs.index(index), pred,
                   multicast=True)
    return producer.next(pred)


@gluon.jit
def issue_mma(consumer, c_bars, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, producer, p_bars, acc_tmem, use_acc, pred):
    c_index = consumer.index
    mbarrier.wait(c_bars.index(c_index), consumer.phase, pred)
    async_mma_scaled_impl(a_bufs.index(c_index), b_bufs.index(c_index), a_scale_bufs.index(c_index),
                          b_scale_bufs.index(c_index), acc_tmem, p_bars.index(producer.index), use_acc, pred)
    return consumer.next(pred), producer.next(pred)


@gluon.aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)


@gluon.aggregate
class ClcTileSchedulerConsumer:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    num_pid_m: gl.tensor
    num_pid_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_planar_pid_buffers: gl.shared_memory_descriptor
    clc_planar_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    counter: Counter
    consumed_counter: Counter

    @gluon.jit
    def initialize(M, N, TILE_M: gl.constexpr, TILE_N: gl.constexpr, MINOR_DIM: gl.constexpr,
                   GRID_TILE_WIDTH: gl.constexpr, clc_result_buffers, clc_barriers, clc_planar_pid_buffers,
                   clc_planar_ready_bars, clc_consumed_bars):
        tile_id = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(M, TILE_M)
        num_pid_n = gl.cdiv(N, TILE_N)
        pid_m, pid_n = _planar_snake(tile_id, num_pid_m, num_pid_n, MINOR_DIM, GRID_TILE_WIDTH)
        has_work = gl.to_tensor(True)
        counter = Counter.create(0, clc_barriers.shape[0])
        consumed_counter = Counter.create(0, clc_barriers.shape[0])
        return ClcTileSchedulerConsumer(
            has_work,
            tile_id,
            pid_m,
            pid_n,
            num_pid_m,
            num_pid_n,
            TILE_M,
            TILE_N,
            MINOR_DIM,
            GRID_TILE_WIDTH,
            clc_result_buffers,
            clc_barriers,
            clc_planar_pid_buffers,
            clc_planar_ready_bars,
            clc_consumed_bars,
            counter,
            consumed_counter,
        )

    @gluon.jit
    def get_offsets(self):
        return self.pid_m * self.TILE_M, self.pid_n * self.TILE_N

    @gluon.jit
    def step(self, iteration):
        # The 0-th iteration uses the program_id as the tile_id.
        # At the end of each iteration we prefetch the next tile.
        # As such we must signal the consumed slot at the end of
        # each iteration skipping the first one.
        consumed_counter = self.consumed_counter
        if iteration > 0:
            mbarrier.arrive(self.clc_consumed_bars.index(consumed_counter.index))
            consumed_counter = consumed_counter.next()
        counter = self.counter
        barrier = self.clc_barriers.index(counter.index)
        result = self.clc_result_buffers.index(counter.index)
        mbarrier.wait(barrier, counter.phase)
        clc_res = clc.load_result(result)
        mbarrier.wait(self.clc_planar_ready_bars.index(counter.index), counter.phase)
        planar_slot = self.clc_planar_pid_buffers.index(counter.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        packed_pid = planar_slot.load(planar_layout).reshape([])
        pid_m = ((packed_pid >> 32) & 0xFFFFFFFF).to(gl.int32)
        pid_n = (packed_pid & 0xFFFFFFFF).to(gl.int32)
        has_work = clc_res.is_canceled()
        tile_id = self.tile_id
        if has_work:
            tile_id = clc_res.program_id(0)
        return ClcTileSchedulerConsumer(
            has_work,
            tile_id,
            pid_m,
            pid_n,
            self.num_pid_m,
            self.num_pid_n,
            self.TILE_M,
            self.TILE_N,
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
            self.clc_result_buffers,
            self.clc_barriers,
            self.clc_planar_pid_buffers,
            self.clc_planar_ready_bars,
            self.clc_consumed_bars,
            counter.next(),
            consumed_counter,
        )


@gluon.aggregate
class SpsTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    NUM_PID_M: gl.tensor
    NUM_PID_N: gl.tensor

    @gluon.jit
    def initialize(TILE_M: gl.constexpr, TILE_N: gl.constexpr, MINOR_DIM: gl.constexpr, GRID_TILE_WIDTH: gl.constexpr,
                   NUM_PID_M, NUM_PID_N):
        tile_id = gl.program_id(axis=0)
        has_work = tile_id < NUM_PID_M * NUM_PID_N
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            pid_m, pid_n = _planar_snake(tile_id, NUM_PID_M, NUM_PID_N, MINOR_DIM, GRID_TILE_WIDTH)
        return SpsTileScheduler(has_work, tile_id, pid_m, pid_n, TILE_M, TILE_N, MINOR_DIM, GRID_TILE_WIDTH, NUM_PID_M,
                                NUM_PID_N)

    @gluon.jit
    def get_offsets(self):
        return self.pid_m * self.TILE_M, self.pid_n * self.TILE_N

    @gluon.jit
    def step(self, iteration):
        next_tile_id = self.tile_id + gl.num_programs(axis=0)
        has_work = next_tile_id < self.NUM_PID_M * self.NUM_PID_N
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            pid_m, pid_n = _planar_snake(next_tile_id, self.NUM_PID_M, self.NUM_PID_N, self.MINOR_DIM,
                                         self.GRID_TILE_WIDTH)
        return SpsTileScheduler(has_work, next_tile_id, pid_m, pid_n, self.TILE_M, self.TILE_N, self.MINOR_DIM,
                                self.GRID_TILE_WIDTH, self.NUM_PID_M, self.NUM_PID_N)


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


@gluon.aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_scale_desc: tma.tensor_descriptor
    b_scale_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    a_scale_bufs: gl.shared_memory_descriptor
    b_scale_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_planar_pid_buffers: gl.shared_memory_descriptor
    clc_planar_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    MINOR_DIM: gl.constexpr
    GRID_TILE_WIDTH: gl.constexpr
    NUM_PID_M: gl.tensor
    NUM_PID_N: gl.tensor
    scheduler: gl.constexpr

    @gluon.jit
    def get_clc_consumer(self):
        return ClcTileSchedulerConsumer.initialize(
            self.c_desc.shape[0],
            self.c_desc.shape[1],
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[0],
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
            self.clc_result_buffers,
            self.clc_barriers,
            self.clc_planar_pid_buffers,
            self.clc_planar_ready_bars,
            self.clc_consumed_bars,
        )

    @gluon.jit
    def get_sps_scheduler(self):
        return SpsTileScheduler.initialize(
            self.a_desc.block_shape[0],
            self.b_desc.block_shape[0],
            self.MINOR_DIM,
            self.GRID_TILE_WIDTH,
            self.NUM_PID_M,
            self.NUM_PID_N,
        )


@gluon.jit
def mma_scaled_load_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * 2
    state = Counter.create(1, p.load_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        for k in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            state = issue_loads(state, scheduler.pid_m, scheduler.pid_n, k, p.a_desc, p.b_desc, p.a_scale_desc,
                                p.b_scale_desc, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs, p.load_ready_bars,
                                pred=True)
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def mma_scaled_mma_partition(p):
    A_ELEM_PER_BYTE: gl.constexpr = 2
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * A_ELEM_PER_BYTE
    K = p.a_desc.shape[1] * 2
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for k in range(0, K, BLOCK_K):
            _, load_state = issue_mma(load_state, p.load_ready_bars, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs,
                                      load_state, p.load_empty_bars, acc_buf, use_acc, pred=True)
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def mma_scaled_epilogue_partition(p):
    tile_m: gl.constexpr = p.c_desc.block_shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_shape[0]
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_shape[1]
    subtile_factor: gl.constexpr = BLOCK_N // EPILOGUE_BLOCK_N
    subtile_stages: gl.constexpr = 1 if subtile_factor == 1 else 2
    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    acc_smems = gl.allocate_shared_memory(p.c_desc.dtype, [subtile_stages, tile_m, EPILOGUE_BLOCK_N], p.c_desc.layout)
    sub_acc_state = Counter.create(0, subtile_stages)
    if p.scheduler == SCHEDULER_CLC:
        scheduler = p.get_clc_consumer()
    else:
        scheduler = p.get_sps_scheduler()
    i = 0
    while scheduler.has_work:
        off_m, off_n = scheduler.get_offsets()
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)

        for s in gl.static_range(subtile_factor):
            acc_sub = acc_buf.slice(EPILOGUE_BLOCK_N * s, EPILOGUE_BLOCK_N)
            acc_smem = acc_smems.index(sub_acc_state.index)
            acc = acc_sub.load().to(p.c_desc.dtype)
            tma.store_wait(pendings=subtile_stages - 1)
            acc_smem.store(acc)
            tma.async_store(p.c_desc, [off_m, off_n + EPILOGUE_BLOCK_N * s], acc_smem)
            sub_acc_state = sub_acc_state.next()
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1
    tma.store_wait(0)


@gluon.jit
def mma_scaled_clc_partition(p):
    TILE_M: gl.constexpr = p.a_desc.block_shape[0]
    TILE_N: gl.constexpr = p.b_desc.block_shape[0]
    has_work = gl.to_tensor(True)
    num_pid_m = gl.cdiv(p.c_desc.shape[0], TILE_M)
    num_pid_n = gl.cdiv(p.c_desc.shape[1], TILE_N)
    state = Counter.create(0, p.clc_barriers.shape[0])
    consumed_state = Counter.create(1, p.clc_barriers.shape[0])
    ACC_STAGES: gl.constexpr = p.clc_barriers.shape[0]
    i = 0
    while has_work:
        # Reuse the slot only after all consumer partitions signaled consumed.
        mbarrier.wait(p.clc_consumed_bars.index(consumed_state.index), consumed_state.phase, pred=(i >= ACC_STAGES))
        barrier = p.clc_barriers.index(state.index)
        result = p.clc_result_buffers.index(state.index)
        # 16: clc.try_cancel has a `.b128` modifier
        mbarrier.expect(barrier, 16, from_cta=0x0)
        clc.try_cancel(result, barrier)
        mbarrier.wait(barrier, state.phase)
        clc_res = clc.load_result(result)
        has_work = clc_res.is_canceled()
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            tile_id = clc_res.program_id(0)
            pid_m, pid_n = _planar_snake(tile_id, num_pid_m, num_pid_n, p.MINOR_DIM, p.GRID_TILE_WIDTH)
        packed_pid = (pid_m.to(gl.int64) << 32) | (pid_n.to(gl.int64) & 0xFFFFFFFF)
        planar_slot = p.clc_planar_pid_buffers.index(state.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        planar_slot.store(gl.full([1], packed_pid, gl.int64, layout=planar_layout))
        mbarrier.arrive(p.clc_planar_ready_bars.index(state.index))
        state = state.next()
        consumed_state = consumed_state.next()
        i += 1


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@gluon.jit
def mma_scaled_warp_specialized_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, M, N, K, A_ELEM_PER_BYTE,
                                       num_buffers: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                                       BLOCK_K: gl.constexpr, EPILOGUE_BLOCK_N: gl.constexpr,
                                       num_acc_buffers: gl.constexpr, GRID_MINOR_DIM: gl.constexpr,
                                       GRID_TILE_WIDTH: gl.constexpr, CGA_LAYOUT: gl.constexpr,
                                       scheduler: gl.constexpr):
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)
    N_PARTITIONS: gl.constexpr = 4 if scheduler == SCHEDULER_CLC else 3

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_shape,
                                             b_scale_desc.layout)

    tmem_layout: gl.constexpr = TensorMemoryLayout([BLOCK_M_PER_CTA, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT,
                                                   two_ctas=TWO_CTAS)
    acc_bufs = allocate_tensor_memory(gl.float32, [num_acc_buffers, BLOCK_M, BLOCK_N], tmem_layout)

    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0),
         a_scale_bufs.index(0), b_scale_bufs.index(0)], multicast=True, two_ctas=acc_bufs.index(0).type.layout.two_ctas)

    load_empty_bars = mbarrier.allocate_mbarrier(batch=num_buffers)
    load_ready_bars = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)

    acc_empty_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers, two_ctas=TWO_CTAS)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    for i in gl.static_range(num_acc_buffers):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    clc_barriers = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    clc_planar_ready_bars = mbarrier.allocate_mbarrier(batch=num_acc_buffers)
    cga_layout_clc: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    clc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=cga_layout_clc)
    clc_consumed_bars = gl.allocate_shared_memory(gl.int64, [num_acc_buffers, 1], clc_layout)
    if scheduler == SCHEDULER_CLC:
        for i in gl.static_range(num_acc_buffers):
            mbarrier.init(clc_barriers.index(i), count=1)
            mbarrier.init(clc_planar_ready_bars.index(i), count=1)
            mbarrier.init(clc_consumed_bars.index(i), count=N_PARTITIONS - 1)

    clc_result_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 2], clc_layout)
    clc_planar_pid_buffers = gl.allocate_shared_memory(gl.int64, [clc_barriers.shape[0], 1], clc_layout)

    p = PartitionArgs(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                      load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars, acc_ready_bars, clc_result_buffers,
                      clc_barriers, clc_planar_pid_buffers, clc_planar_ready_bars, clc_consumed_bars, GRID_MINOR_DIM,
                      GRID_TILE_WIDTH, gl.cdiv(M, BLOCK_M), gl.cdiv(N, BLOCK_N), scheduler)

    if scheduler == SCHEDULER_CLC:
        gl.warp_specialize([
            (mma_scaled_epilogue_partition, (p, )),
            (mma_scaled_mma_partition, (p, )),
            (mma_scaled_load_partition, (p, )),
            (mma_scaled_clc_partition, (p, )),
        ], [1, 1, 1], [24, 24, 24])
    else:
        gl.warp_specialize([
            (mma_scaled_epilogue_partition, (p, )),
            (mma_scaled_mma_partition, (p, )),
            (mma_scaled_load_partition, (p, )),
        ], [1, 1], [24, 24])


def mma_scaled_warp_specialized_grid(M, N, BLOCK_M, BLOCK_N, num_ctas, scheduler, device):
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if scheduler == SCHEDULER_CLC:
        return (num_pid, )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return (max(1, min(num_pid, sm_count // num_ctas)), )


def scheduler_for_k(K):
    return SCHEDULER_SPS if K <= 8192 else SCHEDULER_CLC


def make_dummy_descriptors(A, B, A_scale, B_scale, out_dtype, M, N):
    """Create TMA descriptors; the block-size hook fills the real shapes."""
    dummy_block_2d = [1, 1]
    dummy_layout_2d = gl.NVMMASharedLayout.get_default_for(dummy_block_2d, gl.uint8)
    a_desc = TensorDescriptor.from_tensor(A, dummy_block_2d, dummy_layout_2d)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block_2d, dummy_layout_2d)

    C = torch.empty((M, N), device="cuda", dtype=out_dtype)
    C_dtype = getattr(gl, str(out_dtype).split('.')[1])
    c_layout = gl.NVMMASharedLayout.get_default_for(dummy_block_2d, C_dtype)
    c_desc = TensorDescriptor.from_tensor(C, dummy_block_2d, c_layout)

    A_scale_5d = A_scale.reshape(1, A_scale.shape[0], A_scale.shape[1], 2, 256)
    B_scale_5d = B_scale.reshape(1, B_scale.shape[0], B_scale.shape[1], 2, 256)
    dummy_block_5d = [1, 1, 1, 2, 256]
    dummy_layout_5d = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    a_scale_desc = TensorDescriptor.from_tensor(A_scale_5d, dummy_block_5d, dummy_layout_5d)
    b_scale_desc = TensorDescriptor.from_tensor(B_scale_5d, dummy_block_5d, dummy_layout_5d)
    return a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc


def mxfp4_matmul(A, B, A_scale, B_scale, out_dtype=torch.float16):
    """Run the fixed, performance-tuned 2CTA MXFP4 x MXFP4 GEMM."""
    if A.dtype != torch.uint8 or B.dtype != torch.uint8:
        raise TypeError("A and B must be packed MXFP4 tensors with dtype torch.uint8")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same packed K dimension")

    M, N = A.shape[0], B.shape[0]
    K = A.shape[1] * 2
    A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc = make_dummy_descriptors(
        A, B, A_scale, B_scale, out_dtype, M, N)
    cga_layout = ((1, 0), )
    mma_scaled_tma_set_block_size_hook({
        "a_desc": A_desc,
        "b_desc": B_desc,
        "c_desc": C_desc,
        "a_scale_desc": A_scale_desc,
        "b_scale_desc": B_scale_desc,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "EPILOGUE_BLOCK_N": EPILOGUE_BLOCK_N,
        "CGA_LAYOUT": cga_layout,
    })

    grid = mma_scaled_warp_specialized_grid(M, N, BLOCK_M, BLOCK_N, NUM_CTAS, scheduler_for_k(K), A.device)
    mma_scaled_warp_specialized_kernel[grid](
        A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, M, N, K, 2,
        NUM_BUFFERS, BLOCK_M, BLOCK_N, BLOCK_K, EPILOGUE_BLOCK_N, NUM_ACC_BUFFERS,
        GRID_MINOR_DIM, GRID_TILE_WIDTH, cga_layout, scheduler=scheduler_for_k(K), num_ctas=NUM_CTAS)
    return C_desc.base


# ---------------------------------------------------------------------------
# Correctness check and benchmark
# ---------------------------------------------------------------------------


def run_benchmark():
    print(f"MXFP4 x MXFP4 2CTA GEMM ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}, {NUM_BUFFERS=})")
    print(f"{'MNK':>8}  {'TFLOPS':>12}  {'max abs error':>16}")
    for mnk in (8192, 16384, 32768):
        torch.manual_seed(0)
        A, A_scale, A_ref = random_mxfp4_tensor(mnk, mnk)
        B, B_scale, B_ref = random_mxfp4_tensor(mnk, mnk)
        A_scale = swizzle_scales_packed_block(A_scale)
        B_scale = swizzle_scales_packed_block(B_scale)
        reference = A_ref @ B_ref.T

        output = mxfp4_matmul(A, B, A_scale, B_scale)
        # FP16 output can accumulate a larger absolute rounding error as K
        # grows. Scale the absolute bound with K while checking every output.
        atol = 1e-3 + 0.6e-6 * mnk
        torch.testing.assert_close(reference, output.to(torch.float32), atol=atol, rtol=1e-3)
        max_error = (reference - output.to(torch.float32)).abs().max().item()

        ms = triton.testing.do_bench(lambda: mxfp4_matmul(A, B, A_scale, B_scale))
        tflops = 2.0 * mnk**3 * 1e-12 / (ms * 1e-3)
        print(f"{mnk:>8}  {tflops:>12.1f}  {max_error:>16.6f}")


if __name__ == "__main__":
    if not is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    run_benchmark()
