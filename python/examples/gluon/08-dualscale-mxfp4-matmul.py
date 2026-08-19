"""
Dual-scale MXFP4 x MXFP4 GEMM.

This example keeps the fine-scale Blackwell MMA path from
``07-mxfp4-matmul.py`` and adds an FP32 coarse scale for every 512 K values.
The coarse scale is applied to each 512-K partial after it is loaded into
registers; only the running output tile is retained in Tensor Memory.
"""

import importlib.util
from pathlib import Path

import torch

import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch, upcast_from_mxfp_torch

from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tensor_memory_descriptor,
    tcgen05_copy,
    tcgen05_commit,
    tcgen05_mma_barrier_count,
    tcgen05_mma_scaled,
    mbarrier,
    tma,
)


def _load_baseline():
    path = Path(__file__).with_name("07-mxfp4-matmul.py")
    spec = importlib.util.spec_from_file_location("mxfp4_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = _load_baseline()


# Keep the tuned baseline tile and pipeline configuration as the starting
# point.  The dual accumulator is the only intentional Tensor Memory change.
BLOCK_M = baseline.BLOCK_M
# Keep the tuned baseline N tile.  The experimental sequential path below
# remains available for compiler/correctness isolation, while the production
# path uses warp-specialized load, MMA, and epilogue partitions.
BLOCK_N = baseline.BLOCK_N
BLOCK_K = baseline.BLOCK_K
COARSE_K = 512
EPILOGUE_BLOCK_N = BLOCK_N
NUM_BUFFERS = baseline.NUM_BUFFERS
GRID_MINOR_DIM = baseline.GRID_MINOR_DIM
GRID_TILE_WIDTH = baseline.GRID_TILE_WIDTH
NUM_CTAS = baseline.NUM_CTAS


def quantize_and_pack_dualscale(values):
    """Return packed FP4, fine E8M0 scales, coarse FP32 scales, and a reference.

    The coarse scale is the exact per-512-block max divided by the largest
    finite E2M1 value (6).  Fine quantization is then the canonical repository
    MXFP4 quantizer applied to the normalized values.
    """
    if values.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("values must be float16, bfloat16, or float32")
    if values.ndim != 2 or values.shape[1] % COARSE_K != 0:
        raise ValueError(f"values must be 2D with K divisible by {COARSE_K}")

    values = values.to(torch.float32)
    rows, K = values.shape
    coarse = values.abs().reshape(rows, K // COARSE_K, COARSE_K).amax(dim=-1) / 6.0
    coarse = torch.where(coarse == 0, torch.ones_like(coarse), coarse)
    normalized = (values.reshape(rows, K // COARSE_K, COARSE_K) /
                  coarse[..., None]).reshape(rows, K)
    packed, fine = downcast_to_mxfp_torch(normalized, torch.uint8, axis=1)
    normalized_ref = upcast_from_mxfp_torch(packed, fine, torch.float32, axis=1)
    reference = normalized_ref.reshape(rows, K // COARSE_K, COARSE_K) * coarse[..., None]
    return packed, fine, coarse.contiguous(), reference.reshape(rows, K)


def random_dualscale_tensor(rows, K):
    source = torch.randn((rows, K), device="cuda", dtype=torch.float32)
    return quantize_and_pack_dualscale(source)


@gluon.aggregate
class DualScaleArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_scale_desc: tma.tensor_descriptor
    b_scale_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    a_scale_bufs: gl.shared_memory_descriptor
    b_scale_bufs: gl.shared_memory_descriptor
    a_coarse_ptr: gl.tensor
    b_coarse_ptr: gl.tensor
    coarse_count: gl.constexpr
    load_empty: gl.shared_memory_descriptor
    load_ready: gl.shared_memory_descriptor
    partial: tensor_memory_descriptor
    partial_empty: gl.shared_memory_descriptor
    partial_ready: gl.shared_memory_descriptor
    pid_m: gl.tensor
    pid_n: gl.tensor


@gluon.jit
def dual_load_partition(p):
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * 2
    K = p.a_desc.shape[1] * 2
    state = baseline.Counter.create(1, p.load_empty.shape[0])
    off_m = p.pid_m * p.a_desc.block_shape[0]
    off_n = p.pid_n * p.b_desc.block_shape[0]
    for k in range(0, K, BLOCK_K):
        mbarrier.wait(p.load_empty.index(state.index), state.phase)
        state = baseline.issue_loads(
            state, p.pid_m, p.pid_n, k, p.a_desc, p.b_desc, p.a_scale_desc, p.b_scale_desc,
            p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs, p.load_ready, True)


@gluon.jit
def dual_mma_partition(p):
    BLOCK_K: gl.constexpr = p.a_desc.block_shape[1] * 2
    K = p.a_desc.shape[1] * 2
    load_state = baseline.Counter.create(0, p.load_empty.shape[0])
    empty_state = baseline.Counter.create(0, 1)
    for coarse_k in range(0, K, 512):
        mbarrier.wait(p.partial_empty, empty_state.phase)
        use_acc = False
        for k in range(coarse_k, coarse_k + 512, BLOCK_K):
            _, load_state = baseline.issue_mma(
                load_state, p.load_ready, p.a_bufs, p.b_bufs, p.a_scale_bufs, p.b_scale_bufs,
                load_state, p.load_empty, p.partial, use_acc, True)
            use_acc = True
        tcgen05_commit(p.partial_ready)
        empty_state = empty_state.next()


@gluon.jit
def dual_epilogue_partition(p):
    BLOCK_M: gl.constexpr = p.c_desc.block_shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_shape[0]
    EPILOGUE_N: gl.constexpr = p.c_desc.block_shape[1]
    subtile_stages: gl.constexpr = 1
    TWO_CTAS: gl.constexpr = gl.num_ctas() > 1
    K = p.a_desc.shape[1] * 2
    coarse_count: gl.constexpr = K // 512
    acc_smems = gl.allocate_shared_memory(p.c_desc.dtype, [subtile_stages, BLOCK_M, EPILOGUE_N], p.c_desc.layout)
    sub_state = baseline.Counter.create(0, subtile_stages)
    matrix_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [32, 1], [gl.num_warps(), 1], [1, 0],
        cga_layout=((1, 0), ) if TWO_CTAS else ())
    ready_state = baseline.Counter.create(0, 1)
    off_m = p.pid_m * BLOCK_M
    off_n = p.pid_n * BLOCK_N
    running = gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, matrix_layout)
    for coarse_idx in range(coarse_count):
        mbarrier.wait(p.partial_ready, ready_state.phase)
        partial = p.partial.load(matrix_layout)
        running = partial
        mbarrier.arrive(p.partial_empty, count=1)
        ready_state = ready_state.next()

    acc_smem = acc_smems.index(sub_state.index)
    acc_smem.store(running.to(p.c_desc.dtype))
    tma.async_store(p.c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


@gluon.jit
def dualscale_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_coarse_ptr, b_coarse_ptr,
                     coarse_count, M, N, K, num_buffers: gl.constexpr, BLOCK_M: gl.constexpr,
                     BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, EPILOGUE_BLOCK_N: gl.constexpr,
                     CGA_LAYOUT: gl.constexpr):
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_shape,
                                             a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_shape,
                                             b_scale_desc.layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [BLOCK_M_PER_CTA, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT, two_ctas=TWO_CTAS)
    partial = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0), a_scale_bufs.index(0), b_scale_bufs.index(0)],
        multicast=True, two_ctas=TWO_CTAS)
    load_empty = mbarrier.allocate_mbarrier(batch=num_buffers)
    load_ready = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready.index(i), count=1)

    partial_empty = mbarrier.allocate_mbarrier(two_ctas=TWO_CTAS)
    partial_ready = mbarrier.allocate_mbarrier()
    mbarrier.init(partial_empty, count=1)
    mbarrier.init(partial_ready, count=1)

    args = DualScaleArgs(
        a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
        a_coarse_ptr, b_coarse_ptr, coarse_count,
        load_empty, load_ready,
        partial, partial_empty, partial_ready,
        gl.program_id(0) // gl.cdiv(N, BLOCK_N),
        gl.program_id(0) % gl.cdiv(N, BLOCK_N))
    dual_load_partition(args)
    dual_mma_partition(args)
    dual_epilogue_partition(args)


@gluon.jit
def dualscale_sequential_kernel(a_desc, b_desc, c_desc, a_scale_desc, b_scale_desc, a_coarse_ptr, b_coarse_ptr,
                                coarse_count, M, N, K, num_buffers: gl.constexpr, BLOCK_M: gl.constexpr,
                                BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, EPILOGUE_BLOCK_N: gl.constexpr,
                                CGA_LAYOUT: gl.constexpr):
    """Correctness/performance fallback with one tile resident per program.

    This version deliberately keeps the whole coarse loop in one partition.
    It is useful as a compiler-friendly reference for the fused arithmetic:
    the MMA partial is reused for every coarse block, while ``running`` stays
    in registers instead of a second Tensor Memory allocation.
    """
    NUM_CTAS: gl.constexpr = gl.num_ctas()
    TWO_CTAS: gl.constexpr = NUM_CTAS > 1
    BLOCK_M_PER_CTA: gl.constexpr = BLOCK_M // NUM_CTAS
    gl.static_assert(BLOCK_M_PER_CTA == 64 or BLOCK_M_PER_CTA == 128)

    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_buffers] + a_desc.block_shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_buffers] + b_desc.block_shape, b_desc.layout)
    a_scale_bufs = gl.allocate_shared_memory(a_scale_desc.dtype, [num_buffers] + a_scale_desc.block_shape,
                                              a_scale_desc.layout)
    b_scale_bufs = gl.allocate_shared_memory(b_scale_desc.dtype, [num_buffers] + b_scale_desc.block_shape,
                                              b_scale_desc.layout)
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        [BLOCK_M_PER_CTA, BLOCK_N], col_stride=1, cga_layout=CGA_LAYOUT, two_ctas=TWO_CTAS)
    partial = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0), a_scale_bufs.index(0), b_scale_bufs.index(0)],
        multicast=True, two_ctas=TWO_CTAS)
    load_empty = mbarrier.allocate_mbarrier(batch=num_buffers)
    load_ready = mbarrier.allocate_mbarrier(batch=num_buffers, two_ctas=TWO_CTAS)
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready.index(i), count=1)
    partial_ready = mbarrier.allocate_mbarrier()
    mbarrier.init(partial_ready, count=1)

    pid_m = gl.program_id(0) // gl.cdiv(N, BLOCK_N)
    pid_n = gl.program_id(0) % gl.cdiv(N, BLOCK_N)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    load_state = baseline.Counter.create(1, num_buffers)
    mma_state = baseline.Counter.create(0, num_buffers)
    ready_phase = gl.to_tensor(0)
    matrix_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [32, 1], [gl.num_warps(), 1], [1, 0],
        cga_layout=((1, 0), ) if TWO_CTAS else ())
    running = gl.full([BLOCK_M, BLOCK_N], 0.0, gl.float32, matrix_layout)

    for coarse_idx in range(K // 512):
        use_acc = False
        for k in range(coarse_idx * 512, (coarse_idx + 1) * 512, BLOCK_K):
            mbarrier.wait(load_empty.index(load_state.index), load_state.phase)
            load_state = baseline.issue_loads(
                load_state, pid_m, pid_n, k, a_desc, b_desc, a_scale_desc, b_scale_desc,
                a_bufs, b_bufs, a_scale_bufs, b_scale_bufs, load_ready, True)
            _, mma_state = baseline.issue_mma(
                mma_state, load_ready, a_bufs, b_bufs, a_scale_bufs, b_scale_bufs,
                load_state, load_empty, partial, use_acc, True)
            use_acc = True
        tcgen05_commit(partial_ready)
        mbarrier.wait(partial_ready, ready_phase)
        partial_reg = partial.load(matrix_layout)
        partial_layout: gl.constexpr = partial_reg.type.layout
        coarse_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, partial_layout))
        coarse_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, partial_layout))
        a_scale = gl.load(a_coarse_ptr + (off_m + coarse_m) * coarse_count + coarse_idx)
        b_scale = gl.load(b_coarse_ptr + (off_n + coarse_n) * coarse_count + coarse_idx)
        a_scale = gl.convert_layout(a_scale, gl.SliceLayout(1, partial_layout))[:, None]
        b_scale = gl.convert_layout(b_scale, gl.SliceLayout(0, partial_layout))[None, :]
        a_scale = gl.convert_layout(a_scale, partial_layout)
        b_scale = gl.convert_layout(b_scale, partial_layout)
        scaled = partial_reg * a_scale
        scaled = scaled * b_scale
        running = running + scaled
        ready_phase = ready_phase ^ 1

    acc_smem = gl.allocate_shared_memory(c_desc.dtype, [BLOCK_M, EPILOGUE_BLOCK_N], c_desc.layout)
    acc_smem.store(running.to(c_desc.dtype))
    tma.async_store(c_desc, [off_m, off_n], acc_smem)
    tma.store_wait(0)


def dualscale_matmul(A, B, A_scale, B_scale, A_coarse, B_coarse, out_dtype=torch.float16):
    if A.dtype != torch.uint8 or B.dtype != torch.uint8:
        raise TypeError("A and B must be packed MXFP4 tensors with dtype torch.uint8")
    if A.shape[1] != B.shape[1] or A.shape[1] % (COARSE_K // 2) != 0:
        raise ValueError("A and B must have the same packed K dimension divisible by 256")
    M, N = A.shape[0], B.shape[0]
    K = A.shape[1] * 2
    if A_coarse.shape != (M, K // COARSE_K) or B_coarse.shape != (N, K // COARSE_K):
        raise ValueError("coarse scales must have shape [rows, K // 512]")

    A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc = baseline.make_dummy_descriptors(
        A, B, A_scale, B_scale, out_dtype, M, N)
    cga_layout = ((1, 0), ) if NUM_CTAS > 1 else ()
    baseline.mma_scaled_tma_set_block_size_hook({
        "a_desc": A_desc, "b_desc": B_desc, "c_desc": C_desc,
        "a_scale_desc": A_scale_desc, "b_scale_desc": B_scale_desc,
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "EPILOGUE_BLOCK_N": EPILOGUE_BLOCK_N, "CGA_LAYOUT": cga_layout,
    })
    device = A.device
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    # Direct tile dispatch keeps this experimental dual-scale kernel small
    # enough to compile; each program owns one output tile.
    grid = (num_pid, )
    # Keep the public entry point on the compiler-viable correctness path.
    # The warp-specialized partitioned kernel above remains experimental until
    # its lowering time and performance are validated.
    dualscale_sequential_kernel[grid](
        A_desc, B_desc, C_desc, A_scale_desc, B_scale_desc, A_coarse, B_coarse, K // COARSE_K,
        M, N, K, NUM_BUFFERS, BLOCK_M, BLOCK_N, BLOCK_K, EPILOGUE_BLOCK_N, cga_layout, num_ctas=NUM_CTAS)
    return C_desc.base


def dualscale_matmul_reference(A, B, A_fine, B_fine, A_coarse, B_coarse, out_dtype=torch.float16):
    """Reference launcher that validates the dual-scale data flow.

    It uses the proven fine-MXFP4 kernel once per 512-K region and applies the
    coarse outer product between launches.  This intentionally exists beside
    ``dualscale_matmul`` so the fused Tensor Memory implementation can be
    debugged against a simple, independently readable path.
    """
    if A.shape[1] != B.shape[1] or A.shape[1] % (COARSE_K // 2) != 0:
        raise ValueError("A and B must have the same packed K dimension divisible by 256")
    M, N = A.shape[0], B.shape[0]
    K = A.shape[1] * 2
    if A_fine.shape != (M, K // 32) or B_fine.shape != (N, K // 32):
        raise ValueError("fine scales must have shape [rows, K // 32]")
    if A_coarse.shape != (M, K // COARSE_K) or B_coarse.shape != (N, K // COARSE_K):
        raise ValueError("coarse scales must have shape [rows, K // 512]")

    result = torch.zeros((M, N), device=A.device, dtype=torch.float32)
    fine_groups_per_coarse = COARSE_K // 32
    packed_per_coarse = COARSE_K // 2
    for coarse_idx in range(K // COARSE_K):
        k0 = coarse_idx * packed_per_coarse
        k1 = k0 + packed_per_coarse
        s0 = coarse_idx * fine_groups_per_coarse
        s1 = s0 + fine_groups_per_coarse
        partial = baseline.mxfp4_matmul(
            A[:, k0:k1], B[:, k0:k1],
            baseline.swizzle_scales_packed_block(A_fine[:, s0:s1]),
            baseline.swizzle_scales_packed_block(B_fine[:, s0:s1]),
            out_dtype=torch.float16)
        result += partial.to(torch.float32) * A_coarse[:, coarse_idx, None] * B_coarse[None, :, coarse_idx]
    return result.to(out_dtype)


def run_benchmark():
    print(f"Dual-scale MXFP4 x MXFP4 2CTA GEMM ({BLOCK_M=}, {BLOCK_N=}, {BLOCK_K=}, {NUM_BUFFERS=})")
    print(f"{'MNK':>8}  {'TFLOPS':>12}  {'max abs error':>16}")
    for mnk in (8192, 16384, 32768):
        torch.manual_seed(0)
        A, A_scale, A_coarse, A_ref = random_dualscale_tensor(mnk, mnk)
        B, B_scale, B_coarse, B_ref = random_dualscale_tensor(mnk, mnk)
        A_scale = baseline.swizzle_scales_packed_block(A_scale)
        B_scale = baseline.swizzle_scales_packed_block(B_scale)
        reference = A_ref @ B_ref.T
        output = dualscale_matmul(A, B, A_scale, B_scale, A_coarse, B_coarse)
        atol = 1e-3 + 0.6e-6 * mnk
        torch.testing.assert_close(reference, output.to(torch.float32), atol=atol, rtol=1e-3)
        max_error = (reference - output.to(torch.float32)).abs().max().item()
        ms = triton.testing.do_bench(lambda: dualscale_matmul(A, B, A_scale, B_scale, A_coarse, B_coarse))
        tflops = 2.0 * mnk**3 * 1e-12 / (ms * 1e-3)
        print(f"{mnk:>8}  {tflops:>12.1f}  {max_error:>16.6f}")


if __name__ == "__main__":
    if not baseline.is_blackwell():
        raise RuntimeError("This example requires a Blackwell NVIDIA GPU")
    run_benchmark()
