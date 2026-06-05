"""
Mixed-Precision Flash Attention: MXFP4 QK + MXFP8 PV on NVIDIA B200 (sm_100)
==============================================================================

Combines:
- QK GEMM: E2M1 (MXFP4) — 2x arithmetic throughput for logit computation
- PV GEMM: E4M3 (MXFP8) — preserves value aggregation fidelity

This is the **opposite** of the recommended configuration (MXFP8 QK + MXFP4 PV),
implemented as a reference for accuracy comparison.

Data formats:
- Q, K: E2M1 packed uint8 + E8M0 scales (group_size=32 along HEAD_DIM)
- V: float8_e4m3fn + E8M0 scales (group_size=32 along sequence dim N)
- P: quantized on-the-fly to E4M3 with E8M0 scales

The kernel computes:
    O = softmax(Q @ K^T * sm_scale) @ V
"""

import torch
import triton
import triton.language as tl
import time

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ============================================================================
# Helper Functions
# ============================================================================


def fp8e8m0_to_float32(scale_uint8: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 scale (uint8) to float32: value = 2^(x - 127)."""
    x = scale_uint8.to(torch.int32)
    x = x << 23
    return x.view(torch.float32)


def quantize_to_mxfp8(x: torch.Tensor, group_size: int = 32) -> tuple:
    """Quantize float tensor to MXFP8 E4M3 with E8M0 block scales."""
    assert x.shape[-1] % group_size == 0

    original_shape = x.shape
    x_f32 = x.float()

    *batch_dims, K = x_f32.shape
    num_groups = K // group_size
    x_grouped = x_f32.reshape(*batch_dims, num_groups, group_size)

    amax = x_grouped.abs().amax(dim=-1)
    e4m3_max = 448.0

    amax_safe = amax.clamp(min=1e-12)
    log2_scale = torch.ceil(torch.log2(amax_safe / e4m3_max))
    e8m0_encoding = (log2_scale + 127).clamp(0, 254).to(torch.uint8)

    scale_f32 = fp8e8m0_to_float32(e8m0_encoding)
    scale_expanded = scale_f32.unsqueeze(-1).expand_as(x_grouped)
    scale_expanded = scale_expanded.reshape(original_shape)

    x_scaled = x_f32 / scale_expanded
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)

    return x_fp8, e8m0_encoding


def dequantize_mxfp8(x_fp8: torch.Tensor, scales: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Dequantize MXFP8 E4M3 tensor back to float32."""
    x_f32 = x_fp8.to(torch.float32)
    scale_f32 = fp8e8m0_to_float32(scales)
    scale_expanded = scale_f32.repeat_interleave(group_size, dim=-1)
    scale_expanded = scale_expanded[..., :x_f32.shape[-1]]
    return x_f32 * scale_expanded


# E2M1 representable positive values:
_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _float_to_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    """Convert float tensor to E2M1 4-bit encoding (uint8, values 0-15)."""
    sign = (x < 0).to(torch.uint8) << 3
    ax = x.abs().clamp(max=6.0)

    code = torch.zeros_like(ax, dtype=torch.uint8)
    code = torch.where(ax >= 0.25, torch.ones_like(code), code)
    code = torch.where(ax >= 0.75, torch.full_like(code, 2), code)
    code = torch.where(ax >= 1.25, torch.full_like(code, 3), code)
    code = torch.where(ax >= 1.75, torch.full_like(code, 4), code)
    code = torch.where(ax >= 2.5, torch.full_like(code, 5), code)
    code = torch.where(ax >= 3.5, torch.full_like(code, 6), code)
    code = torch.where(ax >= 5.0, torch.full_like(code, 7), code)

    return sign | code


def _e2m1_nibble_to_float(nibble: torch.Tensor) -> torch.Tensor:
    """Convert E2M1 4-bit encoding back to float32."""
    sign = ((nibble >> 3) & 1).float() * -2.0 + 1.0
    mag_code = (nibble & 0x7).long()
    values = _E2M1_VALUES.to(nibble.device)
    mag = values[mag_code]
    return sign * mag


def quantize_to_mxfp4(x: torch.Tensor, group_size: int = 32) -> tuple:
    """Quantize float tensor to MXFP4 (E2M1) with E8M0 block scales."""
    assert x.shape[-1] % group_size == 0
    assert x.shape[-1] % 2 == 0

    original_shape = x.shape
    x_f32 = x.float()

    *batch_dims, K = x_f32.shape
    num_groups = K // group_size
    x_grouped = x_f32.reshape(*batch_dims, num_groups, group_size)

    amax = x_grouped.abs().amax(dim=-1)
    e2m1_max = 6.0

    amax_safe = amax.clamp(min=e2m1_max * (2**-126))
    log2_ratio = torch.ceil(torch.log2(amax_safe / e2m1_max))
    log2_ratio = log2_ratio.clamp(-127, 127)
    e8m0_encoding = (log2_ratio + 127).to(torch.uint8)

    scale_f32 = fp8e8m0_to_float32(e8m0_encoding)
    scale_expanded = scale_f32.unsqueeze(-1).expand_as(x_grouped)
    x_scaled = x_grouped / scale_expanded
    x_scaled = x_scaled.reshape(original_shape)

    nibbles = _float_to_e2m1_nibble(x_scaled)

    even = nibbles[..., 0::2]
    odd = nibbles[..., 1::2]
    packed = (odd << 4) | even

    return packed, e8m0_encoding


def dequantize_mxfp4(x_packed: torch.Tensor, scales: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Dequantize MXFP4 packed tensor back to float32."""
    even = x_packed & 0x0F
    odd = (x_packed >> 4) & 0x0F

    *batch_dims, K_half = even.shape
    K = K_half * 2
    full = torch.zeros(*batch_dims, K, dtype=torch.uint8, device=x_packed.device)
    full[..., 0::2] = even
    full[..., 1::2] = odd

    x_f32 = _e2m1_nibble_to_float(full)

    scale_f32 = fp8e8m0_to_float32(scales)
    scale_expanded = scale_f32.repeat_interleave(group_size, dim=-1)
    scale_expanded = scale_expanded[..., :K]
    return x_f32 * scale_expanded


# ============================================================================
# Triton Kernel
# ============================================================================


@triton.jit
def _mixed_attn_fwd_inner(
    acc, l_i, m_i, q, q_scale,  #
    K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,  #
    stride_kn, stride_kk,  #
    stride_ks_n, stride_ks_k,  #
    stride_vn, stride_vk,  #
    stride_vs_d, stride_vs_n,  #
    start_m, qk_scale,  #
    offs_m, offs_n,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):
    # Determine loop bounds based on causal stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  # STAGE == 3: non-causal, full range
        lo, hi = 0, N_CTX

    HEAD_DIM_PACKED: tl.constexpr = HEAD_DIM // 2

    offs_k_head_packed = tl.arange(0, HEAD_DIM_PACKED)
    offs_k_n = tl.arange(0, BLOCK_N)
    offs_scale_k = tl.arange(0, HEAD_DIM // 32)
    offs_scale_n = tl.arange(0, BLOCK_N // 32)
    offs_head_full = tl.arange(0, HEAD_DIM)

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # ============================================================
        # QK GEMM: MXFP4 (E2M1)
        # ============================================================

        # Load K [BLOCK_N, HEAD_DIM//2] packed e2m1
        k_ptrs = K_ptr + (start_n + offs_k_n[:, None]) * stride_kn + offs_k_head_packed[None, :] * stride_kk
        k = tl.load(k_ptrs)  # [BLOCK_N, HEAD_DIM//2]

        # Load K scale [BLOCK_N, HEAD_DIM // 32]
        ks_ptrs = K_scale_ptr + (start_n + offs_k_n[:, None]) * stride_ks_n + offs_scale_k[None, :] * stride_ks_k
        k_scale = tl.load(ks_ptrs)  # [BLOCK_N, HEAD_DIM // 32]

        # Q @ K^T via dot_scaled e2m1
        qk = tl.dot_scaled(q, q_scale, "e2m1", tl.trans(k), k_scale, "e2m1")

        # ============================================================
        # Online Softmax
        # ============================================================

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij

        # ============================================================
        # Quantize P to MXFP8 (E4M3) on the fly
        # ============================================================

        # Compute per-group-of-32 amax along BLOCK_N
        p_reshaped = tl.reshape(p, [BLOCK_M, BLOCK_N // 32, 32])
        p_amax = tl.max(p_reshaped, 2)  # [BLOCK_M, BLOCK_N // 32]

        # E8M0 scale: encoding = ceil(log2(amax)) + 127
        p_amax_safe = tl.maximum(p_amax, 1e-12)
        p_log2 = tl.math.ceil(tl.math.log2(p_amax_safe))
        p_e8m0 = tl.maximum(p_log2 + 127, 0.0).to(tl.uint8)  # [BLOCK_M, BLOCK_N // 32]

        # Compute float scale for quantizing P
        p_scale_f32 = tl.math.exp2((p_e8m0.to(tl.float32) - 127.0))
        # Expand scale to [BLOCK_M, BLOCK_N]
        p_scale_expanded = tl.reshape(
            tl.broadcast_to(p_scale_f32[:, :, None], [BLOCK_M, BLOCK_N // 32, 32]),
            [BLOCK_M, BLOCK_N]
        )
        # Quantize P to fp8
        p_scaled = p / p_scale_expanded
        p_fp8 = p_scaled.to(tl.float8e4nv)
        p_scale = p_e8m0  # [BLOCK_M, BLOCK_N // 32]

        # ============================================================
        # PV GEMM: MXFP8 (E4M3)
        # ============================================================

        # Load V [BLOCK_N, HEAD_DIM] as float8_e4m3fn (row-major, no transpose)
        v_ptrs = V_ptr + (start_n + offs_k_n[:, None]) * stride_vn + offs_head_full[None, :] * stride_vk
        v = tl.load(v_ptrs)  # [BLOCK_N, HEAD_DIM]

        # Load V scale [HEAD_DIM, BLOCK_N // 32]
        vs_ptrs = V_scale_ptr + offs_head_full[:, None] * stride_vs_d + (start_n // 32 + offs_scale_n[None, :]) * stride_vs_n
        v_scale = tl.load(vs_ptrs)  # [HEAD_DIM, BLOCK_N // 32]

        # P @ V via dot_scaled e4m3
        acc = tl.dot_scaled(p_fp8, p_scale, "e4m3", v, v_scale, "e4m3", acc)

        m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def _mixed_attn_fwd(
    Q, K, V, Out,  #
    Q_scale, K_scale, V_scale,  #
    sm_scale,  #
    stride_qz, stride_qh, stride_qm, stride_qk,  #
    stride_kz, stride_kh, stride_kn, stride_kk,  #
    stride_vz, stride_vh, stride_vn, stride_vk,  #
    stride_oz, stride_oh, stride_om, stride_ok,  #
    stride_qsz, stride_qsh, stride_qsm, stride_qsk,  #
    stride_ksz, stride_ksh, stride_ksn, stride_ksk,  #
    stride_vsz, stride_vsh, stride_vsd, stride_vsn,  #
    Z, H, N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Base pointers
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    qs_offset = off_z * stride_qsz + off_h * stride_qsh
    ks_offset = off_z * stride_ksz + off_h * stride_ksh
    vs_offset = off_z * stride_vsz + off_h * stride_vsh

    Q_ptr = Q + q_offset
    K_ptr = K + k_offset
    V_ptr = V + v_offset
    Q_scale_ptr = Q_scale + qs_offset
    K_scale_ptr = K_scale + ks_offset
    V_scale_ptr = V_scale + vs_offset

    HEAD_DIM_PACKED: tl.constexpr = HEAD_DIM // 2

    # Load Q block [BLOCK_M, HEAD_DIM//2] packed e2m1
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_head_packed = tl.arange(0, HEAD_DIM_PACKED)
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_head_packed[None, :] * stride_qk
    q = tl.load(q_ptrs)  # [BLOCK_M, HEAD_DIM//2]

    # Load Q scale [BLOCK_M, HEAD_DIM // 32]
    offs_scale_k = tl.arange(0, HEAD_DIM // 32)
    qs_ptrs = Q_scale_ptr + offs_m[:, None] * stride_qsm + offs_scale_k[None, :] * stride_qsk
    q_scale = tl.load(qs_ptrs)  # [BLOCK_M, HEAD_DIM // 32]

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Scale with log2(e) for exp2-based softmax
    qk_scale = sm_scale * 1.44269504

    offs_n = tl.arange(0, BLOCK_N)

    # Run attention inner loop
    if STAGE & 1:
        acc, l_i, m_i = _mixed_attn_fwd_inner(
            acc, l_i, m_i, q, q_scale,
            K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
            stride_kn, stride_kk,
            stride_ksn, stride_ksk,
            stride_vn, stride_vk,
            stride_vsd, stride_vsn,
            start_m, qk_scale,
            offs_m, offs_n,
            N_CTX, BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE,
        )
    if STAGE & 2:
        acc, l_i, m_i = _mixed_attn_fwd_inner(
            acc, l_i, m_i, q, q_scale,
            K_ptr, K_scale_ptr, V_ptr, V_scale_ptr,
            stride_kn, stride_kk,
            stride_ksn, stride_ksk,
            stride_vn, stride_vk,
            stride_vsd, stride_vsn,
            start_m, qk_scale,
            offs_m, offs_n,
            N_CTX, BLOCK_M, HEAD_DIM, BLOCK_N,
            2,
        )

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    offs_head = tl.arange(0, HEAD_DIM)
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_head[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty))


# ============================================================================
# Wrapper Function
# ============================================================================


def mixed_mxfp4qk_mxfp8pv_flash_attention(
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Mixed-Precision Flash Attention: MXFP4 QK + MXFP8 PV.

    Args:
        q_packed: [B, H, M, D//2] uint8 — packed E2M1 queries (MXFP4, along HEAD_DIM)
        k_packed: [B, H, N, D//2] uint8 — packed E2M1 keys (MXFP4, along HEAD_DIM)
        v: [B, H, N, D] float8_e4m3fn — values (MXFP8)
        q_scale: [B, H, M, D//32] uint8 — E8M0 scales for Q
        k_scale: [B, H, N, D//32] uint8 — E8M0 scales for K
        v_scale: [B, H, D, N//32] uint8 — E8M0 scales for V (groups along seq dim)
        causal: whether to apply causal mask
        sm_scale: softmax scale (default: 1/sqrt(HEAD_DIM))

    Returns:
        output: [B, H, M, D] float32
    """
    B, H, M, D_packed = q_packed.shape
    D = D_packed * 2  # HEAD_DIM
    N = k_packed.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64

    assert D in (64, 128), f"HEAD_DIM must be 64 or 128, got {D}"
    assert M % BLOCK_M == 0, f"M={M} must be divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be divisible by BLOCK_N={BLOCK_N}"

    output = torch.empty((B, H, M, D), dtype=torch.float32, device=q_packed.device)

    STAGE = 3 if causal else 1

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    _mixed_attn_fwd[grid](
        q_packed, k_packed, v, output,
        q_scale, k_scale, v_scale,
        sm_scale,
        # Q strides [B, H, M, D//2]
        q_packed.stride(0), q_packed.stride(1), q_packed.stride(2), q_packed.stride(3),
        # K strides [B, H, N, D//2]
        k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
        # V strides [B, H, N, D]
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # O strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Q_scale strides [B, H, M, D//32]
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2), q_scale.stride(3),
        # K_scale strides [B, H, N, D//32]
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), k_scale.stride(3),
        # V_scale strides [B, H, D, N//32]
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2), v_scale.stride(3),
        # Dimensions
        B, H, N,
        # Compile-time constants
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=STAGE,
        num_warps=4,
        num_stages=3,
    )

    return output


# ============================================================================
# Test: Correctness vs SDPA
# ============================================================================


def test_mixed_flash_attention():
    """Test mixed-precision flash attention against torch SDPA on dequantized inputs."""
    print("=" * 70)
    print("Testing Mixed Flash Attention (MXFP4 QK + MXFP8 PV) vs SDPA")
    print("=" * 70)

    torch.manual_seed(42)

    configs = [
        # (B, H, N, D, causal)
        (1, 1, 256, 128, False),
        (1, 4, 512, 128, False),
        (2, 4, 1024, 128, False),
        (2, 4, 1024, 128, True),
        (1, 8, 2048, 128, False),
    ]

    for B, H, N, D, causal in configs:
        # Generate random fp16 data
        q_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
        k_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
        v_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1

        # Quantize Q, K to MXFP4 (along HEAD_DIM)
        q_packed, q_scale = quantize_to_mxfp4(q_f16)
        k_packed, k_scale = quantize_to_mxfp4(k_f16)

        # Quantize V to MXFP8 (along sequence dim N)
        # V_scale shape: [B, H, D, N//32] — groups of 32 along sequence
        v_flat = v_f16.permute(0, 1, 3, 2).contiguous()  # [B, H, D, N]
        v_flat_2d = v_flat.reshape(-1, N)
        v_fp8_flat, v_scale_flat = quantize_to_mxfp8(v_flat_2d)  # scale along N
        v_fp8_transposed = v_fp8_flat.reshape(B, H, D, N)  # [B, H, D, N]
        v_scale = v_scale_flat.reshape(B, H, D, N // 32)  # [B, H, D, N//32]
        # V data for kernel: [B, H, N, D] (standard layout)
        v_fp8 = v_fp8_transposed.permute(0, 1, 3, 2).contiguous()  # [B, H, N, D]

        # Dequantize for reference
        q_deq = dequantize_mxfp4(q_packed, q_scale).float()
        k_deq = dequantize_mxfp4(k_packed, k_scale).float()
        v_deq_t = dequantize_mxfp8(v_fp8_transposed.reshape(-1, N),
                                     v_scale_flat).reshape(B, H, D, N)
        v_deq = v_deq_t.permute(0, 1, 3, 2).contiguous().float()

        # Reference: torch SDPA
        sm_scale = 1.0 / (D ** 0.5)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q_deq, k_deq, v_deq, is_causal=causal, scale=sm_scale
        )

        # Our kernel
        out = mixed_mxfp4qk_mxfp8pv_flash_attention(
            q_packed, k_packed, v_fp8, q_scale, k_scale, v_scale,
            causal=causal, sm_scale=sm_scale
        )

        # Compare
        max_diff = (ref - out).abs().max().item()
        mean_diff = (ref - out).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0), out.flatten().unsqueeze(0)
        ).item()

        threshold = 1.0 if causal else 0.5
        status = "PASS" if max_diff < threshold else "FAIL"
        print(f"  [{status}] B={B}, H={H}, N={N}, D={D}, causal={causal}: "
              f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}")

        if status == "FAIL":
            print(f"    WARNING: Large difference detected!")

    print()


# ============================================================================
# Benchmark
# ============================================================================


def benchmark_mixed_flash_attention():
    """Benchmark mixed-precision flash attention and report TFLOPS."""
    print("=" * 70)
    print("Benchmarking Mixed Flash Attention (MXFP4 QK + MXFP8 PV)")
    print("=" * 70)

    torch.manual_seed(42)

    configs = [
        # (B, H, N, D)
        (4, 32, 2048, 128),
        (2, 32, 4096, 128),
        (1, 32, 8192, 128),
        (1, 40, 64 * 1024, 128),
    ]

    for B, H, N, D in configs:
        q_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
        k_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
        v_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1

        q_packed, q_scale = quantize_to_mxfp4(q_f16)
        k_packed, k_scale = quantize_to_mxfp4(k_f16)

        v_flat = v_f16.permute(0, 1, 3, 2).contiguous().reshape(-1, N)
        v_fp8_flat, v_scale_flat = quantize_to_mxfp8(v_flat)
        v_fp8 = v_fp8_flat.reshape(B, H, D, N).permute(0, 1, 3, 2).contiguous()
        v_scale = v_scale_flat.reshape(B, H, D, N // 32)

        sm_scale = 1.0 / (D ** 0.5)

        # Warmup
        for _ in range(5):
            out = mixed_mxfp4qk_mxfp8pv_flash_attention(
                q_packed, k_packed, v_fp8, q_scale, k_scale, v_scale,
                causal=False, sm_scale=sm_scale
            )
        torch.cuda.synchronize()

        # Benchmark
        num_iters = 20
        start = time.perf_counter()
        for _ in range(num_iters):
            out = mixed_mxfp4qk_mxfp8pv_flash_attention(
                q_packed, k_packed, v_fp8, q_scale, k_scale, v_scale,
                causal=False, sm_scale=sm_scale
            )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters

        flops = 4 * B * H * N * N * D
        tflops = flops / elapsed / 1e12

        print(f"  B={B}, H={H}, N={N}, D={D}: {elapsed*1000:.2f} ms, {tflops:.1f} TFLOPS")

    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print()

    test_mixed_flash_attention()
    benchmark_mixed_flash_attention()
