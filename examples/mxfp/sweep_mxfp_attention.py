"""
Sweep all MXFP Flash Attention variants: accuracy and performance comparison.
==============================================================================

Compares 4 variants against FP16 SDPA ground truth:
1. Pure MXFP8 (E4M3 QK + E4M3 PV)
2. Pure MXFP4 (E2M1 QK + E2M1 PV)
3. MXFP8 QK + MXFP4 PV (recommended)
4. MXFP4 QK + MXFP8 PV (reference)

Accuracy is measured against FP16 SDPA (NOT dequantized inputs), giving true
end-to-end quality loss from quantization.

Usage:
    CUDA_VISIBLE_DEVICES=3 python sweep_mxfp_attention.py [--causal] [--seq-lens 512,1024,2048,4096,8192]
"""

import argparse
import sys
import time
import torch
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ============================================================================
# Import kernels
# ============================================================================

sys.path.insert(0, "/home/yiliu7/workspace/triton-fork/examples/mxfp")

from mxfp8_flash_attention import (
    mxfp8_flash_attention,
    quantize_to_mxfp8,
    dequantize_mxfp8,
)
from mxfp4_flash_attention import (
    mxfp4_flash_attention,
    quantize_to_mxfp4,
    dequantize_mxfp4,
)
from mxfp8_qk_mxfp4_pv_flash_attention import (
    mixed_mxfp8qk_mxfp4pv_flash_attention,
)
from mxfp4_qk_mxfp8_pv_flash_attention import (
    mixed_mxfp4qk_mxfp8pv_flash_attention,
)


# ============================================================================
# Data preparation
# ============================================================================


def prepare_inputs(B, H, N, D):
    """Generate FP16 Q, K, V and quantize to all formats."""
    q_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
    k_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1
    v_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.1

    # FP16 reference
    ref_inputs = (q_f16, k_f16, v_f16)

    # MXFP8: Q, K along HEAD_DIM; V along seq dim
    q_fp8, q_scale_fp8 = quantize_to_mxfp8(q_f16)
    k_fp8, k_scale_fp8 = quantize_to_mxfp8(k_f16)
    v_flat_fp8 = v_f16.permute(0, 1, 3, 2).contiguous().reshape(-1, N)
    v_fp8_flat, v_scale_fp8_flat = quantize_to_mxfp8(v_flat_fp8)
    v_fp8 = v_fp8_flat.reshape(B, H, D, N).permute(0, 1, 3, 2).contiguous()
    v_scale_fp8 = v_scale_fp8_flat.reshape(B, H, D, N // 32)

    # MXFP4: Q, K along HEAD_DIM; V along seq dim (col-major packed)
    q_packed, q_scale_fp4 = quantize_to_mxfp4(q_f16)
    k_packed, k_scale_fp4 = quantize_to_mxfp4(k_f16)
    v_transposed = v_f16.permute(0, 1, 3, 2).contiguous()
    v_flat_fp4 = v_transposed.reshape(-1, N)
    v_packed_flat, v_scale_fp4_flat = quantize_to_mxfp4(v_flat_fp4)
    v_packed = v_packed_flat.reshape(B, H, D, N // 2)
    v_scale_fp4 = v_scale_fp4_flat.reshape(B, H, D, N // 32)

    inputs = {
        "fp16": ref_inputs,
        "mxfp8": (q_fp8, k_fp8, v_fp8, q_scale_fp8, k_scale_fp8, v_scale_fp8),
        "mxfp4": (q_packed, k_packed, v_packed, q_scale_fp4, k_scale_fp4, v_scale_fp4),
        # Mixed: MXFP8 QK + MXFP4 PV
        "mxfp8qk_mxfp4pv": (q_fp8, k_fp8, v_packed, q_scale_fp8, k_scale_fp8, v_scale_fp4),
        # Mixed: MXFP4 QK + MXFP8 PV
        "mxfp4qk_mxfp8pv": (q_packed, k_packed, v_fp8, q_scale_fp4, k_scale_fp4, v_scale_fp8),
    }

    return inputs


def run_variant(name, inputs, causal, sm_scale):
    """Run a single variant and return output tensor."""
    if name == "fp16":
        q, k, v = inputs
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float(), v.float(), is_causal=causal, scale=sm_scale
        )
    elif name == "mxfp8":
        q, k, v, qs, ks, vs = inputs
        return mxfp8_flash_attention(q, k, v, qs, ks, vs, causal=causal, sm_scale=sm_scale)
    elif name == "mxfp4":
        q, k, v, qs, ks, vs = inputs
        return mxfp4_flash_attention(q, k, v, qs, ks, vs, causal=causal, sm_scale=sm_scale)
    elif name == "mxfp8qk_mxfp4pv":
        q, k, v, qs, ks, vs = inputs
        return mixed_mxfp8qk_mxfp4pv_flash_attention(q, k, v, qs, ks, vs, causal=causal, sm_scale=sm_scale)
    elif name == "mxfp4qk_mxfp8pv":
        q, k, v, qs, ks, vs = inputs
        return mixed_mxfp4qk_mxfp8pv_flash_attention(q, k, v, qs, ks, vs, causal=causal, sm_scale=sm_scale)


# ============================================================================
# Metrics
# ============================================================================


def compute_metrics(ref, out):
    """Compute accuracy metrics between reference and output."""
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()

    max_diff = (ref_flat - out_flat).abs().max().item()
    mean_diff = (ref_flat - out_flat).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), out_flat.unsqueeze(0)
    ).item()

    # SNR in dB
    signal_power = (ref_flat ** 2).mean().item()
    noise_power = ((ref_flat - out_flat) ** 2).mean().item()
    snr_db = 10 * torch.log10(torch.tensor(signal_power / max(noise_power, 1e-20))).item()

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
        "snr_db": snr_db,
    }


def benchmark_variant(name, inputs, causal, sm_scale, warmup=5, iters=20):
    """Benchmark a variant and return time in ms."""
    # Warmup
    for _ in range(warmup):
        run_variant(name, inputs, causal, sm_scale)
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        run_variant(name, inputs, causal, sm_scale)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / iters * 1000

    return elapsed_ms


# ============================================================================
# Sweep
# ============================================================================

VARIANT_NAMES = {
    "mxfp8": "Pure MXFP8",
    "mxfp4": "Pure MXFP4",
    "mxfp8qk_mxfp4pv": "MXFP8 QK + MXFP4 PV",
    "mxfp4qk_mxfp8pv": "MXFP4 QK + MXFP8 PV",
}

VARIANTS = ["mxfp8", "mxfp4", "mxfp8qk_mxfp4pv", "mxfp4qk_mxfp8pv"]


def run_accuracy_sweep(configs, causal):
    """Run accuracy sweep across all configs and variants."""
    print()
    print("=" * 90)
    print(f"  ACCURACY SWEEP (vs FP16 SDPA)  {'[CAUSAL]' if causal else '[NON-CAUSAL]'}")
    print("=" * 90)

    # Header
    print(f"\n{'Config':<25} {'Variant':<25} {'cos_sim':<12} {'SNR(dB)':<10} {'max_diff':<12} {'mean_diff':<12}")
    print("-" * 90)

    for B, H, N, D in configs:
        torch.manual_seed(42)
        sm_scale = 1.0 / (D ** 0.5)
        inputs = prepare_inputs(B, H, N, D)

        # FP16 reference
        ref = run_variant("fp16", inputs["fp16"], causal, sm_scale)

        config_str = f"B={B} H={H} N={N} D={D}"

        for variant in VARIANTS:
            out = run_variant(variant, inputs[variant], causal, sm_scale)
            metrics = compute_metrics(ref, out)

            print(f"  {config_str:<23} {VARIANT_NAMES[variant]:<25} "
                  f"{metrics['cos_sim']:<12.6f} {metrics['snr_db']:<10.1f} "
                  f"{metrics['max_diff']:<12.4f} {metrics['mean_diff']:<12.6f}")

            config_str = ""  # Only print config once per group

        print()


def run_perf_sweep(configs, causal):
    """Run performance sweep across all configs and variants."""
    print()
    print("=" * 90)
    print(f"  PERFORMANCE SWEEP  {'[CAUSAL]' if causal else '[NON-CAUSAL]'}")
    print("=" * 90)

    # Header
    print(f"\n{'Config':<25} {'Variant':<25} {'Time(ms)':<12} {'TFLOPS':<10} {'Speedup':<10}")
    print("-" * 90)

    for B, H, N, D in configs:
        torch.manual_seed(42)
        sm_scale = 1.0 / (D ** 0.5)
        inputs = prepare_inputs(B, H, N, D)

        flops = 4 * B * H * N * N * D
        config_str = f"B={B} H={H} N={N} D={D}"

        times = {}
        for variant in VARIANTS:
            elapsed_ms = benchmark_variant(variant, inputs[variant], causal, sm_scale)
            tflops = flops / (elapsed_ms / 1000) / 1e12
            times[variant] = elapsed_ms

            # Speedup relative to pure MXFP8
            speedup = times.get("mxfp8", elapsed_ms) / elapsed_ms

            print(f"  {config_str:<23} {VARIANT_NAMES[variant]:<25} "
                  f"{elapsed_ms:<12.2f} {tflops:<10.1f} {speedup:<10.2f}x")

            config_str = ""

        print()


def run_combined_summary(configs, causal):
    """Print a combined summary table for easy comparison."""
    print()
    print("=" * 90)
    print(f"  COMBINED SUMMARY  {'[CAUSAL]' if causal else '[NON-CAUSAL]'}")
    print("=" * 90)
    print()
    print(f"  {'Variant':<25} | {'Avg cos_sim':<14} | {'Avg SNR(dB)':<14} | {'Avg TFLOPS':<12} | {'vs MXFP8':<10}")
    print(f"  {'-'*25}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}-+-{'-'*10}")

    all_metrics = {v: {"cos_sim": [], "snr_db": [], "tflops": []} for v in VARIANTS}

    for B, H, N, D in configs:
        torch.manual_seed(42)
        sm_scale = 1.0 / (D ** 0.5)
        inputs = prepare_inputs(B, H, N, D)
        flops = 4 * B * H * N * N * D

        ref = run_variant("fp16", inputs["fp16"], causal, sm_scale)

        for variant in VARIANTS:
            out = run_variant(variant, inputs[variant], causal, sm_scale)
            metrics = compute_metrics(ref, out)
            all_metrics[variant]["cos_sim"].append(metrics["cos_sim"])
            all_metrics[variant]["snr_db"].append(metrics["snr_db"])

            elapsed_ms = benchmark_variant(variant, inputs[variant], causal, sm_scale, warmup=3, iters=10)
            tflops = flops / (elapsed_ms / 1000) / 1e12
            all_metrics[variant]["tflops"].append(tflops)

    for variant in VARIANTS:
        avg_cos = sum(all_metrics[variant]["cos_sim"]) / len(all_metrics[variant]["cos_sim"])
        avg_snr = sum(all_metrics[variant]["snr_db"]) / len(all_metrics[variant]["snr_db"])
        avg_tflops = sum(all_metrics[variant]["tflops"]) / len(all_metrics[variant]["tflops"])
        avg_mxfp8_tflops = sum(all_metrics["mxfp8"]["tflops"]) / len(all_metrics["mxfp8"]["tflops"])
        speedup = avg_tflops / avg_mxfp8_tflops

        marker = " <--" if variant == "mxfp8qk_mxfp4pv" else ""
        print(f"  {VARIANT_NAMES[variant]:<25} | {avg_cos:<14.6f} | {avg_snr:<14.1f} | {avg_tflops:<12.1f} | {speedup:<10.2f}x{marker}")

    print()
    print("  NOTE: cos_sim and SNR are measured against FP16 SDPA (true end-to-end quality).")
    print("  Higher cos_sim / SNR = better accuracy. Higher TFLOPS = faster.")
    print()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Sweep MXFP attention variants")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--seq-lens", type=str, default="512,1024,2048,4096,8192",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["accuracy", "perf", "summary", "all"],
                        help="Which sweep to run")
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    configs = [(args.batch, args.heads, N, args.head_dim) for N in seq_lens]

    print(f"Device: {DEVICE}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print(f"Configs: B={args.batch}, H={args.heads}, D={args.head_dim}, "
          f"N={seq_lens}, causal={args.causal}")

    if args.mode in ("accuracy", "all"):
        run_accuracy_sweep(configs, args.causal)

    if args.mode in ("perf", "all"):
        run_perf_sweep(configs, args.causal)

    if args.mode in ("summary", "all"):
        run_combined_summary(configs, args.causal)


if __name__ == "__main__":
    main()
