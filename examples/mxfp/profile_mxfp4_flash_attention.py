"""
Detailed profiling of MXFP4 Flash Attention kernel using:
1. Triton Proton - built-in Triton profiler for kernel-level metrics
2. NVIDIA Nsight Compute (ncu) - hardware counter profiling

Usage:
    # Proton profiling (generates proton trace):
    python profile_mxfp4_flash_attention.py --proton

    # Generate ncu command (run separately with sudo/root):
    python profile_mxfp4_flash_attention.py --ncu

    # Run ncu profiling directly:
    ncu --set full -o mxfp4_fa_profile python profile_mxfp4_flash_attention.py --ncu-target

    # Both:
    python profile_mxfp4_flash_attention.py --proton --ncu
"""

import argparse
import os
import sys
import torch
import triton
import triton.language as tl
import time

# Import from the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp4_flash_attention import (
    quantize_to_mxfp4,
    mxfp4_flash_attention,
    _mxfp4_attn_fwd,
)

DEVICE = torch.device('cuda')


def prepare_inputs(B=2, H=32, N=4096, D=128):
    """Prepare MXFP4 inputs for profiling."""
    torch.manual_seed(42)

    q_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.05
    k_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.05
    v_f16 = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16) * 0.05

    q_packed, q_scale = quantize_to_mxfp4(q_f16)
    k_packed, k_scale = quantize_to_mxfp4(k_f16)

    v_transposed = v_f16.permute(0, 1, 3, 2).contiguous()
    v_flat = v_transposed.reshape(-1, N)
    v_packed_flat, v_scale_flat = quantize_to_mxfp4(v_flat)
    v_packed = v_packed_flat.reshape(B, H, D, N // 2)
    v_scale = v_scale_flat.reshape(B, H, D, N // 32)

    sm_scale = 1.0 / (D ** 0.5)

    return q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, sm_scale


def profile_with_proton(num_warmup=5, num_iters=20):
    """Profile using Triton Proton - captures kernel execution metrics."""
    import triton.profiler as proton

    print("=" * 70)
    print("Profiling MXFP4 Flash Attention with Triton Proton")
    print("=" * 70)

    configs = [
        (2, 32, 4096, 128, False, "B2_H32_N4096_D128_noncausal"),
        (2, 32, 4096, 128, True, "B2_H32_N4096_D128_causal"),
        (1, 32, 8192, 128, False, "B1_H32_N8192_D128_noncausal"),
        (1, 40, 16384, 128, False, "B1_H40_N16384_D128_noncausal"),
    ]

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proton_profiles")
    os.makedirs(output_dir, exist_ok=True)

    for B, H, N, D, causal, name in configs:
        print(f"\n--- Config: {name} ---")
        q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, sm_scale = prepare_inputs(B, H, N, D)

        # Warmup
        for _ in range(num_warmup):
            out = mxfp4_flash_attention(q_packed, k_packed, v_packed,
                                         q_scale, k_scale, v_scale,
                                         causal=causal, sm_scale=sm_scale)
        torch.cuda.synchronize()

        # Profile with Proton
        profile_path = os.path.join(output_dir, f"mxfp4_fa_{name}")
        session_id = proton.start(profile_path)

        # Use proton context/scope for fine-grained annotation
        proton.enter_scope(f"mxfp4_flash_attn_{name}")
        for i in range(num_iters):
            proton.enter_scope(f"iter_{i}")
            out = mxfp4_flash_attention(q_packed, k_packed, v_packed,
                                         q_scale, k_scale, v_scale,
                                         causal=causal, sm_scale=sm_scale)
            proton.exit_scope()
        torch.cuda.synchronize()
        proton.exit_scope()

        proton.finalize()

        # Compute expected metrics
        flops = 4 * B * H * N * N * D
        elapsed_for_est = None

        # Time separately for TFLOPS estimation
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            out = mxfp4_flash_attention(q_packed, k_packed, v_packed,
                                         q_scale, k_scale, v_scale,
                                         causal=causal, sm_scale=sm_scale)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iters

        tflops = flops / elapsed / 1e12
        # Memory bandwidth estimation
        # Reads: Q + K + V (packed) + scales
        q_bytes = B * H * N * (D // 2)  # packed
        k_bytes = B * H * N * (D // 2)
        v_bytes = B * H * D * (N // 2)
        scale_bytes = B * H * N * (D // 32) * 2 + B * H * D * (N // 32)  # q_scale + k_scale + v_scale
        o_bytes = B * H * N * D * 4  # float32 output
        total_bytes = q_bytes + k_bytes + v_bytes + scale_bytes + o_bytes
        bandwidth_tb = total_bytes / elapsed / 1e12

        print(f"  Time:       {elapsed * 1000:.3f} ms")
        print(f"  TFLOPS:     {tflops:.1f}")
        print(f"  Bandwidth:  {bandwidth_tb * 1000:.1f} GB/s")
        print(f"  Data moved: {total_bytes / 1e6:.1f} MB")
        print(f"  Proton profile saved to: {profile_path}.hatchet")

    print(f"\n\nProton profiles saved in: {output_dir}/")
    print("View with: proton-viewer <profile>.hatchet")
    print("Or convert: python -m triton.profiler.viewer -i <profile>.hatchet")


def profile_with_ncu_target(num_iters=3):
    """Target function when running under ncu. Runs kernel with minimal overhead."""
    print("Running MXFP4 Flash Attention as ncu target...")

    configs = [
        (2, 32, 4096, 128, False, "noncausal_4096"),
        (2, 32, 4096, 128, True, "causal_4096"),
    ]

    for B, H, N, D, causal, name in configs:
        q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, sm_scale = prepare_inputs(B, H, N, D)

        # Warmup (outside ncu capture ideally, but needed for JIT)
        out = mxfp4_flash_attention(q_packed, k_packed, v_packed,
                                     q_scale, k_scale, v_scale,
                                     causal=causal, sm_scale=sm_scale)
        torch.cuda.synchronize()

        # Profiled iterations
        # ncu will capture these kernel launches
        for _ in range(num_iters):
            out = mxfp4_flash_attention(q_packed, k_packed, v_packed,
                                         q_scale, k_scale, v_scale,
                                         causal=causal, sm_scale=sm_scale)
        torch.cuda.synchronize()
        print(f"  [{name}] Done - {num_iters} iterations")


def generate_ncu_commands():
    """Generate ncu profiling commands for different detail levels."""
    print("=" * 70)
    print("NCU Profiling Commands for MXFP4 Flash Attention")
    print("=" * 70)

    script_path = os.path.abspath(__file__)
    output_dir = os.path.join(os.path.dirname(script_path), "ncu_profiles")

    print(f"\nOutput directory: {output_dir}")
    print(f"Create it: mkdir -p {output_dir}\n")

    commands = [
        {
            "name": "Quick overview (roofline + summary)",
            "cmd": (
                f"ncu --set roofline "
                f"--target-processes all "
                f"--kernel-name '_mxfp4_attn_fwd' "
                f"--launch-skip 1 --launch-count 3 "
                f"-o {output_dir}/mxfp4_fa_roofline "
                f"python {script_path} --ncu-target"
            ),
        },
        {
            "name": "Full analysis (all metrics)",
            "cmd": (
                f"ncu --set full "
                f"--target-processes all "
                f"--kernel-name '_mxfp4_attn_fwd' "
                f"--launch-skip 1 --launch-count 3 "
                f"-o {output_dir}/mxfp4_fa_full "
                f"python {script_path} --ncu-target"
            ),
        },
        {
            "name": "Memory analysis (bandwidth, L2, shared memory)",
            "cmd": (
                f"ncu --set memory "
                f"--target-processes all "
                f"--kernel-name '_mxfp4_attn_fwd' "
                f"--launch-skip 1 --launch-count 3 "
                f"-o {output_dir}/mxfp4_fa_memory "
                f"python {script_path} --ncu-target"
            ),
        },
        {
            "name": "Compute analysis (instruction mix, warp efficiency)",
            "cmd": (
                f"ncu --set compute "
                f"--target-processes all "
                f"--kernel-name '_mxfp4_attn_fwd' "
                f"--launch-skip 1 --launch-count 3 "
                f"-o {output_dir}/mxfp4_fa_compute "
                f"python {script_path} --ncu-target"
            ),
        },
        {
            "name": "Specific metrics (occupancy, MMA utilization, stalls)",
            "cmd": (
                f"ncu "
                f"--metrics "
                f"sm__throughput.avg.pct_of_peak_sustained_elapsed,"
                f"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,"
                f"l1tex__throughput.avg.pct_of_peak_sustained_elapsed,"
                f"lts__throughput.avg.pct_of_peak_sustained_elapsed,"
                f"sm__warps_active.avg.pct_of_peak_sustained_elapsed,"
                f"smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,"
                f"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
                f"smsp__inst_executed_pipe_tensor.sum,"
                f"smsp__inst_executed.sum,"
                f"dram__bytes_read.sum,"
                f"dram__bytes_write.sum,"
                f"lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,"
                f"sm__sass_thread_inst_executed_op_dadd_pred_on.sum,"
                f"sm__sass_thread_inst_executed_op_dfma_pred_on.sum "
                f"--target-processes all "
                f"--kernel-name '_mxfp4_attn_fwd' "
                f"--launch-skip 1 --launch-count 3 "
                f"-o {output_dir}/mxfp4_fa_metrics "
                f"python {script_path} --ncu-target"
            ),
        },
    ]

    for i, item in enumerate(commands, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}] {item['name']}")
        print(f"{'─' * 60}")
        print(f"  {item['cmd']}")

    print(f"\n\n{'=' * 70}")
    print("After profiling, view results:")
    print("  ncu-ui <output>.ncu-rep      # GUI viewer")
    print("  ncu --import <output>.ncu-rep # CLI summary")
    print("=" * 70)


def profile_with_torch_profiler(num_warmup=5, num_iters=10):
    """Profile using PyTorch profiler with CUDA activity tracing."""
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    print("=" * 70)
    print("Profiling MXFP4 Flash Attention with PyTorch Profiler")
    print("=" * 70)

    B, H, N, D = 2, 32, 4096, 128
    q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, sm_scale = prepare_inputs(B, H, N, D)

    # Warmup
    for _ in range(num_warmup):
        out = mxfp4_flash_attention(q_packed, k_packed, v_packed,
                                     q_scale, k_scale, v_scale,
                                     causal=False, sm_scale=sm_scale)
    torch.cuda.synchronize()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_profiles")
    os.makedirs(output_dir, exist_ok=True)

    # Profile with activities
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(num_iters):
            out = mxfp4_flash_attention(q_packed, k_packed, v_packed,
                                         q_scale, k_scale, v_scale,
                                         causal=False, sm_scale=sm_scale)
        torch.cuda.synchronize()

    # Print table
    print("\n--- Kernel Summary (sorted by CUDA time) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Export chrome trace
    trace_path = os.path.join(output_dir, "mxfp4_fa_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    print("View in: chrome://tracing or https://ui.perfetto.dev/")

    # Export stacks for flamegraph
    stacks_path = os.path.join(output_dir, "mxfp4_fa_stacks.txt")
    prof.export_stacks(stacks_path, "self_cuda_time_total")
    print(f"CUDA stacks exported to: {stacks_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile MXFP4 Flash Attention kernel")
    parser.add_argument("--proton", action="store_true", help="Profile with Triton Proton")
    parser.add_argument("--ncu", action="store_true", help="Generate ncu profiling commands")
    parser.add_argument("--ncu-target", action="store_true", help="Run as ncu profiling target")
    parser.add_argument("--torch-profiler", action="store_true", help="Profile with PyTorch profiler")
    parser.add_argument("--all", action="store_true", help="Run all profiling methods (except ncu-target)")

    args = parser.parse_args()

    if not any([args.proton, args.ncu, args.ncu_target, args.torch_profiler, args.all]):
        parser.print_help()
        print("\n\nNo profiling method specified. Use --proton, --ncu, --torch-profiler, or --all")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print()

    if args.ncu_target:
        profile_with_ncu_target()
        return

    if args.proton or args.all:
        profile_with_proton()
        print()

    if args.torch_profiler or args.all:
        profile_with_torch_profiler()
        print()

    if args.ncu or args.all:
        generate_ncu_commands()
        print()


if __name__ == "__main__":
    main()
