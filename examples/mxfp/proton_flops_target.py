"""Run proton profiling with flops/bytes metrics via LaunchHook."""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp4_flash_attention import quantize_to_mxfp4, mxfp4_flash_attention
import triton.profiler as proton

torch.manual_seed(42)

configs = [
    (2, 32, 4096, 128, False, "B2H32N4096D128_noncausal"),
    (2, 32, 4096, 128, True, "B2H32N4096D128_causal"),
    (1, 32, 8192, 128, False, "B1H32N8192D128_noncausal"),
]

for B, H, N, D, causal, name in configs:
    q_f16 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.05
    k_f16 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.05
    v_f16 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16) * 0.05
    q_packed, q_scale = quantize_to_mxfp4(q_f16)
    k_packed, k_scale = quantize_to_mxfp4(k_f16)
    v_t = v_f16.permute(0, 1, 3, 2).contiguous()
    v_flat = v_t.reshape(-1, N)
    v_packed_flat, v_scale_flat = quantize_to_mxfp4(v_flat)
    v_packed = v_packed_flat.reshape(B, H, D, N // 2)
    v_scale = v_scale_flat.reshape(B, H, D, N // 32)
    sm_scale = 1.0 / (D ** 0.5)

    # warmup
    for _ in range(3):
        out = mxfp4_flash_attention(q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, causal=causal, sm_scale=sm_scale)
    torch.cuda.synchronize()

    proton.enter_scope(name)
    for _ in range(10):
        out = mxfp4_flash_attention(q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, causal=causal, sm_scale=sm_scale)
    torch.cuda.synchronize()
    proton.exit_scope()

print("Done")
