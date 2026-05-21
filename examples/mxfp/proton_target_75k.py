"""Profiling target: B=1, H=40, N=75648, D=128 (closest valid to 75600)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triton_sm100_fix  # noqa: F401 — disable buggy pass on sm_100
import torch
from mxfp4_flash_attention import quantize_to_mxfp4, mxfp4_flash_attention
import triton.profiler as proton

torch.manual_seed(42)
B, H, N, D = 1, 40, 75648, 128
causal = False
name = f"B{B}_H{H}_N{N}_D{D}_noncausal"

print(f"Config: B={B}, H={H}, N={N}, D={D}, causal={causal}")

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

# profiled
proton.enter_scope(name)
for _ in range(10):
    out = mxfp4_flash_attention(q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, causal=causal, sm_scale=sm_scale)
torch.cuda.synchronize()
proton.exit_scope()

import time
torch.cuda.synchronize()
start = time.perf_counter()
niters = 10
for _ in range(niters):
    out = mxfp4_flash_attention(q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, causal=causal, sm_scale=sm_scale)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / niters

flops = 4 * B * H * N * N * D
tflops = flops / elapsed / 1e12
print(f"Time: {elapsed*1000:.3f} ms, TFLOPS: {tflops:.1f}")
print("Done")
