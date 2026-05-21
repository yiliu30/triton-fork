"""Minimal wrapper: instrument SDPA then run Wan2.2 for 2 steps to collect DC stats."""
import sys
sys.path.insert(0, "/home/yiliu7/workspace/vllm-qdq-plugin/src")
sys.path.insert(0, "/home/yiliu7/workspace/vllm-omni")

# Instrument BEFORE any model import
from vllm_qdq_plugin.sage3_attn.sage3.verify_dc_offset import instrument_sdpa, print_report, restore_sdpa
instrument_sdpa(max_calls=200)

# Now run minimal Wan inference
import torch
from diffusers import WanPipeline

MODEL = "/storage/yiliu7/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

print("Loading Wan2.2 pipeline...")
pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
pipe.to("cuda")

print("Running 2 inference steps to collect attention DC statistics...")
with torch.no_grad():
    pipe(
        prompt="A cat walking on grass",
        height=480,
        width=832,
        num_frames=5,
        num_inference_steps=2,
        guidance_scale=4.0,
    )

print_report()
restore_sdpa()
