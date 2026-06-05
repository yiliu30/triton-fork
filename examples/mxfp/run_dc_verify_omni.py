"""Wrapper: instrument SDPA for DC verification, then run vllm-omni Wan2.2 t2v."""
import sys
sys.path.insert(0, "/home/yiliu7/workspace/vllm-qdq-plugin/src")

# Instrument BEFORE any model import
from vllm_qdq_plugin.sage3_attn.sage3.verify_dc_offset import instrument_sdpa, print_report, restore_sdpa
instrument_sdpa(max_calls=500)

import atexit
atexit.register(print_report)
atexit.register(restore_sdpa)

# Now exec the original text_to_video.py with its original sys.argv
sys.argv = [
    "text_to_video.py",
    "--model", "/storage/yiliu7/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "--prompt", "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "--height", "480",
    "--width", "832",
    "--num-frames", "5",
    "--num-inference-steps", "2",
    "--guidance-scale", "4.0",
    "--tensor-parallel-size", "1",
    "--output", "/tmp/dc_verify_output.mp4",
]

exec(open("/home/yiliu7/workspace/vllm-omni/examples/offline_inference/text_to_video/text_to_video.py").read())
