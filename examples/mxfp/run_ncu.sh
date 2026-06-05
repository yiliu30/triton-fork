#!/bin/bash
set -ex
cd /home/yiliu7/workspace/triton-fork/examples/mxfp
mkdir -p ncu_profiles

ncu --set full \
    --kernel-name '_mxfp4_attn_fwd' \
    --launch-skip 3 --launch-count 1 \
    --csv \
    -o ncu_profiles/mxfp4_fa_75k_full \
    /home/yiliu7/workspace/venvs/omni/bin/python proton_target_75k.py
