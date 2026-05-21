#!/bin/bash
cd /home/yiliu7/workspace/triton-fork/examples/mxfp
/opt/nvidia/nsight-compute/2026.1.1/ncu \
    --set full \
    --kernel-name _mxfp4_attn_fwd \
    --launch-skip 3 --launch-count 1 \
    --csv \
    -o ncu_profiles/mxfp4_fa_75k_full \
    /home/yiliu7/workspace/venvs/omni/bin/python proton_target_75k.py \
    > /tmp/ncu_output.log 2>&1
echo "NCU EXIT CODE: $?" >> /tmp/ncu_output.log
