#!/bin/bash
# Run mxfp4_flash_attention.py inside the yi-omni docker container
# Uses host triton (3.6.0) via PYTHONPATH override
export PYTHONPATH=/home/yiliu7/workspace/venvs/omni/lib/python3.12/site-packages:$PYTHONPATH
cd /home/yiliu7/workspace/triton-fork/examples/mxfp
python mxfp4_flash_attention.py
