#!/bin/bash
# Dump PTX and SASS for mxfp4_flash_attention.py
# Usage: ./dump_ptx_sass.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/yiliu7/workspace/venvs/omni/bin/python
NVDISASM=/usr/local/cuda-13.2/bin/nvdisasm
OUTPUT_DIR="$SCRIPT_DIR"

export C_INCLUDE_PATH=$HOME/.local/include:$HOME/.local/include/python3.12${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}

# Clear triton cache to force recompilation
rm -rf ~/.triton/cache

# Run with PTX dump enabled
echo "Compiling and dumping PTX..."
cd "$SCRIPT_DIR"
NVPTX_ENABLE_DUMP=1 $PYTHON mxfp4_flash_attention.py 2>&1 | tee /tmp/mxfp4_full_output.txt

# Extract all PTX
awk '/\/\/ -----\/\/ NVPTX Dump \/\/-----/{found=1; next} /^Device:|^=====/{if(found) found=0} found' /tmp/mxfp4_full_output.txt > "$OUTPUT_DIR/mxfp4_all_kernels.ptx"

# Extract first PTX only
awk '/\/\/ -----\/\/ NVPTX Dump \/\/-----/{count++; if(count==1){found=1; next}} found && /\/\/ -----\/\/ NVPTX Dump \/\/-----/{found=0} found && /^Device:|^=====/{found=0} found' /tmp/mxfp4_full_output.txt > "$OUTPUT_DIR/mxfp4_kernel.ptx"

# Dump SASS from all cached cubins
echo ""
echo "Dumping SASS..."
CUBINS=($(find ~/.triton/cache -name "*.cubin" | sort))
if [ ${#CUBINS[@]} -gt 0 ]; then
    $NVDISASM -g "${CUBINS[0]}" > "$OUTPUT_DIR/mxfp4_kernel.sass"
    $NVDISASM -g "${CUBINS[-1]}" > "$OUTPUT_DIR/mxfp4_kernel_large.sass"
fi

echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/mxfp4_kernel.ptx "$OUTPUT_DIR"/mxfp4_all_kernels.ptx "$OUTPUT_DIR"/mxfp4_kernel.sass "$OUTPUT_DIR"/mxfp4_kernel_large.sass
