#!/bin/bash

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

source ~/conda.env
conda activate fuxi
module load compiler/dtk/25.04

which python
which hipcc

python inference_base.py
# python inference_short.py
# python inference_medium.py
# python inference_long.py