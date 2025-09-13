#!/bin/bash

unset ROCBLAS_TENSILE_LIBPATH
echo "START TIME: $(date)"

module purge

source ~/conda.env
conda activate fuxi
module load compiler/dtk/25.04

which python
which hipcc

python train_fuxi_base.py
# python train_fuxi_short.py
# python train_fuxi_medium.py
# python train_fuxi_long.py
