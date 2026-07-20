#!/bin/bash
# Download N320 grid file required by AIFS model
# The grid file will be placed in the current directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
modelscope download --model OneScience/AIFS_Single_v1 model/grid-n320.npz --local_dir "${SCRIPT_DIR}/"
