#!/bin/bash -l
##SBATCH -n 32
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -J fourcastnet
#SBATCH -t 240:00:90
#SBATCH -p kshdexclu09
#SBATCH --gres=dcu:4
#SBATCH --exclusive
#SBATCH --mem=110G
#SBATCH -o ./log_shared/output-%j.log
#SBATCH -e ./log_shared/error-%j.log
# log_shared文件夹要手动创建

# Create a log directory for the job
time=`date +%Y.%m.%d-%H.%M.%S-$SLURM_JOB_ID`
mkdir ./log_shared/${time}

# Load modules
module use /public/software/modules
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/dtk/24.04

source /public/home/shizhm2050/Anaconda3/anaconda3/bin/activate
conda activate corrdiff-gexl

# Set the script path
SCRIPT_PATH="$(pwd)/train.py"

# Set OMP_NUM_THREADS environment variable
#export OMP_NUM_THREADS=1
## export HIP_LAUNCH_BLOCKING=1
#
#export NCCL_SOCKET_IFNAME=ib0
#export NCCL_NET_GDR_LEVEL=PHB
#export MASTER_ADDR=$(hostname)
#export NCCL_IB_HCA=mlx5_0:1
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_HCA=mlx5_0:1

export MASTER_ADDR=$(hostname)

# Run the script
#APP="python $SCRIPT_PATH"
config_file=./conf/AFNO.yaml
config='afno_backbone'
run_num='check_exp'

set -x
srun -u --mpi=pmix_v3 \
    bash -c "
    source export_DDP_vars.sh
    python train_fourcastnet.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "

# # Move the log files to the log directory
mv $(pwd)/log_shared/output-$SLURM_JOB_ID.log $(pwd)/log_shared/${time}/output-${time}.log
mv $(pwd)/log_shared/error-$SLURM_JOB_ID.log $(pwd)/log_shared/${time}/error-${time}.log

# # Remove the __pycache__ directory
# rm -rf $(pwd)/src/climax/__pycache__
# rm -rf $(pwd)/src/climax/utils/__pycache__
