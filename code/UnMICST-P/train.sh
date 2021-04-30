#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -p [GPU_NAME]
#SBATCH -n 8 # number of cores
#SBATCH --mem 30000 # memory pool for all cores
#SBATCH --gres=gpu:1 # memory pool for all cores
#SBATCH -t 1-00:00 # time (D-HH:MM)

#SBATCH -o slurm/_train_%j.%N.out # STDOUT
#SBATCH -e slurm/_train_%j.%N.err # STDERR

module load python/3.6.3-fasrc01
module load cuda/9.2.88-fasrc01
source activate [VENV_NAME]
module load GCCcore/6.4.0
nvidia-smi -L

# Run (-u to prevent buffering)
CUDA_LAUNCH_BLOCKING=1 python -u train.py --config configs/psp_segmenter.yml