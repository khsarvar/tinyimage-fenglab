#!/bin/bash

#SBATCH --account=torch_pr_485_general
#SBATCH --job-name=distill
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=16GB

module purge

# Optional image path:
# /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
singularity exec --nv \
            --overlay /scratch/sk10583/FengLab/my_pytorch.ext3:ro \
            /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
            /bin/bash -c "source /ext3/env.sh; python test.py"

