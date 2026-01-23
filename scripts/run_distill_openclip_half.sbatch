#!/bin/bash

#SBATCH --job-name=distill_openclip_half
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH -p rtx8000
#SBATCH --gres=gpu:rtx8000:1

module purge

singularity exec --nv \
            --overlay /scratch/sk10583/FengLab/tinyimage/tinyimage-fenglab/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
            /bin/bash -c "source /ext3/env.sh; python distill_openclip_half.py --val_from_annotations --epochs 50 --batch_size 512"


