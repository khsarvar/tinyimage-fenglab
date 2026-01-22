#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH -p rtx8000
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=torch

module purge

singularity exec --nv \
            --overlay /scratch/sk10583/FengLab/tinyimage/tinyimage-fenglab/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
            /bin/bash -c "source /ext3/env.sh; python finetune_OpenClip_imagenet.py --batch_size 256 --epochs 30 --unfreeze_k 1 --fp16 --val_from_annotations"
