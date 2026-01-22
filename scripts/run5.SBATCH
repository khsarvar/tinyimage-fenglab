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
            /bin/bash -c "source /ext3/env.sh; python train_tinynet.py --arch resnet18 --depth 3 --batch_size 512 --epoch 100"
