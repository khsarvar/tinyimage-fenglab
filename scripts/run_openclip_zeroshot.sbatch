#!/bin/bash

#SBATCH --account=torch_pr_485_general
#SBATCH --job-name=openclip_zeroshot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"

module purge

singularity exec --nv \
            --overlay /scratch/sk10583/FengLab/my_pytorch.ext3:ro \
            /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
            /bin/bash -lc "source /ext3/env.sh; cd /scratch/sk10583/FengLab; unset SSL_CERT_FILE SSL_CERT_DIR; CERT_PATH=\$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true); if [ -n \"\$CERT_PATH\" ] && [ -f \"\$CERT_PATH\" ]; then export SSL_CERT_FILE=\"\$CERT_PATH\"; fi; python openclip_imagenet.py --data-root tiny-imagenet-200 --device cuda --amp"
