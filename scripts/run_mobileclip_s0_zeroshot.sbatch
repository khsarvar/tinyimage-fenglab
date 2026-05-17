#!/bin/bash

#SBATCH --account=torch_pr_485_general
#SBATCH --job-name=zeroshot_mobileclip_s0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"

set -euo pipefail

module purge

DATA_ROOT="${DATA_ROOT:-tiny-imagenet-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"
OVERLAY_PATH="${OVERLAY_PATH:-/scratch/sk10583/FengLab/my_pytorch.ext3}"

mkdir -p logs

singularity exec --nv \
  --overlay "${OVERLAY_PATH}:ro" \
  "${SIF_IMAGE}" \
  /bin/bash -lc "source /ext3/env.sh; cd /scratch/sk10583/FengLab; \
  unset SSL_CERT_FILE SSL_CERT_DIR; \
  CERT_PATH=\$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true); \
  if [ -n \"\$CERT_PATH\" ] && [ -f \"\$CERT_PATH\" ]; then export SSL_CERT_FILE=\"\$CERT_PATH\"; fi; \
  python openclip_imagenet.py \
    --data-root ${DATA_ROOT} \
    --model MobileCLIP2-S0 \
    --pretrained dfndr2b \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --device cuda \
    --amp"
