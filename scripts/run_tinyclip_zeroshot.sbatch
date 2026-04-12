#!/bin/bash

#SBATCH --account=torch_pr_485_general
#SBATCH --job-name=tinyclip-zs
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
HF_MODEL_ID="${HF_MODEL_ID:-wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"

SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}"
OVERLAY_PATH="${OVERLAY_PATH:-/scratch/sk10583/FengLab/my_pytorch.ext3}"

mkdir -p logs

singularity exec --nv \
  --overlay "${OVERLAY_PATH}:ro" \
  "${SIF_IMAGE}" \
  /bin/bash -lc "source /ext3/env.sh; cd /scratch/sk10583/FengLab; unset SSL_CERT_FILE SSL_CERT_DIR; CERT_PATH=\$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true); if [ -n \"\$CERT_PATH\" ] && [ -f \"\$CERT_PATH\" ]; then export SSL_CERT_FILE=\"\$CERT_PATH\"; fi; LOCAL_ARG=\"\"; if [ -n \"${LOCAL_MODEL_DIR}\" ]; then LOCAL_ARG=\"--local-model-dir ${LOCAL_MODEL_DIR}\"; fi; python tinyclip_tinyimagenet_zeroshot.py --data-root ${DATA_ROOT} --hf-model-id ${HF_MODEL_ID} --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} --device ${DEVICE} \$LOCAL_ARG"
