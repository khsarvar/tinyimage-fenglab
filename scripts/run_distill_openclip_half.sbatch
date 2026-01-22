#!/bin/bash
#SBATCH --job-name=distill_openclip_half
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---- resource requests (edit for your cluster) ----
# #SBATCH --partition=your_partition
# #SBATCH --time=24:00:00
# #SBATCH --mem=32G
# #SBATCH --gres=gpu:1

set -euo pipefail

# Ensure we run from project root even if submitted from scripts/
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ---- environment setup (edit as needed) ----
# module load cuda/12.1
# source ~/venvs/myenv/bin/activate

python distill_openclip_half.py
