#!/bin/bash
#SBATCH --job-name=edaic_gpu
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# Usage (on Capella): sbatch --export=ALL,CMD="python 02_extract.py --config configs/extract_w2v2_base.yaml" slurm/gpu_job.sh
set -e
cd "$HOME/py_coding/edaic_mm"
source slurm/env.sh
export HF_HUB_OFFLINE=1
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'NO CUDA - aborting')"
echo "CMD: $CMD"
eval "$CMD"
