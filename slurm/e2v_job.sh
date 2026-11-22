#!/bin/bash
#SBATCH --job-name=edaic_e2v
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# emotion2vec runs in its own environment (~/e2v_env: funasr + its own torch).
# Usage (Capella): sbatch --export=ALL,CMD="python 02_extract.py --config configs/extract_e2v_plus_large_full.yaml" slurm/e2v_job.sh
set -e
cd "$HOME/py_coding/edaic_mm"
module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3
source "$HOME/e2v_env/bin/activate"
export WS="/data/horse/ws/yudo868h-Yunzhuo_space"
export EDAIC_RAW="$WS/edaic_raw" EDAIC_FEAT="$WS/edaic_features" EDAIC_RUNS="$WS/edaic_runs"
export HF_HOME="$WS/hf_cache" HF_HUB_OFFLINE=1 MODELSCOPE_CACHE="$WS/modelscope_cache"
python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 'NO CUDA - aborting')"
echo "CMD: $CMD"
eval "$CMD"
