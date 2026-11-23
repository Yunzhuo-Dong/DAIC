#!/bin/bash
#SBATCH --job-name=edaic_llm
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# LLM work runs in its own environment (~/llm_env: transformers, accelerate, sentence-transformers).
# Usage (Capella): sbatch --export=ALL,CMD="python 11_llm_score.py --config configs/llm_qwen.yaml" slurm/llm_job.sh
set -e
cd "$HOME/py_coding/edaic_mm"
module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3
source "$HOME/llm_env/bin/activate"
export WS="/data/horse/ws/yudo868h-Yunzhuo_space"
export EDAIC_RAW="$WS/edaic_raw" EDAIC_FEAT="$WS/edaic_features" EDAIC_RUNS="$WS/edaic_runs"
export HF_HOME="$WS/hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 'NO CUDA - aborting')"
echo "CMD: $CMD"
eval "$CMD"
