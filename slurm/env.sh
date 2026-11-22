# Capella (GPU). Usage: cd "$HOME/py_coding/edaic_mm" && source slurm/env.sh
module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3
source "$HOME/diffusion_env/bin/activate"
export WS="/data/horse/ws/yudo868h-Yunzhuo_space"
export CODE="$HOME/py_coding/edaic_mm"
export EDAIC_RAW="$WS/edaic_raw"
export EDAIC_FEAT="$WS/edaic_features"
export EDAIC_RUNS="$WS/edaic_runs"
export HF_HOME="$WS/hf_cache"
export TOKENIZERS_PARALLELISM=false
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
