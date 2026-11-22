# Barnard (CPU). Usage: cd "$HOME/py_coding/edaic_mm" && source slurm/barnard_env.sh
module purge
module load release/24.04 GCCcore/12.3.0 Python/3.11.3
source "$HOME/octb_cpu_env/bin/activate"
export WS="/data/horse/ws/yudo868h-Yunzhuo_space"
export CODE="$HOME/py_coding/edaic_mm"
export EDAIC_RAW="$WS/edaic_raw"
export EDAIC_FEAT="$WS/edaic_features"
export EDAIC_RUNS="$WS/edaic_runs"
export HF_HOME="$WS/hf_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
