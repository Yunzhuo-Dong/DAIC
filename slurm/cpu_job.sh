#!/bin/bash
#SBATCH --job-name=edaic_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# Usage (on Barnard): sbatch --export=ALL,CMD="python 02_extract.py --config configs/extract_praat.yaml --workers 16" slurm/cpu_job.sh
set -e
cd "$HOME/py_coding/edaic_mm"
source slurm/barnard_env.sh
echo "CMD: $CMD"
eval "$CMD"
