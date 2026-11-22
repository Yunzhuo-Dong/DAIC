#!/bin/bash
# All ridge baselines, both heads (regression + PHQ>=10 classifier), every feature set that exists.
# Barnard: sbatch --time=02:00:00 --export=ALL,CMD="bash slurm/ridge_suite.sh" slurm/cpu_job.sh
set -e
cd "$(dirname "$0")/.."
for f in egemaps praat boaw_egemaps ds_vgg16 ds_densenet201; do
  for s in "" _full; do
    [ -d "$EDAIC_FEAT/$f$s" ] || continue
    python 03_train.py --config configs/exp_feat_ridge.yaml --set features.0.name=$f$s name=${f}${s}_ridge
    python 03_train.py --config configs/exp_feat_ridge.yaml --set features.0.name=$f$s name=${f}${s}_clf model.type=ridge_clf
  done
done
for f in w2v2_base w2v2_large w2v2_large_full; do
  [ -d "$EDAIC_FEAT/$f" ] || continue
  python 03_train.py --config configs/exp_layers_ridge.yaml --set features.0.name=$f name=${f}_layers_ridge
  python 03_train.py --config configs/exp_layers_ridge.yaml --set features.0.name=$f name=${f}_layers_clf model.type=ridge_clf
done
echo SUITE_DONE
