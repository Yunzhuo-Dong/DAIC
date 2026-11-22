# Run once on each cluster's login node after sourcing the matching env file.
#   Capella: source slurm/env.sh         && bash slurm/setup_packages.sh gpu
#   Barnard: source slurm/barnard_env.sh && bash slurm/setup_packages.sh cpu
set -e
pip install pyyaml pandas scikit-learn soundfile praat-parselmouth
if [ "$1" = "gpu" ]; then
  pip install "transformers>=4.40"
  # download wav2vec 2.0 once (compute nodes have no internet)
  python -c "from transformers import AutoFeatureExtractor, Wav2Vec2Model; m='facebook/wav2vec2-base'; AutoFeatureExtractor.from_pretrained(m); Wav2Vec2Model.from_pretrained(m); print('cached', m)"
fi
python -c "import yaml, pandas, sklearn, soundfile, parselmouth; print('imports OK')"
