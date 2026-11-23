"""Text embeddings of the participant transcript (Qwen3-Embedding) -> one vector per session.

    sbatch --export=ALL,CMD="python 12_text_embed.py --config configs/llm_qwen.yaml" slurm/llm_job.sh

Writes $EDAIC_FEAT/<embed.name>/<id>.npz (session = embedding), so the usual ridge pipeline works:
    python 03_train.py --config configs/exp_feat_ridge.yaml --set features.0.name=text_qwen3emb8b_part name=text_ridge
    # audio + text (early fusion)
    python 03_train.py --config configs/exp_feat_ridge.yaml --set features.0.name=text_qwen3emb8b_part \
        features.1.name=e2v_plus_large name=text_e2v_ridge
Long transcripts are split into chunks of `chunk_words` words; chunk embeddings are averaged.
"""
import argparse
import os

import numpy as np

from edaic.config import load_config
from edaic.data import load_index, load_transcript, session_segments
from edaic.utils import log, save_npz


def session_text(r, cfg, source):
    if source == "participant":
        tr, _, _ = session_segments(r.id, r.transcript_path, cfg, r.audio_path)
    else:
        tr = load_transcript(r.transcript_path)
    return " ".join(t.strip() for t in tr.text.astype(str) if t.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llm_qwen.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    e = cfg["embed"]
    out_dir = os.path.join(cfg["paths"]["features"], e["name"])
    idx = load_index(cfg["paths"]["index"])

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(e["model"], device="cuda", model_kwargs={"torch_dtype": "bfloat16"})
    prefix = e.get("instruction", "")
    n_words = int(e.get("chunk_words", 400))
    for k, r in enumerate(idx.itertuples(), 1):
        out = os.path.join(out_dir, f"{int(r.id)}.npz")
        if os.path.exists(out):
            continue
        words = session_text(r, cfg, e["text_source"]).split()
        chunks = [" ".join(words[i:i + n_words]) for i in range(0, max(len(words), 1), n_words)] or [""]
        emb = model.encode([prefix + c for c in chunks], normalize_embeddings=True, batch_size=8)
        w = np.array([max(len(c.split()), 1) for c in chunks], float)[:, None]
        vec = (emb * w).sum(0) / w.sum()
        save_npz(out, session=vec.astype(np.float32), session_names=np.array([f"t{i}" for i in range(len(vec))]),
                 n_words=np.array([len(words)]))
        if k % 25 == 0:
            log(f"{k}/{len(idx)}")
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
