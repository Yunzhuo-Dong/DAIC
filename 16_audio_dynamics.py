"""Dynamics of the participant's emotion2vec embeddings over the interview (CPU, no model needed).

Uses the per-segment embeddings already stored by 02_extract.py (participant speech, `segments` + `seg_times`):
    disp_mean     mean cosine distance of the segments to the session centroid (duration-weighted): variability
    disp_p90      90th percentile of that distance: how far the most atypical moments go
    step_change   mean cosine distance between consecutive segments: moment-to-moment change
    drift         cosine distance between the centroids of the first and second half: change over the interview

    python 16_audio_dynamics.py --config configs/multimodal.yaml
    -> $EDAIC_FEAT/e2v_dynamics/<id>.npz
"""
import argparse
import os

import numpy as np
import pandas as pd

from edaic.config import load_config
from edaic.data import load_index
from edaic.utils import log, save_npz

NAMES = ["disp_mean", "disp_p90", "step_change", "drift"]


def cos_dist(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return 1 - (a * b).sum(-1)


def dynamics(seg, times):
    seg = seg.astype(np.float32)
    w = np.maximum(times[:, 1] - times[:, 0], 1e-3)
    if len(seg) < 4:
        return [np.nan] * 4
    order = np.argsort(times[:, 0])                  # time order
    seg, w = seg[order], w[order]
    c = (seg * w[:, None]).sum(0) / w.sum()
    d = cos_dist(seg, c[None])
    half = int(np.searchsorted(np.cumsum(w), w.sum() / 2))
    half = min(max(half, 1), len(seg) - 1)
    c1 = (seg[:half] * w[:half, None]).sum(0) / w[:half].sum()
    c2 = (seg[half:] * w[half:, None]).sum(0) / w[half:].sum()
    return [float(np.average(d, weights=w)), float(np.percentile(d, 90)),
            float(cos_dist(seg[1:], seg[:-1]).mean()), float(cos_dist(c1, c2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multimodal.yaml")
    ap.add_argument("--source", default="e2v_plus_large", help="emotion2vec feature set with per-segment embeddings")
    a = ap.parse_args()
    cfg = load_config(a.config)
    feat = cfg["paths"]["features"]
    idx = load_index(cfg["paths"]["index"])
    rows, missing = [], []
    for r in idx.itertuples():
        p = os.path.join(feat, a.source, f"{int(r.id)}.npz")
        if not os.path.exists(p):
            missing.append(int(r.id))
            continue
        z = np.load(p)
        if "segments" not in z.files or "seg_times" not in z.files:
            raise SystemExit(f"{p} has no per-segment embeddings (segments / seg_times)")
        v = dynamics(z["segments"], z["seg_times"])
        save_npz(os.path.join(feat, "e2v_dynamics", f"{int(r.id)}.npz"),
                 session=np.array(v, np.float32), session_names=np.array(NAMES))
        rows.append({"id": int(r.id), "split": r.split, "phq": r.phq_score, "n_segments": len(z["seg_times"]),
                     **dict(zip(NAMES, v))})
    df = pd.DataFrame(rows)
    log(f"wrote {len(df)} sessions -> {feat}/e2v_dynamics ({len(missing)} without {a.source}: {missing[:10]})")
    lab = df[df.split.isin(["train", "dev"])]
    print("\nSpearman correlation with PHQ-8 (train+dev; descriptive only):")
    print(lab[NAMES + ["n_segments"]].corrwith(lab.phq, method="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
