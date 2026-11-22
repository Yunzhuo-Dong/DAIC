"""Build sessions.csv: one row per participant with split, labels, file paths and speech stats.

    python 01_index.py --config configs/data.yaml
"""
import argparse
import os

import numpy as np

from edaic.config import load_config
from edaic.data import load_labels, session_paths, session_segments
from edaic.utils import log, save_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw, out = cfg["paths"]["raw"], cfg["paths"]["index"]
    seg_cfg = cfg["segments"]

    lab = load_labels(raw)
    if "item_names" in lab.attrs:
        log(f"PHQ-8 item columns: {lab.attrs['item_names']}")
    paths, stats = [], []
    for pid in lab["id"]:
        sp = session_paths(raw, pid)
        paths.append({f"{k}_path": v for k, v in sp.items() if k != "dir"})
        row = {"has_audio": os.path.exists(sp["audio"]), "has_transcript": os.path.exists(sp["transcript"]),
               "has_egemaps": os.path.exists(sp["egemaps"]), "n_segments": np.nan, "speech_min": np.nan}
        if row["has_transcript"]:
            tr, segs, cstats = session_segments(pid, sp["transcript"], cfg, sp["audio"])
            row.update(cstats)
            row["n_segments"] = len(segs)
            row["speech_min"] = float((segs[:, 1] - segs[:, 0]).sum() / 60) if len(segs) else 0.0
        stats.append(row)
    for k in paths[0]:
        lab[k] = [p[k] for p in paths]
    for k in dict.fromkeys(k for s in stats for k in s):
        lab[k] = [s.get(k, np.nan) for s in stats]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_df(out, lab)
    log(f"wrote {out}")
    print(lab.groupby("split").agg(n=("id", "size"), phq_mean=("phq_score", "mean"),
                                   phq_std=("phq_score", "std"), depressed=("phq_binary", "sum"),
                                   audio=("has_audio", "sum"), transcript=("has_transcript", "sum"),
                                   speech_min=("speech_min", "mean")).round(2).to_string())
    if "rows_mixed" in lab:
        aff = lab[(lab.rows_mixed > 0) | (lab.rows_ellie_only > 0)]
        print(f"\ncleaning: {len(aff)} sessions contained Ellie speech; "
              f"rows removed (Ellie only) = {int(lab.rows_ellie_only.sum())}, "
              f"rows stripped (mixed) = {int(lab.rows_mixed.sum())}, "
              f"pre-intro rows removed = {int(lab.rows_pre_intro.sum())} "
              f"setup-keyword rows removed = {int(lab.rows_setup_kw.sum())}")
        print("interview start found by:", lab.start_rule.value_counts().to_dict())
        print(aff.groupby("split").size().to_string())
    bad = lab[~(lab.has_audio & lab.has_transcript & lab.has_egemaps)]
    if len(bad):
        log(f"WARNING: {len(bad)} sessions miss files: {bad['id'].tolist()}")


if __name__ == "__main__":
    main()
