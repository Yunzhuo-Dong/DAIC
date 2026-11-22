"""Re-select the participant from existing diarization output (no GPU, ~1 min).

    python 01c_select_participant.py --config configs/diarize.yaml            # summary only
    python 01c_select_participant.py --config configs/diarize.yaml --apply    # write selections

Rule (per session, from turns.csv + speakers.csv written by 01b_diarize.py):
  1. Ellie        = speakers covering >= ellie_overlap of the transcript rows that match Ellie's script
                    AND (>= ellie_quiet_db quieter than the loudest speaker OR not the longest speaker)
  2. experimenter = speakers whose last turn ends before setup_end_s (setup talk only)
  3. participant  = loudest remaining speaker with >= min_seconds of speech and >= rel_min_seconds x the
                    longest remaining speaker; other clusters with >= merge_min_seconds, little Ellie overlap
                    and within merge_db are merged in (one person split into two clusters)
Rewrites speakers.csv (columns role, chosen) and participant_segments.csv, and writes
selection_summary.csv with a split agreement check:
  ellie_rows_in_participant  rows matching Ellie's script that fall inside participant turns (want ~0)
  kept_rows_in_participant   rule-kept participant rows inside participant turns (want high)
  coverage_ratio             participant minutes / minutes of rule-kept transcript rows
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from edaic.config import load_config
from edaic.data import clean_transcript, load_index, load_phrases, load_transcript, normalise
from edaic.utils import log, save_df


def cover(s, e, segs):
    if e <= s or len(segs) == 0:
        return 0.0
    return float(min(np.clip(np.minimum(segs[:, 1], e) - np.maximum(segs[:, 0], s), 0, None).sum() / (e - s), 1.0))


def mean_cover(rows, segs):
    return float(np.mean([cover(s, e, segs) for s, e in zip(rows.start, rows.end)])) if len(rows) else np.nan


def select(turns, spk, ellie_rows, scfg):
    st = spk.set_index("speaker")[["seconds", "share", "rms_db"]].copy()
    g = turns.groupby("speaker")
    st["first"], st["last"] = g.start.min(), g.end.max()
    st["ellie_overlap"] = [mean_cover(ellie_rows, turns[turns.speaker == k][["start", "end"]].to_numpy())
                           for k in st.index]
    ov = st.ellie_overlap.fillna(0)
    big = st.seconds >= float(scfg.get("min_seconds", 20))
    top = st.loc[big, "rms_db"].max() if big.any() else st.rms_db.max()
    longest = st.seconds.max()
    st["role"] = "other"
    st.loc[st["last"] <= float(scfg["setup_end_s"]), "role"] = "experimenter"
    # Ellie: overlaps her scripted rows AND (clearly quieter OR not the longest speaker).
    # ASR often merges Ellie's question with the answer, so the participant (usually the longest
    # speaker) can overlap those rows too; in loud recordings Ellie is not much quieter.
    is_ellie = (ov >= float(scfg["ellie_overlap"])) & (
        (st.rms_db <= top - float(scfg.get("ellie_quiet_db", 6.0))) | (st.seconds < longest))
    st.loc[is_ellie, "role"] = "ellie"
    pool = st[(st.role == "other") & big]
    if not len(pool):
        pool = st[(st.role != "ellie") & big]
    if not len(pool):
        pool = st[st.role != "ellie"] if (st.role != "ellie").any() else st
    # a candidate must hold a real part of the conversation, not a short loud burst
    cand = pool[pool.seconds >= float(scfg.get("rel_min_seconds", 0.25)) * pool.seconds.max()]
    main = cand.rms_db.idxmax()
    st.loc[main, "role"] = "participant"
    # merge clusters that look like the same person split in two (never Ellie / experimenter / bursts)
    merge = ((st.role == "other") & (st.seconds >= float(scfg.get("merge_min_seconds", 60)))
             & (ov < float(scfg.get("merge_max_ellie_overlap", 0.3)))
             & (st.rms_db >= st.loc[main, "rms_db"] - float(scfg["merge_db"])))
    st.loc[merge, "role"] = "participant"
    st["chosen"] = st.role == "participant"
    return st.reset_index().sort_values("seconds", ascending=False)


def original_rule(spk, min_share=0.2, margin=3.0):
    """Selection rule of 01b_diarize.py (longest speaker unless another is clearly louder)."""
    longest = spk.sort_values("seconds").iloc[-1]
    cand = spk[spk.share >= min_share]
    if len(cand):
        loud = cand.sort_values("rms_db").iloc[-1]
        if loud.rms_db - longest.rms_db >= margin:
            return {loud.speaker}
    return {longest.speaker}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/diarize.yaml")
    ap.add_argument("--apply", action="store_true", help="write speakers.csv / participant_segments.csv (default: summary only)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root, scfg = cfg["diarization"]["out_dir"], cfg["diarization"]["selection"]
    phrases = load_phrases(cfg["cleaning"]["phrases_file"])
    pat = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b")
    idx = load_index(cfg["paths"]["index"])

    rows = []
    for r in idx.itertuples():
        d = os.path.join(root, str(r.id))
        if not os.path.exists(os.path.join(d, "turns.csv")):
            rows.append({"id": r.id, "split": r.split, "error": "no diarization"})
            continue
        turns = pd.read_csv(os.path.join(d, "turns.csv"))
        spk = pd.read_csv(os.path.join(d, "speakers.csv"))
        tr = load_transcript(r.transcript_path)
        norm = tr.text.map(normalise)
        ellie_rows = tr[norm.map(lambda t: bool(pat.search(t)))]
        clean, _ = clean_transcript(tr, cfg["cleaning"], phrases)
        st = select(turns, spk, ellie_rows, scfg)
        part = turns[turns.speaker.isin(st.loc[st.chosen, "speaker"])].sort_values("start")[["start", "end"]]
        segs = part.to_numpy()
        ell_rms = st.loc[st.role == "ellie", "rms_db"]
        p_rms = st.loc[st.chosen, "rms_db"].max()
        kept_min = float((clean.end - clean.start).sum() / 60)
        rows.append({
            "id": r.id, "split": r.split, "n_speakers": len(st), "n_participant_clusters": int(st.chosen.sum()),
            "ellie_found": bool(len(ell_rms)), "participant_min": float((part.end - part.start).sum() / 60),
            "participant_share": float(st.loc[st.chosen, "share"].sum()),
            "gap_to_ellie_db": float(p_rms - ell_rms.max()) if len(ell_rms) else np.nan,
            "gap_to_next_db": float(p_rms - st.loc[~st.chosen, "rms_db"].max()) if (~st.chosen).any() else np.nan,
            "ellie_rows": len(ellie_rows), "ellie_rows_in_participant": mean_cover(ellie_rows, segs),
            "kept_rows_in_participant": mean_cover(clean, segs),
            "coverage_ratio": float((part.end - part.start).sum() / 60 / kept_min) if kept_min else np.nan,
            "differs_from_01b_rule": bool(set(st.loc[st.chosen, "speaker"]) != original_rule(spk)),
            "roles": " ".join(f"{k}:{rl}:{sec:.0f}s:{db:.0f}dB" for k, rl, sec, db in
                              zip(st.speaker, st.role, st.seconds, st.rms_db)),
        })
        if a.apply:
            save_df(os.path.join(d, "speakers.csv"), st)
            save_df(os.path.join(d, "participant_segments.csv"), part)

    s = pd.DataFrame(rows)
    save_df(os.path.join(root, "selection_summary.csv"), s)
    pd.set_option("display.width", 200)
    cols = ["participant_min", "participant_share", "gap_to_ellie_db", "gap_to_next_db",
            "ellie_rows_in_participant", "kept_rows_in_participant", "coverage_ratio"]
    print(s[cols].describe().round(2).to_string())
    print(f"\ndiffers from the original 01b rule in {int(s.differs_from_01b_rule.sum())} sessions")
    print(f"participant merged from >1 cluster in {int((s.n_participant_clusters > 1).sum())} sessions")
    print(f"Ellie cluster found in {int(s.ellie_found.sum())}/{len(s)} sessions")
    flag = s[(s.coverage_ratio < 0.4) | (s.ellie_rows_in_participant > 0.3) | (s.gap_to_next_db < 0)]
    print(f"\nto check ({len(flag)}); roles = speaker:role:seconds:loudness")
    print(flag[["id", "split", "participant_min", "ellie_rows_in_participant", "coverage_ratio",
                "roles"]].round(2).to_string(index=False, max_colwidth=120))
    if not a.apply:
        print("\n(summary only - rerun with --apply to write the selections)")
    log(f"wrote {os.path.join(root, 'selection_summary.csv')}")


if __name__ == "__main__":
    main()
