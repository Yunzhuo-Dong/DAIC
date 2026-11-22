"""Two feature sets without any audio content (CPU, ~1 min), for the shortcut check (RP3):

    python 07_structure.py --config configs/diarize.yaml

$EDAIC_FEAT/interview_structure/<id>.npz   how the INTERVIEW went (not the participant's voice):
    recording length, speech of Ellie / experimenter / others, number of speakers,
    number of non-participant turns (follow-up questions)
$EDAIC_FEAT/participant_timing/<id>.npz    how the PARTICIPANT behaves in time (no voice quality):
    speaking time and share, number and length of turns, response latency
    (other speaker ends -> participant starts), pauses between own turns, words per minute

Train them like any other feature set:
    python 03_train.py --config configs/exp_feat_ridge.yaml --set features.0.name=interview_structure name=structure_ridge
"""
import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf

from edaic.config import load_config
from edaic.data import load_index, session_segments
from edaic.utils import log, save_npz


def _stats(x, prefix):
    x = np.asarray(x, float)
    if not len(x):
        return {f"{prefix}_{k}": np.nan for k in ("mean", "median", "p90")}
    return {f"{prefix}_mean": x.mean(), f"{prefix}_median": np.median(x), f"{prefix}_p90": np.percentile(x, 90)}


def session_features(r, cfg, diar_dir):
    d = os.path.join(diar_dir, str(int(r.id)))
    turns = pd.read_csv(os.path.join(d, "turns.csv")).sort_values("start")
    spk = pd.read_csv(os.path.join(d, "speakers.csv"))
    spk["speaker"] = spk.speaker.astype(str)
    turns["speaker"] = turns.speaker.astype(str)
    role = dict(zip(spk.speaker, spk.get("role", np.where(spk.chosen, "participant", "other"))))
    turns["role"] = turns.speaker.map(role).fillna("other")
    turns["dur"] = turns.end - turns.start
    is_p = turns.role == "participant"
    rec_min = sf.info(r.audio_path).duration / 60

    structure = {
        "recording_min": rec_min,
        "ellie_min": turns.loc[turns.role == "ellie", "dur"].sum() / 60,
        "experimenter_min": turns.loc[turns.role == "experimenter", "dur"].sum() / 60,
        "other_min": turns.loc[~is_p, "dur"].sum() / 60,
        "n_speakers": float(spk.shape[0]),
        "n_nonparticipant_turns": float((~is_p).sum()),
        "nonparticipant_turns_per_min": float((~is_p).sum()) / rec_min,
    }

    p = turns[is_p]
    # response latency: a non-participant turn ends, the next turn is the participant's
    lat, prev = [], None
    for t in turns.itertuples():
        if prev is not None and prev.role != "participant" and t.role == "participant":
            gap = t.start - prev.end
            if -1.0 <= gap <= 10.0:
                lat.append(max(gap, 0.0))
        prev = t
    own_gaps = (p.start.to_numpy()[1:] - p.end.to_numpy()[:-1]) if len(p) > 1 else np.array([])
    own_gaps = own_gaps[(own_gaps > 0) & (own_gaps < 10)]
    tr, _, _ = session_segments(r.id, r.transcript_path, cfg, r.audio_path)
    words = tr.text.str.split().str.len().fillna(0).sum()
    p_min = p.dur.sum() / 60
    timing = {
        "participant_min": p_min,
        "participant_share": p_min / max(turns.dur.sum() / 60, 1e-6),
        "participant_turns": float(len(p)),
        **_stats(p.dur, "turn_dur_s"),
        **_stats(lat, "response_latency_s"),
        **_stats(own_gaps, "own_pause_s"),
        "words_per_min": words / p_min if p_min > 0 else np.nan,
        "words_per_turn": words / len(p) if len(p) else np.nan,
    }
    return structure, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/diarize.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["segments"]["source"] = "diarization"
    diar_dir = cfg["diarization"]["out_dir"]
    feat = cfg["paths"]["features"]
    idx = load_index(cfg["paths"]["index"])
    rows, fails = [], []
    for r in idx.itertuples():
        try:
            s, t = session_features(r, cfg, diar_dir)
        except Exception as e:
            fails.append((r.id, f"{type(e).__name__}: {e}"))
            continue
        for name, f in (("interview_structure", s), ("participant_timing", t)):
            save_npz(os.path.join(feat, name, f"{int(r.id)}.npz"),
                     session=np.array(list(f.values()), np.float32), session_names=np.array(list(f)))
        rows.append({"id": r.id, "split": r.split, "phq": r.phq_score, **s, **t})
    df = pd.DataFrame(rows)
    log(f"wrote {len(df)} sessions, {len(fails)} failed {fails[:3]}")
    lab = df[df.split.isin(["train", "dev"])]
    corr = lab.drop(columns=["id", "split", "phq"]).corrwith(lab.phq, method="spearman").sort_values(key=abs, ascending=False)
    print("\nSpearman correlation with PHQ-8 (train+dev):")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
