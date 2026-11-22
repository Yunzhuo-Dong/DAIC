"""Speaker diarization of every session with pyannote; keep the participant's turns.

    sbatch --export=ALL,CMD="python 01b_diarize.py --config configs/diarize.yaml" slurm/diar_job.sh

For each session writes $EDAIC_CLEAN/<id>/
    turns.csv                 all diarized turns (start, end, speaker)
    speakers.csv              per speaker: total seconds, share, loudness (dB), chosen flag
    participant_segments.csv  turns of the speaker chosen as participant (start, end)
and $EDAIC_CLEAN/diarization_summary.csv, which includes an agreement check against the
rule-based Ellie detection (transcript rows matched to Ellie's script should NOT fall
inside participant turns).

Participant = the speaker with the most speech, unless another speaker holding >= `min_share`
of the speech is >= `loud_margin_db` louder (close-talk microphone), then that one.
Existing session folders are skipped (resumable).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf

from edaic.config import load_config, save_config
from edaic.data import clean_transcript, load_index, load_phrases, load_transcript
from edaic.utils import log, save_df


def load_pipeline(cfg, device):
    import torch
    from pyannote.audio import Pipeline
    p = Pipeline.from_pretrained(cfg["pipeline"])  # token is read from $HF_HOME/token
    if p is None:
        raise SystemExit("pipeline could not be loaded: accept the model terms on Hugging Face and log in")
    return p.to(torch.device(device))


def run_pipeline(pipeline, audio, sr, cfg):
    """Return DataFrame[start, end, speaker]. Audio is passed in memory (no ffmpeg needed)."""
    import torch
    kwargs = {k: cfg[k] for k in ("num_speakers", "min_speakers", "max_speakers") if cfg.get(k) is not None}
    out = pipeline({"waveform": torch.from_numpy(audio[None, :]), "sample_rate": sr}, **kwargs)
    ann = getattr(out, "exclusive_speaker_diarization", None) if cfg.get("exclusive", True) else None
    ann = ann if ann is not None else getattr(out, "speaker_diarization", out)  # v4 output or v3 Annotation
    rows = []
    if hasattr(ann, "itertracks"):
        rows = [(t.start, t.end, str(s)) for t, _, s in ann.itertracks(yield_label=True)]
    else:
        rows = [(t.start, t.end, str(s)) for t, s in ann]
    return pd.DataFrame(rows, columns=["start", "end", "speaker"])


def choose_participant(turns, audio, sr, min_share=0.2, loud_margin_db=3.0):
    stats = []
    total = (turns.end - turns.start).sum()
    for spk, g in turns.groupby("speaker"):
        dur = float((g.end - g.start).sum())
        x = np.concatenate([audio[int(s * sr):int(e * sr)] for s, e in zip(g.start, g.end)] or [np.zeros(1)])
        rms_db = float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10))
        stats.append({"speaker": spk, "seconds": dur, "share": dur / total if total else 0.0, "rms_db": rms_db})
    st = pd.DataFrame(stats)
    longest = st.sort_values("seconds").iloc[-1]
    cand = st[st.share >= min_share]
    chosen = longest["speaker"]
    if len(cand):
        loudest = cand.sort_values("rms_db").iloc[-1]
        if loudest["rms_db"] - longest["rms_db"] >= loud_margin_db:  # clearly louder = close-talk mic
            chosen = loudest["speaker"]
    st["chosen"] = st.speaker == chosen
    return chosen, st.sort_values("seconds", ascending=False)


def overlap_frac(s, e, segs):
    """Fraction of [s, e] covered by segments [n, 2]."""
    if e <= s or len(segs) == 0:
        return 0.0
    ov = np.clip(np.minimum(segs[:, 1], e) - np.maximum(segs[:, 0], s), 0, None).sum()
    return float(min(ov / (e - s), 1.0))


def agreement(transcript_path, part, data_cfg, phrases):
    """Share of rule-detected Ellie rows / kept rows that fall inside participant turns."""
    tr = load_transcript(transcript_path)
    clean, _ = clean_transcript(tr, data_cfg["cleaning"], phrases)
    kept = set(zip(clean.start.round(3), clean.end.round(3)))
    segs = part[["start", "end"]].to_numpy()
    ellie, keep = [], []
    for s, e in zip(tr.start, tr.end):
        f = overlap_frac(s, e, segs)
        (keep if (round(s, 3), round(e, 3)) in kept else ellie).append(f)
    return {"rule_removed_rows": len(ellie), "rule_removed_in_participant": float(np.mean(ellie)) if ellie else np.nan,
            "rule_kept_rows": len(keep), "rule_kept_in_participant": float(np.mean(keep)) if keep else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/diarize.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. diarization.max_speakers=3")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    dcfg = cfg["diarization"]
    out_root = dcfg["out_dir"]
    os.makedirs(out_root, exist_ok=True)
    save_config(cfg, os.path.join(out_root, "diarize_config.yaml"))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not dcfg.get("allow_cpu", False):
        raise SystemExit("no GPU: submit via slurm/diar_job.sh")
    pipeline = load_pipeline(dcfg, device)
    phrases = load_phrases(cfg["cleaning"]["phrases_file"])

    idx = load_index(cfg["paths"]["index"])
    if a.ids:
        idx = idx[idx.id.isin(a.ids)]
    if a.limit:
        idx = idx.head(a.limit)
    log(f"diarizing {len(idx)} sessions on {device} -> {out_root}")

    summary = []
    for i, r in enumerate(idx.itertuples(), 1):
        d = os.path.join(out_root, str(r.id))
        done = os.path.exists(os.path.join(d, "participant_segments.csv"))
        try:
            audio, sr = sf.read(r.audio_path, dtype="float32", always_2d=True)
            audio = audio.mean(1)
            if done and not a.overwrite:
                turns = pd.read_csv(os.path.join(d, "turns.csv"))
                st = pd.read_csv(os.path.join(d, "speakers.csv"))
                chosen, secs = st.loc[st.chosen, "speaker"].iloc[0], 0.0
            else:
                t0 = time.time()
                turns = run_pipeline(pipeline, audio, sr, dcfg)
                chosen, st = choose_participant(turns, audio, sr, float(dcfg.get("min_share", 0.2)),
                                                float(dcfg.get("loud_margin_db", 3.0)))
                secs = time.time() - t0
                save_df(os.path.join(d, "turns.csv"), turns)
                save_df(os.path.join(d, "speakers.csv"), st)
                save_df(os.path.join(d, "participant_segments.csv"), turns[turns.speaker == chosen][["start", "end"]])
            part = turns[turns.speaker.astype(str) == str(chosen)]
            row = {"id": r.id, "split": r.split, "n_speakers": int(st.shape[0]),
                   "participant_min": float((part.end - part.start).sum() / 60),
                   "participant_share": float(st.loc[st.chosen, "share"].iloc[0]),
                   "loudness_gap_db": float(st.loc[st.chosen, "rms_db"].iloc[0] - st.loc[~st.chosen, "rms_db"].max())
                   if (~st.chosen).any() else np.nan,
                   "runtime_s": secs}
            row.update(agreement(r.transcript_path, part, cfg, phrases))
            summary.append(row)
        except Exception as e:
            log(f"{r.id}: FAILED {type(e).__name__}: {e}")
            summary.append({"id": r.id, "split": r.split, "error": f"{type(e).__name__}: {e}"})
        if i % 10 == 0:
            log(f"{i}/{len(idx)}")

    s = pd.DataFrame(summary)
    save_df(os.path.join(out_root, "diarization_summary.csv"), s)
    ok = s[s.get("error").isna()] if "error" in s else s
    log(f"done: {len(ok)} ok, {len(s) - len(ok)} failed")
    if len(ok):
        print(ok[["n_speakers", "participant_min", "participant_share", "loudness_gap_db",
                  "rule_removed_in_participant", "rule_kept_in_participant"]].describe().round(2).to_string())
        print("\nGood sign: rule_removed_in_participant low (Ellie rows outside participant turns), "
              "rule_kept_in_participant high.")


if __name__ == "__main__":
    main()
