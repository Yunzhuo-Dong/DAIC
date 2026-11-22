"""Session index, labels, transcripts and participant speech segments.

Folder layout expected under EDAIC_RAW (as downloaded):
    labels/train_split.csv, dev_split.csv, test_split.csv, Detailed_PHQ8_Labels.csv
    data/300_P/300_AUDIO.wav, 300_Transcript.csv, features/300_OpenSMILE2.3.0_egemaps.csv, ...

Column names are matched loosely (case-insensitive substrings) because the
E-DAIC files are not fully consistent. Run 00_inspect.py once to confirm.
"""
import os
import re

import numpy as np
import pandas as pd

SPLITS = ("train", "dev", "test")


# ---------------------------------------------------------------- tables
def _is_number(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_table(path):
    """Read a CSV whose delimiter may be ',', ';' or tab, with or without a header row.

    Headerless tables (e.g. the E-DAIC BoAW files: 'unknown', time, bins...) are detected when
    most fields of the first line are numbers; columns are then named name, time, f0, f1, ...
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.readline()
    sep = max([",", ";", "\t"], key=head.count)
    toks = [t.strip().strip("'\"") for t in head.rstrip("\n").split(sep)]
    if len(toks) > 3 and sum(_is_number(t) for t in toks) >= 0.8 * len(toks):
        df = pd.read_csv(path, sep=sep, engine="c", low_memory=False, header=None)
        names = [f"f{i}" for i in range(df.shape[1])]
        if not _is_number(toks[0]):
            names[0], names[1] = "name", "time"
        df.columns = names
        return df
    return pd.read_csv(path, sep=sep, engine="c", low_memory=False)


def find_col(df, *needles, required=True):
    """First column whose lower-case name contains all needles."""
    for c in df.columns:
        name = str(c).lower().strip()
        if all(n in name for n in needles):
            return c
    if required:
        raise KeyError(f"no column containing {needles} in {list(df.columns)}")
    return None


# ---------------------------------------------------------------- labels
def load_labels(raw_dir):
    """One row per participant: id, split, gender, phq_score, phq_binary, item_1..item_8."""
    rows = []
    for split in SPLITS:
        path = os.path.join(raw_dir, "labels", f"{split}_split.csv")
        df = read_table(path)
        out = pd.DataFrame({"id": df[find_col(df, "participant")].astype(int), "split": split})
        for name, needles in {"gender": ("gender",), "phq_score": ("phq", "score"),
                              "phq_binary": ("phq", "binary")}.items():
            col = find_col(df, *needles, required=False)
            out[name] = df[col].values if col is not None else np.nan
        rows.append(out)
    lab = pd.concat(rows, ignore_index=True)

    items_path = os.path.join(raw_dir, "labels", "Detailed_PHQ8_Labels.csv")
    if os.path.exists(items_path):
        det = read_table(items_path)
        idc = find_col(det, "participant")
        cand = [c for c in det.columns if c != idc and pd.api.types.is_numeric_dtype(det[c])
                and not any(k in str(c).lower() for k in ("score", "binary", "total", "gender", "pcl", "ptsd"))]
        if len(cand) >= 8:
            items = det[[idc] + cand[:8]].copy()
            items.columns = ["id"] + [f"item_{i}" for i in range(1, 9)]
            items["id"] = items["id"].astype(int)
            lab = lab.merge(items, on="id", how="left")
            lab.attrs["item_names"] = dict(zip([f"item_{i}" for i in range(1, 9)], map(str, cand[:8])))
    if lab["phq_binary"].isna().all():
        lab["phq_binary"] = (lab["phq_score"] >= 10).astype(float)
    return lab


def session_paths(raw_dir, pid):
    d = os.path.join(raw_dir, "data", f"{pid}_P")
    f = os.path.join(d, "features")
    return {
        "dir": d,
        "audio": os.path.join(d, f"{pid}_AUDIO.wav"),
        "transcript": os.path.join(d, f"{pid}_Transcript.csv"),
        "egemaps": os.path.join(f, f"{pid}_OpenSMILE2.3.0_egemaps.csv"),
        "mfcc": os.path.join(f, f"{pid}_OpenSMILE2.3.0_mfcc.csv"),
        "openface": os.path.join(f, f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"),
        "boaw_egemaps": os.path.join(f, f"{pid}_BoAW_openSMILE_2.3.0_eGeMAPS.csv"),
        "boaw_mfcc": os.path.join(f, f"{pid}_BoAW_openSMILE_2.3.0_MFCC.csv"),
        "vgg16": os.path.join(f, f"{pid}_vgg16.csv"),
        "densenet201": os.path.join(f, f"{pid}_densenet201.csv"),
    }


def load_index(path):
    """Read sessions.csv written by 01_index.py."""
    return pd.read_csv(path)


# ---------------------------------------------------------------- transcripts
def load_transcript(path):
    """Return DataFrame[start, end, text] with participant turns only, sorted by start."""
    df = read_table(path)
    start = find_col(df, "start")
    end = find_col(df, "end", required=False) or find_col(df, "stop")
    text = find_col(df, "text", required=False) or find_col(df, "value")
    out = pd.DataFrame({"start": pd.to_numeric(df[start], errors="coerce"),
                        "end": pd.to_numeric(df[end], errors="coerce"),
                        "text": df[text].astype(str)})
    spk = find_col(df, "speaker", required=False)
    if spk is not None:  # DAIC-WOZ style transcripts include Ellie's turns
        out = out[df[spk].astype(str).str.lower().str.contains("participant")]
    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] > out["start"]].sort_values("start").reset_index(drop=True)
    return out


# ---------------------------------------------------------------- Ellie / experimenter cleaning
def normalise(text):
    t = re.sub(r"[^a-z0-9' ]+", " ", str(text).lower().replace("’", "'"))
    return re.sub(r"\s+", " ", t).strip()


def load_phrases(path):
    with open(path, encoding="utf-8") as f:
        ph = {normalise(l) for l in f if l.strip() and not l.lstrip().startswith("#")}
    return sorted((p for p in ph if p), key=len, reverse=True)  # longest first


def clean_transcript(tr, ccfg, phrases):
    """Remove Ellie's scripted prompts and pre-interview experimenter speech.

    - rows before the first Ellie intro row are dropped (experimenter setup talk)
    - Ellie phrases are stripped from the text; rows left with < min_words words are dropped
    - rows that contained Ellie AND participant words are kept (flag `mixed`) unless drop_mixed
    Returns (clean transcript, stats dict).
    """
    tr = tr.reset_index(drop=True).copy()
    norm = tr["text"].map(normalise)
    stats = {"rows_raw": len(tr), "rows_pre_intro": 0, "rows_setup_kw": 0, "rows_ellie_only": 0,
             "rows_mixed": 0, "intro_found": False, "start_rule": "none"}
    pat = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b") if phrases else None
    if ccfg.get("drop_before_intro", True):
        markers = [normalise(m) for m in ccfg.get("intro_markers", [])]
        window = float(ccfg.get("start_window_s", 150))
        early = tr["start"] < window
        intro = [i for i in range(len(tr)) if early[i] and any(m in norm[i] for m in markers)]
        ellie = [i for i in range(len(tr)) if early[i] and pat is not None and pat.search(norm[i])]
        if intro:      # drop setup talk AND the intro row itself (Ellie's monologue; ASR glues words)
            cut, stats["intro_found"], stats["start_rule"] = intro[0] + 1, True, "intro"
        elif ellie:    # first scripted question marks the start; keep that row (answer may follow)
            cut, stats["start_rule"] = ellie[0], "first_question"
        else:          # Ellie never transcribed: skip the setup period by time
            cut, stats["start_rule"] = int((tr["start"] < float(ccfg.get("skip_seconds", 40))).sum()), "time"
        stats["rows_pre_intro"] = cut
        tr, norm = tr.iloc[cut:], norm.iloc[cut:]
        kws = [normalise(k) for k in ccfg.get("setup_keywords", [])]
        setup = np.array([any(k in t for k in kws) for t in norm], dtype=bool) & (tr["start"] < window).to_numpy()
        stats["rows_setup_kw"] = int(setup.sum())
        tr, norm = tr[~setup], norm[~setup]
    stripped = norm.map(lambda t: re.sub(r"\s+", " ", pat.sub(" ", t)).strip() if pat else t)
    had_ellie = stripped != norm
    n_words = stripped.str.split().str.len().fillna(0)
    ellie_only = had_ellie & (n_words < int(ccfg.get("min_words", 1)))
    mixed = had_ellie & ~ellie_only
    stats["rows_ellie_only"], stats["rows_mixed"] = int(ellie_only.sum()), int(mixed.sum())
    tr = tr.assign(text=stripped, mixed=mixed.values)
    keep = ~ellie_only
    if ccfg.get("drop_mixed", False):
        keep &= ~mixed
    tr = tr[keep.values].reset_index(drop=True)
    stats["rows_clean"] = len(tr)
    return tr, stats


_PHRASE_CACHE = {}


def participant_transcript(path, cfg):
    """Load + clean a transcript according to cfg['cleaning'] (paths relative to configs/)."""
    tr = load_transcript(path)
    ccfg = cfg.get("cleaning", {})
    if not ccfg.get("enabled", False):
        return tr, {}
    pf = ccfg["phrases_file"]
    if pf not in _PHRASE_CACHE:
        _PHRASE_CACHE[pf] = load_phrases(pf)
    return clean_transcript(tr, ccfg, _PHRASE_CACHE[pf])


def session_segments(pid, transcript_path, cfg, audio_path=None):
    """(participant transcript, segments [n, 2]) for one session.

    segments.source = transcript  : segments from the cleaned transcript rows (rule-based)
    segments.source = diarization : segments = participant turns from 01b_diarize.py; transcript rows
                                    are kept only if >= min_overlap of them lies inside those turns
    segments.source = full        : the whole recording in max_dur chunks (all speakers; ablation)
    """
    sc = cfg["segments"]
    tr, stats = participant_transcript(transcript_path, cfg)
    if sc.get("source", "transcript") in ("others", "ellie"):
        # others: everyone except the participant; ellie: only clusters labelled Ellie by 01c
        d = os.path.join(sc["diar_dir"], str(int(pid)))
        turns = pd.read_csv(os.path.join(d, "turns.csv"))
        spk = pd.read_csv(os.path.join(d, "speakers.csv"))
        if sc["source"] == "ellie":
            keep = set(spk.loc[spk.get("role", pd.Series("", index=spk.index)) == "ellie", "speaker"].astype(str))
            rows = turns[turns.speaker.astype(str).isin(keep)].sort_values("start")
        else:
            part = set(spk.loc[spk.chosen.astype(bool), "speaker"].astype(str))
            rows = turns[~turns.speaker.astype(str).isin(part)].sort_values("start")
        segs = participant_segments(rows, sc["min_dur"], sc["max_dur"], sc.get("merge_gap", 0.0))
        return tr, segs, stats
    if sc.get("source", "transcript") == "full":
        import soundfile as sf
        dur = sf.info(audio_path).duration
        segs = participant_segments(pd.DataFrame({"start": [0.0], "end": [dur]}), 0.0, sc["max_dur"])
        return tr, segs, stats
    if sc.get("source", "transcript") == "diarization":
        p = os.path.join(sc["diar_dir"], str(int(pid)), "participant_segments.csv")
        turns = pd.read_csv(p).sort_values("start")[["start", "end"]].to_numpy(float)
        cov = []
        for s, e in zip(tr["start"], tr["end"]):
            ov = np.clip(np.minimum(turns[:, 1], e) - np.maximum(turns[:, 0], s), 0, None).sum()
            cov.append(ov / (e - s) if e > s else 0.0)
        keep = np.asarray(cov) >= float(sc.get("min_overlap", 0.5))
        stats["rows_outside_participant"] = int((~keep).sum())
        tr = tr[keep].reset_index(drop=True)
        rows = pd.DataFrame({"start": turns[:, 0], "end": turns[:, 1]})
        segs = participant_segments(rows, sc["min_dur"], sc["max_dur"], sc.get("merge_gap", 0.0))
    else:
        segs = participant_segments(tr, sc["min_dur"], sc["max_dur"], sc.get("merge_gap", 0.0))
    return tr, segs, stats


def merge_rows(transcript, merge_gap):
    """Merge consecutive transcript rows separated by <= merge_gap seconds into turns."""
    merged = []
    for s, e in zip(transcript["start"], transcript["end"]):
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def participant_segments(transcript, min_dur=0.5, max_dur=20.0, merge_gap=0.0):
    """Turn transcript rows into (start, end) segments: merge rows closer than merge_gap,
    drop segments shorter than min_dur, chunk ones longer than max_dur."""
    segs = []
    for s, e in merge_rows(transcript, merge_gap):
        if e - s < min_dur:
            continue
        t = s
        while e - t > max_dur:
            segs.append((t, t + max_dur))
            t += max_dur
        if e - t >= min_dur:
            segs.append((t, e))
    return np.asarray(segs, dtype=np.float64).reshape(-1, 2)


def timing_features(transcript):
    """Turn-taking / speech-rate features from the transcript (no audio needed)."""
    dur = (transcript["end"] - transcript["start"]).to_numpy()
    words = transcript["text"].str.split().str.len().fillna(0).to_numpy()
    gaps = (transcript["start"].to_numpy()[1:] - transcript["end"].to_numpy()[:-1])
    gaps = gaps[gaps > 0]
    total = dur.sum() if len(dur) else np.nan
    return {
        "n_turns": float(len(dur)),
        "speech_total_s": float(total),
        "turn_dur_mean": float(dur.mean()) if len(dur) else np.nan,
        "turn_dur_std": float(dur.std()) if len(dur) else np.nan,
        "words_total": float(words.sum()),
        "words_per_turn": float(words.mean()) if len(words) else np.nan,
        "speech_rate_wps": float(words.sum() / total) if total and total > 0 else np.nan,
        "gap_mean_s": float(gaps.mean()) if len(gaps) else np.nan,
        "gap_median_s": float(np.median(gaps)) if len(gaps) else np.nan,
        "gap_p90_s": float(np.percentile(gaps, 90)) if len(gaps) else np.nan,
    }


def mask_frames(times, segments):
    """Boolean mask of frame times that fall inside any segment."""
    if len(segments) == 0:
        return np.zeros_like(times, dtype=bool)
    idx = np.searchsorted(segments[:, 0], times, side="right") - 1
    ok = idx >= 0
    m = np.zeros_like(times, dtype=bool)
    m[ok] = times[ok] < segments[idx[ok], 1]
    return m


def segment_ids(times, segments):
    """Segment index per frame (-1 when outside all segments)."""
    idx = np.searchsorted(segments[:, 0], times, side="right") - 1
    ok = idx >= 0
    out = np.full(times.shape, -1, dtype=int)
    inside = np.zeros_like(ok)
    inside[ok] = times[ok] < segments[idx[ok], 1]
    out[inside] = idx[inside]
    return out
