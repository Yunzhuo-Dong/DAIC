"""Speaker-state-conditioned facial features from the E-DAIC OpenFace 2.1 files (CPU).

States (from the diarization, turns.csv + speakers.csv roles):
    SPEAK   participant turns
    SILENT  gaps between two participant turns (<= max_gap_s): the participant is not speaking - listening to
            Ellie's next question or pausing. Defined from the participant's own turns only, so it does not depend
            on whether Ellie's (quiet) voice was found by the diarization (it is missing in most WoZ sessions).
Every turn / gap loses `edge_margin_s` at both ends (turn-taking transitions). Frames: success == 1,
confidence >= min_confidence.
Sessions failing the pre-defined video QC rule (see `visual.exclude`) get NaN features -> text-only in the fusion.

    python 15_visual_features.py --config configs/multimodal.yaml --qc     # quality check only (sync, valid frames)
    python 15_visual_features.py --config configs/multimodal.yaml          # QC + features

Features per state (35): for 8 AUs mean intensity + occurrence rate; expressivity (mean AU intensity, mean AU SD);
dynamics (AU change speed, head rotation / translation speed, gaze speed); head pose and gaze variability and level
(SD of pitch/yaw/roll, SD of gaze x/y, mean gaze y, mean head pitch); events per minute and mean duration of AU04,
AU12, AU15 occurrences.
    $EDAIC_FEAT/visual_state/<id>.npz    speak (35) + silent (35) + speak-silent contrast (10)
    $EDAIC_FEAT/visual_global/<id>.npz   the same 35 statistics over all valid frames (control: no speaker states)
    $EDAIC_FEAT/visual_qc.csv            per-session quality check

QC: video length vs audio length; share of valid frames; valid seconds per state; lip-speech synchrony
(cross-correlation of the participant-speaking indicator with mouth opening AU25+AU26 over +-3 s; best lag ~ 0 means
audio, video and diarization are aligned).
"""
import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf

from edaic.config import load_config
from edaic.data import load_index, read_table
from edaic.utils import log, save_df, save_npz

CONTRAST = ["expr_mean", "expr_sd", "au_speed", "head_rot_speed", "gaze_speed", "gaze_y_mean", "head_pitch_mean",
            "AU12_mean", "AU04_mean", "AU12_events_per_min"]


# ---------------------------------------------------------------- inputs
def load_openface(path):
    df = read_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    col = lambda pat: [c for c in df.columns if re.fullmatch(pat, c)]
    out = {"t": df[col(r"timestamp")[0]].to_numpy(float),
           "conf": df[col(r"confidence")[0]].to_numpy(float) if col(r"confidence") else np.ones(len(df)),
           "succ": df[col(r"success")[0]].to_numpy(float) if col(r"success") else np.ones(len(df))}
    out["au_r"] = {c[:4]: df[c].to_numpy(float) for c in col(r"AU\d\d_r")}
    out["au_c"] = {c[:4]: df[c].to_numpy(float) for c in col(r"AU\d\d_c")}
    out["R"] = df[[c for c in ("pose_Rx", "pose_Ry", "pose_Rz") if c in df]].to_numpy(float)
    out["T"] = df[[c for c in ("pose_Tx", "pose_Ty", "pose_Tz") if c in df]].to_numpy(float)
    out["gaze"] = df[[c for c in ("gaze_angle_x", "gaze_angle_y") if c in df]].to_numpy(float)
    return out


def load_turns(diar_dir, pid):
    d = os.path.join(diar_dir, str(int(pid)))
    turns = pd.read_csv(os.path.join(d, "turns.csv")).sort_values("start")
    spk = pd.read_csv(os.path.join(d, "speakers.csv"))
    spk["speaker"], turns["speaker"] = spk.speaker.astype(str), turns.speaker.astype(str)
    role = dict(zip(spk.speaker, spk.get("role", np.where(spk.chosen, "participant", "other"))))
    turns["role"] = turns.speaker.map(role).fillna("other")
    return turns


def in_intervals(t, iv, margin=0.0):
    """Boolean mask: t inside any [start + margin, end - margin]."""
    m = np.zeros(len(t), bool)
    for s, e in iv:
        s, e = s + margin, e - margin
        if e > s:
            m[np.searchsorted(t, s, "left"):np.searchsorted(t, e, "right")] = True
    return m


# ---------------------------------------------------------------- features
def _runs(binary, block, min_len):
    """Lengths (frames) of runs of 1 inside contiguous blocks."""
    lens = []
    for b in np.unique(block):
        x = binary[block == b] > 0.5
        if not x.any():
            continue
        d = np.diff(np.concatenate([[0], x.astype(int), [0]]))
        L = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        lens.extend(L[L >= min_len])
    return np.array(lens, float)


def state_features(of, mask, dt, vcfg):
    """35 statistics over the frames in `mask` (time order); NaN if too few valid seconds."""
    idx = np.flatnonzero(mask)
    names = []
    for au in vcfg["aus"]:
        names += [f"{au}_mean", f"{au}_occ"]
    names += ["expr_mean", "expr_sd", "au_speed", "head_rot_speed", "head_trans_speed", "gaze_speed",
              "head_pitch_sd", "head_yaw_sd", "head_roll_sd", "gaze_x_sd", "gaze_y_sd", "gaze_y_mean", "head_pitch_mean"]
    for au in vcfg["event_aus"]:
        names += [f"{au}_events_per_min", f"{au}_event_dur_s"]
    seconds = len(idx) * dt
    if seconds < float(vcfg["min_state_s"]):
        return dict.fromkeys(names, np.nan), seconds

    t = of["t"][idx]
    # contiguous blocks: consecutive frame indices and no time gap
    brk = np.concatenate([[True], (np.diff(idx) > 1) | (np.diff(t) > 1.5 * dt)])
    block = np.cumsum(brk)
    pair = ~brk[1:]                                  # (i-1, i) pairs inside a block
    dts = np.diff(t)[pair]

    def speed(x):                                    # mean |dx|/dt over contiguous pairs; x [n] or [n, k]
        dx = np.diff(x, axis=0)[pair]
        dx = np.linalg.norm(dx, axis=1) if dx.ndim == 2 else np.abs(dx)
        return float(np.nanmean(dx / dts)) if len(dts) else np.nan

    f = {}
    for au in vcfg["aus"]:
        f[f"{au}_mean"] = float(np.nanmean(of["au_r"][au][idx])) if au in of["au_r"] else np.nan
        f[f"{au}_occ"] = float(np.nanmean(of["au_c"][au][idx])) if au in of["au_c"] else np.nan
    R = np.stack(list(of["au_r"].values()), 1)[idx]
    f["expr_mean"] = float(np.nanmean(R))
    f["expr_sd"] = float(np.nanmean(np.nanstd(R, 0)))
    f["au_speed"] = float(np.nanmean([speed(R[:, j]) for j in range(R.shape[1])]))
    f["head_rot_speed"] = speed(of["R"][idx]) if of["R"].shape[1] == 3 else np.nan
    f["head_trans_speed"] = speed(of["T"][idx]) if of["T"].shape[1] == 3 else np.nan
    f["gaze_speed"] = speed(of["gaze"][idx]) if of["gaze"].shape[1] == 2 else np.nan
    for j, k in enumerate(("pitch", "yaw", "roll")):
        f[f"head_{k}_sd"] = float(np.nanstd(of["R"][idx, j])) if of["R"].shape[1] == 3 else np.nan
    for j, k in enumerate(("x", "y")):
        f[f"gaze_{k}_sd"] = float(np.nanstd(of["gaze"][idx, j])) if of["gaze"].shape[1] == 2 else np.nan
    f["gaze_y_mean"] = float(np.nanmean(of["gaze"][idx, 1])) if of["gaze"].shape[1] == 2 else np.nan
    f["head_pitch_mean"] = float(np.nanmean(of["R"][idx, 0])) if of["R"].shape[1] == 3 else np.nan
    minutes = seconds / 60
    min_len = max(1, int(round(float(vcfg["event_min_s"]) / dt)))
    for au in vcfg["event_aus"]:
        if au not in of["au_c"]:
            f[f"{au}_events_per_min"] = f[f"{au}_event_dur_s"] = np.nan
            continue
        L = _runs(of["au_c"][au][idx], block, min_len)
        f[f"{au}_events_per_min"] = len(L) / minutes
        f[f"{au}_event_dur_s"] = float(L.mean() * dt) if len(L) else 0.0
    return {k: f[k] for k in names}, seconds


def lip_sync(of, valid, speak_iv, max_lag):
    """Best lag (s) and correlation between 'participant speaks' and mouth opening (AU25 + AU26)."""
    mouth_aus = [a for a in ("AU25", "AU26") if a in of["au_r"]]
    if not mouth_aus or valid.sum() < 100:
        return np.nan, np.nan, np.nan
    t = of["t"][valid]
    mouth = sum(of["au_r"][a][valid] for a in mouth_aus)
    grid = np.arange(t[0], t[-1], 0.1)
    m = np.interp(grid, t, mouth)
    s = in_intervals(grid, speak_iv).astype(float)
    if s.std() == 0 or m.std() == 0:
        return np.nan, np.nan, np.nan
    lags = np.arange(-int(max_lag / 0.1), int(max_lag / 0.1) + 1)
    cs = []
    for L in lags:
        a, b = (s[:len(s) - L], m[L:]) if L >= 0 else (s[-L:], m[:len(m) + L])
        cs.append(np.corrcoef(a, b)[0, 1])
    cs = np.array(cs)
    k = int(np.nanargmax(cs))
    return float(lags[k] * 0.1), float(cs[k]), float(cs[lags == 0][0])


def process(args):
    r, cfg, qc_only = args
    vcfg, diar_dir = cfg["visual"], cfg["diarization"]["out_dir"]
    pid = int(r.id)
    qc = {"id": pid, "split": r.split}
    try:
        of = load_openface(r.openface_path)
        turns = load_turns(diar_dir, pid)
    except Exception as e:  # missing / broken video or diarization
        qc["error"] = f"{type(e).__name__}: {e}"
        return qc, None
    t = of["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    valid = (of["succ"] > 0.5) & (of["conf"] >= float(vcfg["min_confidence"]))
    p_iv = turns.loc[turns.role == "participant", ["start", "end"]].to_numpy()
    e_iv = turns.loc[turns.role == "ellie", ["start", "end"]].to_numpy()
    margin = float(vcfg["edge_margin_s"])
    ex = vcfg["exclude"]
    lag, c_best, c0 = lip_sync(of, valid, p_iv, float(vcfg["sync_max_lag_s"]))
    offset = (c_best >= float(ex["min_sync_corr"]) and abs(lag) > float(ex["max_sync_lag_s"])
              and c_best - c0 > float(ex["min_lag_gain"]))
    shift = lag if (offset and ex.get("correct_offset", True)) else 0.0
    if shift:  # video lags the audio by `lag` s -> move the video time axis back
        of = dict(of, t=of["t"] - shift)
        t = of["t"]
    speak = valid & in_intervals(t, p_iv, margin)
    p_sorted = p_iv[np.argsort(p_iv[:, 0])] if len(p_iv) else p_iv
    gaps = np.stack([p_sorted[:-1, 1], p_sorted[1:, 0]], 1) if len(p_sorted) > 1 else np.zeros((0, 2))
    gaps = gaps[(gaps[:, 1] - gaps[:, 0] > 0) & (gaps[:, 1] - gaps[:, 0] <= float(vcfg["max_gap_s"]))]
    silent = valid & in_intervals(t, gaps, margin) & ~in_intervals(t, p_iv)
    listen_ellie = valid & in_intervals(t, e_iv, margin) & ~in_intervals(t, p_iv)   # QC only
    try:
        audio_s = sf.info(r.audio_path).duration
    except Exception:
        audio_s = np.nan
    qc.update({"fps": 1 / dt if dt else np.nan, "video_s": t[-1] - t[0] if len(t) else 0, "audio_s": audio_s,
               "valid_share": float(valid.mean()) if len(t) else 0, "speak_valid_s": speak.sum() * dt,
               "silent_valid_s": silent.sum() * dt, "ellie_listen_valid_s": listen_ellie.sum() * dt,
               "sync_best_lag_s": lag, "sync_corr_best": c_best, "sync_corr_lag0": c0,
               "offset_corrected_s": shift, "offset_uncorrected": bool(offset and not shift)})
    qc["video_minus_audio_s"] = qc["video_s"] - audio_s
    qc["video_ok"], qc["reason"] = video_ok(qc, vcfg["exclude"])
    if qc_only:
        return qc, None
    fs, _ = state_features(of, speak, dt, vcfg)
    fl, _ = state_features(of, silent, dt, vcfg)
    fg, _ = state_features(of, valid, dt, vcfg)
    feats = {"state": {**{f"speak_{k}": v for k, v in fs.items()}, **{f"silent_{k}": v for k, v in fl.items()},
                       **{f"contrast_{k}": fs[k] - fl[k] for k in CONTRAST}},
             "global": {f"all_{k}": v for k, v in fg.items()}}
    if not qc["video_ok"]:  # unreliable video: all NaN -> the fusion falls back to text for this session
        feats = {k: dict.fromkeys(v, np.nan) for k, v in feats.items()}
    return qc, feats


def video_ok(q, ex):
    """Pre-defined exclusion rule (fixed from the QC only, before looking at any PHQ result).

    Alignment is tested directly by lip-speech synchrony: a session is kept when the participant's speech moves the
    mouth at (or, after correction, near) lag 0. A different video/audio length alone is not a reason - the
    recordings simply start together and end at different times. A clear offset (best lag > 1 s and clearly better
    than lag 0) is corrected by shifting the video time axis (correct_offset: true) instead of excluding.
    """
    why = []
    if not q["valid_share"] >= float(ex["min_valid_share"]):
        why.append("valid frames")
    if ex.get("max_length_diff_s") is not None and not abs(q["video_minus_audio_s"]) <= float(ex["max_length_diff_s"]):
        why.append("video/audio length")
    if not q["sync_corr_best"] >= float(ex["min_sync_corr"]):
        why.append("no lip-speech sync")
    elif q.get("offset_uncorrected"):
        why.append("audio-video offset")
    return (not why), "; ".join(why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multimodal.yaml")
    ap.add_argument("--qc", action="store_true", help="quality check only, no features")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    vcfg = cfg["visual"]
    feat = cfg["paths"]["features"]
    idx = load_index(cfg["paths"]["index"])
    if "openface_path" not in idx:
        raise SystemExit("sessions.csv has no openface_path column - re-run 01_index.py")
    if a.limit:
        idx = idx.head(a.limit)
    jobs = [(SimpleNamespace(**r._asdict()), cfg, a.qc) for r in idx.itertuples()]
    with ProcessPoolExecutor(int(vcfg.get("workers", 8))) as ex:
        res = list(ex.map(process, jobs, chunksize=4))

    qc = pd.DataFrame([q for q, _ in res])
    save_df(os.path.join(feat, "visual_qc.csv"), qc)
    n_err = int(qc.get("error", pd.Series(dtype=str)).notna().sum())
    ok = qc[qc.get("error", pd.Series(index=qc.index, dtype=object)).isna()] if n_err else qc
    log(f"QC: {len(qc)} sessions, {n_err} without usable video/diarization")
    if n_err:
        print(qc[qc.error.notna()][["id", "split", "error"]].to_string(index=False, max_colwidth=90))
    cols = ["fps", "video_minus_audio_s", "valid_share", "speak_valid_s", "silent_valid_s", "ellie_listen_valid_s",
            "sync_best_lag_s", "sync_corr_best", "sync_corr_lag0"]
    print("\nQC distribution (all sessions with video):")
    print(ok[cols].describe(percentiles=[0.05, 0.5, 0.95]).T[["count", "mean", "5%", "50%", "95%", "min", "max"]]
          .round(2).to_string())
    print("\nby split (median):")
    print(ok.groupby("split")[cols].median().round(2).T.to_string())
    ms = float(vcfg["min_state_s"])
    flags = {
        "video and audio length differ > 5 s": ok.video_minus_audio_s.abs() > 5,
        "lip-speech best lag > 0.5 s": ok.sync_best_lag_s.abs() > 0.5,
        "lip-speech correlation < 0.1": ok.sync_corr_best < 0.1,
        "valid frames < 50 %": ok.valid_share < 0.5,
        f"speak < {ms:.0f} valid s": ok.speak_valid_s < ms,
        f"silent < {ms:.0f} valid s": ok.silent_valid_s < ms,
    }
    print("\nflags:")
    for k, m in flags.items():
        ids = ok.loc[m.fillna(False), "id"].tolist()
        print(f"  {k:38s} {len(ids):3d}  {ids[:15]}{' ...' if len(ids) > 15 else ''}")
    bad = ok[~ok.video_ok.astype(bool)]
    print(f"\nEXCLUDED by the video rule ({len(bad)}; train {(bad.split == 'train').sum()}, "
          f"dev {(bad.split == 'dev').sum()}, test {(bad.split == 'test').sum()}) -> text-only for these sessions:")
    show = ["id", "split", "video_minus_audio_s", "valid_share", "sync_best_lag_s", "sync_corr_best",
            "sync_corr_lag0", "reason"]
    print(bad[show].round(2).to_string(index=False) if len(bad) else "  none")
    lenflag = ok[(ok.video_minus_audio_s.abs() > 5) & ok.video_ok.astype(bool)]
    if len(lenflag):
        print(f"\nkept although video and audio length differ > 5 s (lip-speech sync at lag ~0 -> aligned start):")
        print(lenflag[show[:-1]].round(2).to_string(index=False))
    sh = ok[ok.offset_corrected_s != 0]
    if len(sh):
        print(f"\naudio-video offset corrected (video time axis shifted by the lip-speech lag):")
        print(sh[show[:-1] + ["offset_corrected_s"]].round(2).to_string(index=False))
    log(f"wrote {os.path.join(feat, 'visual_qc.csv')}")
    if a.qc:
        return

    n = 0
    for (q, f) in res:
        if f is None:
            continue
        for kind in ("state", "global"):
            d = f[kind]
            save_npz(os.path.join(feat, f"visual_{kind}", f"{q['id']}.npz"),
                     session=np.array(list(d.values()), np.float32), session_names=np.array(list(d)))
        n += 1
    log(f"features for {n} sessions -> {feat}/visual_state, {feat}/visual_global "
        f"({len(res[0][1]['state']) if res and res[0][1] else '?'} + "
        f"{len(res[0][1]['global']) if res and res[0][1] else '?'} features)")


if __name__ == "__main__":
    main()
