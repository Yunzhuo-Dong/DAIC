"""Session functionals from frame-level feature files shipped with E-DAIC.

Works for any provided frame table: openSMILE eGeMAPS / MFCC LLDs (10 ms), BoAW histograms,
Deep Spectrum VGG16 / DenseNet201 (4 s window, 1 s hop). Only frames inside participant
segments are kept (participant_only: true), or the whole recording (false).

Config keys (extractor:):
    file_key          column of sessions.csv with the file path (egemaps_path, vgg16_path, ...)
    stats             session statistics, any of mean std p10 p50 p90   (default: all five)
    hop_s, offset_s   frame times when the file has no time column: t = offset_s + i * hop_s
    time_offset_s     added to the file's own time column (e.g. +2 if it stores the start of a 4 s window)
    participant_only  true = participant frames only, false = whole recording
    save_segments     store per-segment means (needed for sequence models; large for 4096-d)
"""
import numpy as np
import pandas as pd

from ..data import find_col, mask_frames, read_table, segment_ids
from .base import Extractor

ALL_STATS = ("mean", "std", "p10", "p50", "p90")
_DROP = ("name", "time", "frame", "confidence", "success", "file", "unnamed")


def functionals(x, stats=ALL_STATS):
    out = []
    with np.errstate(all="ignore"):
        for s in stats:
            if s == "mean":
                out.append(np.nanmean(x, 0))
            elif s == "std":
                out.append(np.nanstd(x, 0))
            elif s.startswith("p"):
                out.append(np.nanpercentile(x, float(s[1:]), axis=0))
            else:
                raise ValueError(f"unknown statistic {s!r}")
    return np.concatenate(out)


class EGeMAPSExtractor(Extractor):
    def extract(self, row, segments, transcript):
        df = read_table(row[self.cfg.get("file_key", "egemaps_path")])
        tcol = find_col(df, "time", required=False)
        if tcol is not None:
            times = pd.to_numeric(df[tcol], errors="coerce").to_numpy() + float(self.cfg.get("time_offset_s", 0.0))
        else:  # assume a fixed hop
            times = float(self.cfg.get("offset_s", 0.0)) + np.arange(len(df)) * float(self.cfg.get("hop_s", 0.01))
        num = df.select_dtypes(include=[np.number])
        num = num[[c for c in num.columns if not any(k in str(c).lower() for k in _DROP)]]
        names = [str(c) for c in num.columns]
        x = num.to_numpy(dtype=np.float32)
        stats = tuple(self.cfg.get("stats", ALL_STATS))

        if self.cfg.get("participant_only", True) and len(segments):
            keep = mask_frames(times, segments)
            seg_of = segment_ids(times, segments)
        else:
            keep = np.ones(len(x), bool)
            seg_of = np.zeros(len(x), int)
            segments = np.array([[np.nanmin(times), np.nanmax(times)]])
        if keep.sum() == 0:
            raise RuntimeError(f"no frames inside participant segments (file has {len(x)} frames, "
                               f"time range {np.nanmin(times):.0f}-{np.nanmax(times):.0f} s)")

        out = {
            "session": functionals(x[keep], stats).astype(np.float32),
            "session_names": np.array([f"{n}_{s}" for s in stats for n in names]),
            "seg_times": segments.astype(np.float32),
            "n_frames": np.array([int(keep.sum())]),
        }
        if self.cfg.get("save_segments", True):
            seg_feats = np.full((len(segments), x.shape[1]), np.nan, np.float32)
            for i in np.unique(seg_of[seg_of >= 0]):
                seg_feats[i] = np.nanmean(x[seg_of == i], 0)
            out["segments"] = seg_feats
        return out
