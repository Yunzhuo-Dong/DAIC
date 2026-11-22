"""Praat voice measures (via parselmouth) per participant segment + transcript timing.

Per segment: F0 mean/std/range (semitones re 100 Hz), voiced fraction, jitter (local),
shimmer (local), HNR, intensity mean/std.
Session vector: mean and std of those over segments, plus timing features
(turn durations, pauses between turns, speech rate) from the transcript.
"""
import numpy as np
import soundfile as sf

from ..data import timing_features
from .base import Extractor

SEG_NAMES = ["f0_mean_st", "f0_std_st", "f0_range_st", "voiced_frac", "jitter_local",
             "shimmer_local", "hnr_db", "intensity_mean_db", "intensity_std_db"]


def segment_measures(x, sr, f0_min=75.0, f0_max=500.0):
    import parselmouth
    from parselmouth.praat import call

    out = dict.fromkeys(SEG_NAMES, np.nan)
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    try:
        f0 = snd.to_pitch(pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array["frequency"]
        voiced = f0[f0 > 0]
        out["voiced_frac"] = len(voiced) / max(len(f0), 1)
        if len(voiced) > 2:
            st = 12 * np.log2(voiced / 100.0)
            out["f0_mean_st"], out["f0_std_st"] = st.mean(), st.std()
            out["f0_range_st"] = np.percentile(st, 95) - np.percentile(st, 5)
        pp = call(snd, "To PointProcess (periodic, cc)", f0_min, f0_max)
        out["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["shimmer_local"] = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = call(snd, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
        out["hnr_db"] = call(harm, "Get mean", 0, 0)
        inten = snd.to_intensity(minimum_pitch=f0_min).values.ravel()
        inten = inten[np.isfinite(inten)]
        if len(inten):
            out["intensity_mean_db"], out["intensity_std_db"] = inten.mean(), inten.std()
    except Exception:  # very short / silent segments: keep NaNs
        pass
    return [float(out[k]) if out[k] is not None else np.nan for k in SEG_NAMES]


class PraatExtractor(Extractor):
    def extract(self, row, segments, transcript):
        audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        audio = audio.mean(1)
        f0_min = float(self.cfg.get("f0_min", 75))
        f0_max = float(self.cfg.get("f0_max", 500))
        seg = np.array([segment_measures(audio[int(s * sr):int(e * sr)], sr, f0_min, f0_max)
                        for s, e in segments], dtype=np.float32).reshape(-1, len(SEG_NAMES))
        with np.errstate(all="ignore"):
            agg = np.concatenate([np.nanmean(seg, 0), np.nanstd(seg, 0)])
        timing = timing_features(transcript)
        names = [f"{n}_mean" for n in SEG_NAMES] + [f"{n}_std" for n in SEG_NAMES] + list(timing)
        sess = np.concatenate([agg, np.array(list(timing.values()), np.float32)]).astype(np.float32)
        return {"session": sess, "session_names": np.array(names),
                "segments": seg, "seg_times": segments.astype(np.float32)}
