"""emotion2vec (FunASR) embeddings: one vector per session.

The audio (whole recording or participant turns, see segments.source) is cut into chunks of at most
segments.max_dur seconds; FunASR averages each chunk over time (granularity="utterance"); chunks are
averaged weighted by duration. A whole 15-min file cannot go through the transformer in one pass.

Stored in <id>.npz:
    session          duration-weighted mean embedding (768-d base / 1024-d large)
    segments         per-chunk embeddings [n_chunks, D] (float16)
    session_emotions mean emotion probabilities over chunks (plus models only)
    emotion_names    the emotion labels
    seg_times        chunk start/end in seconds

Models (download once on a login node into a local folder, see README):
    emotion2vec/emotion2vec_base          self-supervised representation, 768-d
    emotion2vec/emotion2vec_plus_large    fine-tuned for speech emotion recognition, ~300M params
"""
import numpy as np
import soundfile as sf

from .base import Extractor
from .wav2vec2 import _resample


class Emotion2VecExtractor(Extractor):
    needs_gpu = True

    @property
    def name(self):
        return self.cfg.get("name") or "e2v_" + str(self.cfg["model"]).rstrip("/").split("/")[-1]

    def setup(self, device="cpu"):
        from funasr import AutoModel
        kw = {"model": self.cfg["model"], "device": device, "disable_update": True}
        if self.cfg.get("hub"):
            kw["hub"] = self.cfg["hub"]
        self.model = AutoModel(**kw)

    def extract(self, row, segments, transcript):
        audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        audio = _resample(audio.mean(1), sr)
        sr = 16000
        min_len = int(float(self.cfg.get("min_seconds", 1.0)) * sr)
        feats, emos, keep, names = [], [], [], None
        for i, (s, e) in enumerate(segments):
            x = audio[int(s * sr):int(e * sr)]
            if len(x) < min_len:
                continue
            res = self.model.generate(x, granularity="utterance", extract_embedding=True, fs=sr)
            r = res[0]
            feats.append(np.asarray(r["feats"], np.float32).reshape(-1))
            if r.get("scores"):
                emos.append(np.asarray(r["scores"], np.float32))
                names = names or [str(lb).split("/")[-1] for lb in r["labels"]]
            keep.append(i)
        if not feats:
            raise RuntimeError("no usable segments")
        seg = np.stack(feats)
        times = segments[keep]
        w = (times[:, 1] - times[:, 0])[:, None]
        out = {"session": ((seg * w).sum(0) / w.sum()).astype(np.float32),
               "session_names": np.array([f"e2v_{j}" for j in range(seg.shape[1])]),
               "segments": seg.astype(np.float16), "seg_times": times.astype(np.float32)}
        if emos:
            em = np.stack(emos)
            out["session_emotions"] = ((em * w).sum(0) / w.sum()).astype(np.float32)
            out["emotion_names"] = np.array(names)
        return out
