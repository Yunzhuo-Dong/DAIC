"""Layer-wise wav2vec 2.0 embeddings of participant segments.

For every segment the waveform goes through the (frozen) model; each hidden layer
(CNN output + every transformer layer) is mean-pooled over time.
Stored: segments [n_seg, n_layers+1, D] float16 and a duration-weighted session
mean [n_layers+1, D] so that layer probing needs no re-extraction.

Models are loaded from the Hugging Face cache (HF_HOME). Download them once on a
login node (compute nodes have no internet); see README.
"""
import numpy as np
import soundfile as sf

from .base import Extractor


def _resample(x, sr, target=16000):
    if sr == target:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


class Wav2Vec2Extractor(Extractor):
    needs_gpu = True

    @property
    def name(self):
        return self.cfg.get("name") or "wav2vec2_" + self.cfg["model"].split("/")[-1]

    def setup(self, device="cpu"):
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2Model

        self.torch = torch
        self.device = device
        local = bool(self.cfg.get("local_files_only", True))
        self.fe = AutoFeatureExtractor.from_pretrained(self.cfg["model"], local_files_only=local)
        self.model = Wav2Vec2Model.from_pretrained(self.cfg["model"], local_files_only=local)
        self.model.eval().to(device)
        self.half = device != "cpu" and bool(self.cfg.get("fp16", True))

    def extract(self, row, segments, transcript):
        torch = self.torch
        audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        audio = _resample(audio.mean(1), sr)
        sr = 16000
        min_len = int(float(self.cfg.get("min_seconds", 0.5)) * sr)
        feats, keep = [], []
        for i, (s, e) in enumerate(segments):
            x = audio[int(s * sr):int(e * sr)]
            if len(x) < min_len:
                continue
            inp = self.fe(x, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
            with torch.inference_mode(), torch.autocast(self.device.split(":")[0], enabled=self.half):
                hs = self.model(inp, output_hidden_states=True).hidden_states
            feats.append(torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy())
            keep.append(i)
        if not feats:
            raise RuntimeError("no usable segments")
        seg = np.stack(feats).astype(np.float32)              # [n_seg, L+1, D]
        times = segments[keep]
        w = (times[:, 1] - times[:, 0])[:, None, None]
        sess_layers = (seg * w).sum(0) / w.sum()               # [L+1, D]
        return {"segments": seg.astype(np.float16), "session_layers": sess_layers.astype(np.float32),
                "seg_times": times.astype(np.float32)}
