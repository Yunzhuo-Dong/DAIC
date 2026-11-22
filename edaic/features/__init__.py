"""Feature extractors. Each writes one file per session: $EDAIC_FEAT/<name>/<id>.npz

Keys inside every .npz:
    session        1-D float32 vector (session-level features)          [optional for wav2vec]
    session_names  names of the session features
    segments       per-segment features, [n_seg, D] or [n_seg, n_layers, D]
    seg_times      [n_seg, 2] start/end seconds of each participant segment

To add a modality, write a class with the same interface and register it below.
"""
from .base import Extractor
from .egemaps import EGeMAPSExtractor
from .praat import PraatExtractor
from .wav2vec2 import Wav2Vec2Extractor
from .reserved import OpenFaceExtractor, TextEmbeddingExtractor

REGISTRY = {
    "egemaps": EGeMAPSExtractor,       # provided openSMILE eGeMAPS LLDs -> functionals
    "frames": EGeMAPSExtractor,        # any provided frame table (BoAW, Deep Spectrum VGG16 / DenseNet201)
    "praat": PraatExtractor,           # parselmouth voice measures + transcript timing
    "wav2vec2": Wav2Vec2Extractor,     # layer-wise wav2vec 2.0 embeddings (GPU)
    "openface": OpenFaceExtractor,     # reserved: visual (RP2)
    "text": TextEmbeddingExtractor,    # reserved: transcript embeddings (RP2)
}


def build_extractor(cfg):
    kind = cfg["extractor"]["type"]
    if kind not in REGISTRY:
        raise KeyError(f"unknown extractor {kind!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[kind](cfg["extractor"])
