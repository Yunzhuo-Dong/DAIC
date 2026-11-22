"""Reserved extractors for RP2 (visual, text). Interfaces are fixed; bodies come later."""
from .base import Extractor


class OpenFaceExtractor(Extractor):
    """Visual: functionals of the provided OpenFace 2.1 AUs / gaze / pose (row['openface_path']),
    restricted to participant segments, same output keys as EGeMAPSExtractor."""

    def extract(self, row, segments, transcript):
        raise NotImplementedError("OpenFace extractor is reserved for RP2")


class TextEmbeddingExtractor(Extractor):
    """Text: sentence embeddings of participant turns (e.g. a sentence-transformers model).
    segments = one embedding per turn, session = mean embedding."""

    def extract(self, row, segments, transcript):
        raise NotImplementedError("Text extractor is reserved for RP2")
