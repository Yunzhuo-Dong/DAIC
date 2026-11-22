class Extractor:
    """Interface for all feature extractors.

    setup() is called once per process (load models there).
    extract() gets one row of sessions.csv, the participant segments [n, 2] in seconds
    and the participant transcript, and returns a dict of numpy arrays (see package doc).
    """
    needs_gpu = False

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def name(self):
        return self.cfg.get("name", self.cfg["type"])

    def setup(self, device="cpu"):
        pass

    def extract(self, row, segments, transcript):
        raise NotImplementedError
