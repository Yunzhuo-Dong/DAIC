"""Reserved interface for RP2 LLM experiments (05_llm.py).

Planned backends (run locally, no data leaves the cluster):
    zero-shot : Qwen3.8-27B / DeepSeek-V4-Flash prompted to score each PHQ-8 item 0-3, summed
    lora      : Qwen3.8-27B + LoRA + regression head
    omni      : Qwen3-Omni-30B-A3B listening to participant audio
Every backend must write preds_{dev,test}.csv with columns id, y, pred, so that
04_evaluate.py (metrics, late fusion, significance tests) works unchanged.
"""


class LLMScorer:
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self, device="cuda"):
        raise NotImplementedError

    def score(self, transcript_text, audio_path=None):
        """Return {"item_1": int, ..., "item_8": int, "total": float}."""
        raise NotImplementedError
