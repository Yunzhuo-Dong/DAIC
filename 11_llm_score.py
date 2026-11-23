"""Zero-shot PHQ-8 rating of interview transcripts with an open-weight LLM (Qwen), run locally.

    sbatch --export=ALL,CMD="python 11_llm_score.py --config configs/llm_qwen.yaml" slurm/llm_job.sh
    sbatch --export=ALL,CMD="python 11_llm_score.py --config configs/llm_qwen.yaml --set llm.text_source=full llm.name=qwen_full" slurm/llm_job.sh

The model reads a transcript and rates each of the 8 PHQ-8 items 0-3; the items are summed.
No training is involved, so the scores are evaluated on all 219 train+dev sessions and on test.

text_source:
    participant  participant turns only (diarization), Ellie's scripted phrases stripped   <- main
    full         the provided ASR transcript as is, interviewer included (shortcut check, like GPT-4 papers)

Outputs:
    $EDAIC_FEAT/llm_<name>/<id>.npz           session = 8 item scores (+ total) -> usable as ridge features / fusion
    $EDAIC_RUNS/llm_<name>/zeroshot/seed0/preds_{dev,test}.csv   for 04_evaluate.py
    $EDAIC_RUNS/llm_<name>/raw_outputs.jsonl  every prompt's raw answer (for error analysis)
Resumable: sessions already in raw_outputs.jsonl are skipped.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from edaic.config import load_config, save_config
from edaic.data import load_index, load_transcript, session_segments
from edaic.metrics import ccc, report, rmse
from edaic.utils import log, save_df, save_json, save_npz

ITEMS = ["little interest or pleasure in doing things",
         "feeling down, depressed, or hopeless",
         "trouble falling or staying asleep, or sleeping too much",
         "feeling tired or having little energy",
         "poor appetite or overeating",
         "feeling bad about yourself, or that you are a failure or have let yourself or your family down",
         "trouble concentrating on things",
         "moving or speaking so slowly that other people could have noticed, or being fidgety or restless"]

SYSTEM = (
    "You are an experienced clinical psychologist. You read the transcript of a semi-structured interview "
    "and estimate the participant's answers to the PHQ-8 depression questionnaire for the last two weeks. "
    "Each item is rated 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day. "
    "Base every rating on what the participant says and how they say it; when an item is not discussed, "
    "give your best estimate from the overall impression of the interview. Answer with JSON only.")

USER = (
    "Interview transcript ({source}):\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Rate the 8 PHQ-8 items:\n{items}\n\n"
    "Return exactly this JSON and nothing else:\n"
    "{{\"items\": [i1, i2, i3, i4, i5, i6, i7, i8], \"evidence\": \"one short sentence\"}}")


def transcript_text(r, cfg, source, max_chars):
    if source == "participant":
        tr, _, _ = session_segments(r.id, r.transcript_path, cfg, r.audio_path)
        lines = [t for t in tr.text.astype(str) if t.strip()]
        src = "participant's speech only; automatic transcription, interviewer removed"
    else:
        tr = load_transcript(r.transcript_path)
        lines = [t.strip() for t in tr.text.astype(str) if t.strip()]
        src = "automatic transcription of the whole session, interviewer and participant not separated"
    text = "\n".join(lines)
    if len(text) > max_chars:  # keep beginning and end
        half = max_chars // 2
        text = text[:half] + "\n[...]\n" + text[-half:]
    return text, src


def parse(answer):
    """Extract 8 item scores (0-3) from the model's answer; None if impossible."""
    a = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    m = re.search(r"\{.*\}", a, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            items = [int(round(float(v))) for v in obj.get("items", [])]
            if len(items) == 8:
                return [min(max(v, 0), 3) for v in items], obj.get("evidence", "")
        except (ValueError, TypeError):
            pass
    m = re.search(r"\[\s*([0-3](?:\s*,\s*[0-3]){7})\s*\]", a)
    if m:
        return [int(v) for v in re.findall(r"[0-3]", m.group(1))], ""
    return None, ""


# ---------------------------------------------------------------- backends
class HFBackend:
    def __init__(self, c):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(c["model"])
        kw = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(c["model"], **kw)
        except (ValueError, KeyError):  # vision-language checkpoints
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(c["model"], **kw)
        self.model.eval()
        self.c = c

    def __call__(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = dict(self.c.get("chat_template_kwargs", {}))
        try:
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=int(self.c.get("max_new_tokens", 512)),
                                      do_sample=float(self.c.get("temperature", 0)) > 0,
                                      temperature=float(self.c.get("temperature", 0)) or None,
                                      top_p=float(self.c.get("top_p", 1.0)))
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True), ids["input_ids"].shape[1]


class OpenAIBackend:
    """Any OpenAI-compatible LOCAL server (e.g. `vllm serve` on the compute node)."""

    def __init__(self, c):
        import requests
        self.requests, self.c = requests, c

    def __call__(self, system, user):
        body = {"model": self.c["served_name"], "temperature": float(self.c.get("temperature", 0)),
                "max_tokens": int(self.c.get("max_new_tokens", 512)),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.c.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = self.c["chat_template_kwargs"]
        r = self.requests.post(self.c["base_url"].rstrip("/") + "/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"], j.get("usage", {}).get("prompt_tokens", 0)


class MockBackend:  # pipeline test without a model
    def __init__(self, c):
        self.rng = np.random.default_rng(0)

    def __call__(self, system, user):
        return json.dumps({"items": self.rng.integers(0, 4, 8).tolist(), "evidence": "mock"}), len(user) // 4


BACKENDS = {"hf": HFBackend, "openai": OpenAIBackend, "mock": MockBackend}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llm_qwen.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    c = cfg["llm"]
    name = c["name"]
    run_dir = os.path.join(cfg["paths"]["runs"], f"llm_{name}")
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    raw_path = os.path.join(run_dir, "raw_outputs.jsonl")
    done = {}
    if os.path.exists(raw_path):
        for line in open(raw_path):
            d = json.loads(line)
            if d.get("items"):
                done[int(d["id"])] = d

    idx = load_index(cfg["paths"]["index"])
    if a.limit:
        idx = idx.groupby("split").head(a.limit)
    todo = [r for r in idx.itertuples() if int(r.id) not in done]
    log(f"{name}: {len(done)} done, {len(todo)} to rate with {c['backend']}:{c.get('model', c.get('served_name'))}")
    backend = BACKENDS[c["backend"]](c) if todo else None
    items_txt = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(ITEMS))
    for k, r in enumerate(todo, 1):
        text, src = transcript_text(r, cfg, c["text_source"], int(c.get("max_chars", 40000)))
        user = USER.format(source=src, text=text, items=items_txt)
        items, ev, ans, ntok, t0 = None, "", "", 0, time.time()
        for attempt in range(int(c.get("retries", 2)) + 1):
            ans, ntok = backend(SYSTEM, user)
            items, ev = parse(ans)
            if items:
                break
        rec = {"id": int(r.id), "split": r.split, "items": items, "evidence": ev, "prompt_tokens": ntok,
               "seconds": round(time.time() - t0, 1), "raw": ans[-2000:]}
        with open(raw_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if items:
            done[int(r.id)] = rec
        if k % 10 == 0 or not items:
            log(f"{k}/{len(todo)} id={r.id} items={items} ({rec['seconds']} s, {ntok} prompt tokens)")

    # features + predictions + quick metrics
    feat_dir = os.path.join(cfg["paths"]["features"], f"llm_{name}")
    rows = []
    for pid, d in done.items():
        it = np.array(d["items"], np.float32)
        save_npz(os.path.join(feat_dir, f"{pid}.npz"), session=np.append(it, it.sum()),
                 session_names=np.array([f"item_{i}" for i in range(1, 9)] + ["total"]))
        rows.append({"id": pid, "pred": float(it.sum())})
    pr = idx[["id", "split", "phq_score"]].merge(pd.DataFrame(rows), on="id", how="inner")
    for split in ("dev", "test"):
        s = pr[pr.split == split]
        save_df(os.path.join(run_dir, "zeroshot", "seed0", f"preds_{split}.csv"),
                pd.DataFrame({"id": s.id, "y": s.phq_score, "pred": s.pred}))
    td = pr[pr.split.isin(["train", "dev"])]
    m = report(td.phq_score, td.pred)
    save_json(os.path.join(run_dir, "metrics_traindev.json"), m)
    n_fail = len(idx) - len(pr)
    log(f"{name}: rated {len(pr)}/{len(idx)} sessions ({n_fail} unparsable)")
    log(f"zero-shot on train+dev (n={m['n']}): CCC {m['ccc']:.3f}  Pearson {m['pearson']:.3f}  RMSE {m['rmse']:.2f}  "
        f"mean pred {td.pred.mean():.2f} vs true {td.phq_score.mean():.2f}  AUROC(PHQ>=10) {m['auroc']:.3f}")
    log("test: python 04_evaluate.py --split test --runs " + run_dir)


if __name__ == "__main__":
    main()
