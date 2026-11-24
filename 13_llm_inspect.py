"""Look at how the LLM rated each session: item scores, its one-sentence evidence, and the true labels.

    python 13_llm_inspect.py                                   # summary + largest errors (train+dev)
    python 13_llm_inspect.py --id 321 --show-text              # one session in detail, with the transcript it read
    python 13_llm_inspect.py --split test --top 10             # largest errors on test
    python 13_llm_inspect.py --name qwen38_27b_full            # another LLM run

Reads $EDAIC_RUNS/llm_<name>/raw_outputs.jsonl (written by 11_llm_score.py) and the session index.
Writes $EDAIC_RUNS/llm_<name>/review_<split>.csv (one row per session) for reading in Excel.
"""
import argparse
import json
import os
import textwrap

import numpy as np
import pandas as pd

from edaic.config import load_config
from edaic.data import load_index
from edaic.metrics import ccc

ITEM_SHORT = ["interest", "mood", "sleep", "energy", "appetite", "selfworth", "concentration", "psychomotor"]
SPLITS = {"traindev": ["train", "dev"], "train": ["train"], "dev": ["dev"], "test": ["test"], "all": ["train", "dev", "test"]}


def load_outputs(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        if d.get("items") and len(d["items"]) == 8:
            out[int(d["id"])] = d  # last valid answer wins
    return out


def wrap(s, indent="    "):
    return textwrap.fill(str(s), 110, initial_indent=indent, subsequent_indent=indent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/llm_qwen.yaml")
    ap.add_argument("--name", default=None, help="LLM run name (default: llm.name in the config)")
    ap.add_argument("--split", default="traindev", choices=list(SPLITS))
    ap.add_argument("--top", type=int, default=5, help="how many over- and under-estimated sessions to show")
    ap.add_argument("--id", type=int, default=None, help="show one session in detail")
    ap.add_argument("--show-text", action="store_true", help="with --id: print the transcript given to the LLM")
    a = ap.parse_args()
    cfg = load_config(a.config, [])
    name = a.name or cfg["llm"]["name"]
    run_dir = os.path.join(cfg["paths"]["runs"], f"llm_{name}")
    outs = load_outputs(os.path.join(run_dir, "raw_outputs.jsonl"))
    idx = load_index(cfg["paths"]["index"])
    has_items = all(f"item_{i}" in idx for i in range(1, 9))

    rows = []
    for r in idx.itertuples():
        d = outs.get(int(r.id))
        if d is None:
            continue
        row = {"id": int(r.id), "split": r.split, "y": r.phq_score, "pred": int(sum(d["items"])),
               "evidence": d.get("evidence", "")}
        for j, k in enumerate(ITEM_SHORT):
            row[f"p_{k}"] = d["items"][j]
            row[f"t_{k}"] = getattr(r, f"item_{j + 1}") if has_items else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["err"] = df.pred - df.y

    # ------------------------------------------------ one session
    if a.id is not None:
        s = df[df.id == a.id]
        if not len(s):
            raise SystemExit(f"id {a.id} not rated in {run_dir}")
        s = s.iloc[0]
        print(f"\n== session {a.id} ({s.split})   PHQ-8 true {s.y:.0f}   LLM {s.pred}   error {s.err:+.0f}")
        print(f"   {'item':14s} LLM  true")
        for k in ITEM_SHORT:
            t = "" if pd.isna(s[f't_{k}']) else f"{s[f't_{k}']:.0f}"
            print(f"   {k:14s} {s[f'p_{k}']:3d}  {t:>4s}")
        print("\n   LLM evidence:")
        print(wrap(s.evidence))
        if a.show_text:
            from importlib import import_module
            score = import_module("11_llm_score")
            r = idx[idx.id == a.id].iloc[0]
            c = cfg["llm"]
            text, src = score.transcript_text(r, cfg, c["text_source"], int(c.get("max_chars", 40000)))
            print(f"\n   transcript given to the LLM ({src}, {len(text.split())} words):")
            print(wrap(text.replace("\n", " | ")))
        return

    # ------------------------------------------------ summary
    sub = df[df.split.isin(SPLITS[a.split])].copy()
    print(f"\n== {name}: {len(sub)} sessions ({a.split})")
    print(f"PHQ-8 total: CCC {ccc(sub.y, sub.pred):.3f}   mean LLM {sub.pred.mean():.2f} vs true {sub.y.mean():.2f}   "
          f"MAE {sub.err.abs().mean():.2f}")
    if sub[[f't_{k}' for k in ITEM_SHORT]].notna().all(axis=1).any():
        it = sub.dropna(subset=[f't_{k}' for k in ITEM_SHORT])
        print(f"\nper item (n={len(it)} with item labels):")
        print(f"   {'item':14s} {'mean LLM':>8s} {'mean true':>9s} {'exact':>6s} {'+-1':>5s} {'Pearson':>8s} {'CCC':>6s}")
        for k in ITEM_SHORT:
            p, t = it[f"p_{k}"].astype(float), it[f"t_{k}"].astype(float)
            r = np.corrcoef(p, t)[0, 1] if p.std() > 0 and t.std() > 0 else np.nan
            print(f"   {k:14s} {p.mean():8.2f} {t.mean():9.2f} {np.mean(p == t):6.0%} {np.mean((p - t).abs() <= 1):5.0%} "
                  f"{r:8.2f} {ccc(t, p):6.2f}")
    else:
        print("\n(no per-item labels for this split; only the total is compared)")

    for title, part in (("most UNDER-estimated", sub.nsmallest(a.top, "err")),
                        ("most OVER-estimated", sub.nlargest(a.top, "err"))):
        print(f"\n== {title}")
        for s in part.itertuples():
            items = " ".join(str(getattr(s, f"p_{k}")) for k in ITEM_SHORT)
            print(f"-- id {s.id} ({s.split})  true {s.y:.0f}  LLM {s.pred}  error {s.err:+.0f}   items [{items}]")
            print(wrap(s.evidence))

    out = os.path.join(run_dir, f"review_{a.split}.csv")
    sub.sort_values("err").to_csv(out, index=False)
    print(f"\nwrote {out}  (all sessions, sorted by error; open in Excel)")
    print(f"one session in detail: python 13_llm_inspect.py --name {name} --id <id> --show-text")


if __name__ == "__main__":
    main()
