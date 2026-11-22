"""Check / extend the Ellie phrase list from the data (login node, < 1 min).

    python 00_ellie_check.py --config configs/data.yaml

Prints: (1) how many sessions each listed phrase hits, (2) frequent 5-grams shared by many
sessions that the list does not cover yet (Ellie's script repeats; participants rarely do),
(3) the first transcript row of sessions without an Ellie intro (experimenter talk?),
(4) a before/after example. Add real Ellie lines from (2) to configs/ellie_phrases.txt.
"""
import argparse
import os
import re
from collections import Counter, defaultdict

from edaic.config import load_config
from edaic.data import clean_transcript, load_index, load_phrases, load_transcript, normalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--min-df", type=int, default=8)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--example", type=int, default=305)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ccfg = cfg["cleaning"]
    phrases = load_phrases(ccfg["phrases_file"])
    idx = load_index(cfg["paths"]["index"])

    trs = {int(r.id): load_transcript(r.transcript_path) for r in idx.itertuples() if os.path.exists(r.transcript_path)}
    norm = {pid: [normalise(t) for t in tr["text"]] for pid, tr in trs.items()}

    print(f"== (1) phrase hits over {len(norm)} sessions")
    for p in phrases:
        pat = re.compile(r"\b" + re.escape(p) + r"\b")
        n = sum(any(pat.search(t) for t in rows) for rows in norm.values())
        print(f"{n:4d}  {p}")

    df, example = Counter(), {}
    for pid, rows in norm.items():
        seen = set()
        for t in rows:
            w = t.split()
            for i in range(len(w) - a.n + 1):
                g = " ".join(w[i:i + a.n])
                seen.add(g)
                example.setdefault(g, t)
        df.update(seen)
    covered = lambda g: any(g in p for p in phrases)
    cands = [(g, c) for g, c in df.most_common() if c >= a.min_df and not covered(g)]
    print(f"\n== (2) uncovered {a.n}-grams in >= {a.min_df} sessions (top {a.top}); Ellie lines -> add to list")
    for g, c in cands[:a.top]:
        print(f"{c:4d}  {g:45s} | {example[g][:90]}")

    no_intro = defaultdict(list)
    for pid, tr in trs.items():
        _, st = clean_transcript(tr, ccfg, phrases)
        if not st["intro_found"]:
            no_intro["x"].append((pid, tr["text"].iloc[0][:100] if len(tr) else ""))
    print(f"\n== (3) {len(no_intro['x'])} sessions without Ellie intro; first rows of 12:")
    for pid, t in no_intro["x"][:12]:
        print(f"{pid}: {t}")

    if a.example in trs:
        clean, st = clean_transcript(trs[a.example], ccfg, phrases)
        print(f"\n== (4) session {a.example}: {st}")
        for (_, r0), in zip(trs[a.example].head(8).iterrows()):
            print(f"RAW   {r0.start:7.1f} {r0.text[:100]}")
        for _, r in clean.head(8).iterrows():
            print(f"CLEAN {r.start:7.1f} {'[mixed] ' if r.mixed else ''}{r.text[:100]}")


if __name__ == "__main__":
    main()
