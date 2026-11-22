"""Extract one feature set for all sessions and cache it ($EDAIC_FEAT/<name>/<id>.npz).

    python 02_extract.py --config configs/extract_egemaps.yaml --workers 16
    python 02_extract.py --config configs/extract_wav2vec2_base.yaml          # GPU job
Existing files are skipped (safe to re-run after a crash). --limit 3 for a quick test.
"""
import argparse
import os
import traceback
from multiprocessing import Pool

import pandas as pd

from edaic.config import load_config, save_config
from edaic.data import load_index, session_segments
from edaic.features import build_extractor
from edaic.utils import log, save_npz

_EX = None
_CFG = None


def _init(cfg, device):
    global _EX, _CFG
    _CFG = cfg
    _EX = build_extractor(cfg)
    _EX.setup(device)


def _one(row):
    out = os.path.join(_CFG["paths"]["features"], _EX.name, f"{int(row['id'])}.npz")
    if os.path.exists(out) and not _CFG.get("_overwrite"):
        return row["id"], "skip"
    try:
        tr, segs, _ = session_segments(row["id"], row["transcript_path"], _CFG, row["audio_path"])
        save_npz(out, **_EX.extract(row, segs, tr))
        return row["id"], "ok"
    except Exception as e:  # keep going; failures are listed at the end
        return row["id"], f"FAILED {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. extractor.fp16=false")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="run a GPU extractor on CPU (tests only)")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    cfg["_overwrite"] = a.overwrite
    ex = build_extractor(cfg)
    device = "cpu"
    if ex.needs_gpu:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif not a.allow_cpu:
            raise SystemExit("this extractor needs a GPU (submit via slurm/gpu_job.sh) or pass --allow-cpu")

    idx = load_index(cfg["paths"]["index"])
    if a.ids:
        idx = idx[idx["id"].isin(a.ids)]
    if a.limit:
        idx = idx.head(a.limit)
    rows = [r for _, r in idx.iterrows()]
    out_dir = os.path.join(cfg["paths"]["features"], ex.name)
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "extract_config.yaml"))
    log(f"{ex.name}: {len(rows)} sessions on {device}, workers={a.workers} -> {out_dir}")

    results = []
    if a.workers > 1 and not ex.needs_gpu:
        with Pool(a.workers, initializer=_init, initargs=(cfg, device)) as pool:
            for i, r in enumerate(pool.imap_unordered(_one, rows), 1):
                results.append(r)
                if i % 25 == 0:
                    log(f"{i}/{len(rows)}")
    else:
        _init(cfg, device)
        for i, row in enumerate(rows, 1):
            results.append(_one(row))
            if i % 10 == 0:
                log(f"{i}/{len(rows)}")

    res = pd.DataFrame(results, columns=["id", "status"])
    fails = res[res.status.str.startswith("FAILED")]
    log(f"done: ok={sum(res.status == 'ok')} skipped={sum(res.status == 'skip')} failed={len(fails)}")
    for _, r in fails.iterrows():
        print(r["id"], r["status"])


if __name__ == "__main__":
    main()
