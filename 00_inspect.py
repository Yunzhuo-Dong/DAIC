"""Print the real column names / formats of the downloaded files (run once, login node is fine).

    python 00_inspect.py --config configs/data.yaml --id 300
"""
import argparse
import os

import soundfile as sf

from edaic.config import load_config
from edaic.data import SPLITS, read_table, session_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--id", type=int, default=300)
    a = ap.parse_args()
    raw = load_config(a.config)["paths"]["raw"]

    for f in [f"{s}_split.csv" for s in SPLITS] + ["Detailed_PHQ8_Labels.csv"]:
        p = os.path.join(raw, "labels", f)
        if os.path.exists(p):
            df = read_table(p)
            print(f"\n== labels/{f}: {len(df)} rows\n{list(df.columns)}\n{df.head(2).to_string()}")
        else:
            print(f"\n== labels/{f}: MISSING")

    sp = session_paths(raw, a.id)
    tr = read_table(sp["transcript"])
    print(f"\n== transcript: {len(tr)} rows\n{list(tr.columns)}\n{tr.head(3).to_string()}")
    for k in ("egemaps", "mfcc", "openface"):
        df = read_table(sp[k])
        print(f"\n== {k}: {df.shape}\nfirst columns: {list(df.columns[:6])}")
    info = sf.info(sp["audio"])
    print(f"\n== audio: {info.samplerate} Hz, {info.channels} ch, {info.duration / 60:.1f} min")


if __name__ == "__main__":
    main()
