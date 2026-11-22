"""Why do some models fail on the AI-only test set? Post-hoc analysis, no model selection on test.

    python 09_shift_analysis.py --config configs/data.yaml \
        --runs $EDAIC_RUNS/e2v_full_ridge/e2v_plus_large_full $EDAIC_RUNS/w2v2_large_mean_ridge/w2v2_large_Lmean ...

A. Test predictions of each variant (seed-averaged): Pearson vs CCC, bias (mean pred - mean true), pred SD,
   and whether errors grow when the participant speaks little (participant_share / participant_min from the
   participant_timing features). Also CCC within tertiles of participant share.
B. Leave-AI-out inside train+dev (never touches test): ridge trained on the WoZ sessions (IDs < 600),
   evaluated on the 30 AI sessions of train+dev, for each feature set in --features.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edaic.assemble import session_matrix
from edaic.config import load_config
from edaic.data import load_index
from edaic.metrics import ccc


def timing_table(feat_root, ids):
    rows = {}
    for pid in ids:
        p = os.path.join(feat_root, "participant_timing", f"{int(pid)}.npz")
        if os.path.exists(p):
            z = np.load(p)
            rows[int(pid)] = dict(zip(z["session_names"].tolist(), z["session"].tolist()))
    return pd.DataFrame.from_dict(rows, orient="index")


def seed_avg(vdir, split="test"):
    files = sorted(glob.glob(os.path.join(vdir, "seed*", f"preds_{split}.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files]).groupby("id", as_index=False).agg(y=("y", "first"),
                                                                                         pred=("pred", "mean"))


def part_a(runs, tim):
    print("== A. test-set behaviour (seed-averaged predictions)")
    rows, terc = [], []
    for v in runs:
        d = seed_avg(v)
        if d is None:
            print(f"   no test predictions in {v}")
            continue
        d = d.join(tim[["participant_share", "participant_min"]], on="id")
        err = (d.pred - d.y).abs()
        name = f"{os.path.basename(os.path.dirname(v))}/{os.path.basename(v)}"
        rows.append({"model": name, "pearson": pearsonr(d.y, d.pred)[0], "ccc": ccc(d.y, d.pred),
                     "mean_true": d.y.mean(), "mean_pred": d.pred.mean(), "bias": d.pred.mean() - d.y.mean(),
                     "sd_true": d.y.std(), "sd_pred": d.pred.std(),
                     "rho_abs_err_vs_share": spearmanr(err, d.participant_share, nan_policy="omit")[0],
                     "rho_abs_err_vs_min": spearmanr(err, d.participant_min, nan_policy="omit")[0]})
        d["tertile"] = pd.qcut(d.participant_share.rank(method="first"), 3, labels=["low share", "mid share", "high share"])
        for t, g in d.groupby("tertile", observed=True):
            terc.append({"model": name, "tertile": t, "n": len(g), "ccc": ccc(g.y, g.pred),
                         "pearson": pearsonr(g.y, g.pred)[0] if g.pred.std() > 0 else np.nan,
                         "share_range": f"{g.participant_share.min():.2f}-{g.participant_share.max():.2f}"})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\n   CCC within tertiles of participant speech share (test):")
    print(pd.DataFrame(terc).pivot(index="model", columns="tertile", values="ccc").round(3).to_string())


def part_b(cfg, idx, features):
    print("\n== B. leave-AI-out inside train+dev (train on WoZ, evaluate on the AI sessions of train+dev)")
    lab = idx[idx.split.isin(["train", "dev"])]
    woz, ai = lab[lab.id < 600], lab[lab.id >= 600]
    print(f"   WoZ n={len(woz)} (PHQ mean {woz.phq_score.mean():.2f}), AI n={len(ai)} (PHQ mean {ai.phq_score.mean():.2f})")
    rows = []
    for spec in features:
        name, _, layer = spec.partition(":")
        fs = [{"name": name, **({"layer": layer} if layer else {})}]
        try:
            Xw, kw = session_matrix(woz.id.tolist(), cfg["paths"]["features"], fs)
            Xa, ka = session_matrix(ai.id.tolist(), cfg["paths"]["features"], fs)
        except Exception as e:
            print(f"   {spec}: cannot load ({e})")
            continue
        yw = lab.set_index("id").loc[kw, "phq_score"].to_numpy(float)
        ya = lab.set_index("id").loc[ka, "phq_score"].to_numpy(float)
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-1, 5, 25))).fit(Xw, yw)
        p = m.predict(Xa)
        rows.append({"features": spec, "n_ai": len(ya), "ccc_ai": ccc(ya, p), "pearson_ai": pearsonr(ya, p)[0],
                     "bias_ai": p.mean() - ya.mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("   (30 sessions: CIs are wide; use the direction, not the decimals)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--runs", nargs="*", default=[], help="variant folders with seed*/preds_test.csv")
    ap.add_argument("--features", nargs="*",
                    default=["e2v_plus_large_full", "w2v2_large:mean", "w2v2_large_full:mean", "participant_timing"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    idx = load_index(cfg["paths"]["index"])
    tim = timing_table(cfg["paths"]["features"], idx.id)
    if a.runs:
        part_a(a.runs, tim)
    part_b(cfg, idx, a.features)


if __name__ == "__main__":
    main()
