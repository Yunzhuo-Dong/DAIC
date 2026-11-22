"""Summarise experiments; final test evaluation; late fusion; significance tests.

    python 04_evaluate.py --runs $EDAIC_RUNS/egemaps_ridge $EDAIC_RUNS/w2v_base_layers_ridge
    python 04_evaluate.py --runs $EDAIC_RUNS/egemaps_ridge --split test          # only at the end
    python 04_evaluate.py --fuse $EDAIC_RUNS/egemaps_ridge/egemaps $EDAIC_RUNS/praat_ridge/praat
    python 04_evaluate.py --compare $EDAIC_RUNS/A/variant $EDAIC_RUNS/B/variant

A "variant" folder contains seed*/ subfolders. Seeds are averaged in two ways:
mean ± std of per-seed metrics, and metrics of the seed-averaged prediction (+ bootstrap CI).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from edaic.metrics import bootstrap_ci, classif_report, paired_permutation, report
from edaic.utils import save_df


def variant_preds(vdir, split):
    """Seed-averaged predictions and per-seed frames of one variant."""
    files = sorted(glob.glob(os.path.join(vdir, "seed*", f"preds_{split}.csv")))
    if not files:
        return None, []
    frames = [pd.read_csv(f) for f in files]
    agg = {"y": ("y", "first"), "pred": ("pred", "mean")}
    if "label" in frames[0]:
        agg["label"] = ("label", "mean")
    avg = pd.concat(frames).groupby("id", as_index=False).agg(**agg)
    if "label" in avg:
        avg["label"] = (avg["label"] >= 0.5).astype(int)
    return avg, frames


def cv_metrics(vdir):
    p = os.path.join(vdir, "metrics_cv.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        m = json.load(f)
    return {k: v for k, v in m.items() if k.startswith("cv_") and k not in ("cv_reps", "cv_folds")}


def summarise(vdir, split, thr, n_boot):
    avg, frames = variant_preds(vdir, split)
    if avg is None or not len(avg):
        return None
    base = {"experiment": os.path.basename(os.path.dirname(vdir)), "variant": os.path.basename(vdir),
            "n": int(len(avg)), "seeds": len(frames)}
    if "label" in avg:  # classification head
        ens = classif_report(avg.y, avg.pred, avg.label)
        lo, hi = bootstrap_ci(avg.y.values, avg.pred.values, metric=_auroc, n_boot=n_boot)
        base.update({"task": "clf", **cv_metrics(vdir), "auroc": ens["auroc"], "auroc_ci_lo": lo, "auroc_ci_hi": hi,
                     "f1_macro": ens["f1_macro"], "f1_pos": ens["f1_pos"], "bal_acc": ens["bal_acc"]})
        return base
    per_seed = pd.DataFrame([report(f.y, f.pred, thr) for f in frames])
    ens = report(avg.y, avg.pred, thr)
    lo, hi = bootstrap_ci(avg.y.values, avg.pred.values, n_boot=n_boot)
    base.update({"task": "reg", **cv_metrics(vdir),
                 "ccc_mean": per_seed.ccc.mean(), "ccc_std": per_seed.ccc.std(ddof=0),
                 "ccc_ens": ens["ccc"], "ccc_ci_lo": lo, "ccc_ci_hi": hi,
                 "rmse": ens["rmse"], "mae": ens["mae"], "pearson": ens["pearson"],
                 "auroc": ens.get("auroc"), "pred_std": ens["pred_std"]})
    return base


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if 0 < np.sum(y) < len(y) else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="experiment folders")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--fuse", nargs="*", default=[], help="variant folders to late-fuse (mean of predictions)")
    ap.add_argument("--compare", nargs=2, help="two variant folders: paired permutation test on CCC")
    ap.add_argument("--threshold", type=float, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    if a.runs:
        rows = []
        for exp in a.runs:
            for vdir in sorted(d for d in glob.glob(os.path.join(exp, "*")) if os.path.isdir(d)):
                r = summarise(vdir, a.split, a.threshold, a.n_boot)
                if r:
                    rows.append(r)
        df = pd.DataFrame(rows)
        for task, key in (("reg", "ccc"), ("clf", "auroc")):
            sub = df[df.task == task].dropna(axis=1, how="all")
            if not len(sub):
                continue
            sort = f"cv_{key}" if f"cv_{key}" in sub else (f"{key}_mean" if task == "reg" else key)
            title = "PHQ-8 regression" if task == "reg" else "PHQ-8 >= 10 classification"
            print(f"\n== {title} ({a.split}; cv_* = 5x5-fold CV on train+dev, sorted by {sort})")
            show = (["experiment", "variant", "cv_ccc", "cv_ccc_sd", "cv_pred_std", "ccc_ens", "ccc_ci_lo",
                     "ccc_ci_hi", "rmse", "auroc", "pred_std"] if task == "reg" else
                    ["experiment", "variant", "cv_auroc", "cv_auroc_sd", "cv_f1_macro", "cv_bal_acc", "auroc",
                     "auroc_ci_lo", "auroc_ci_hi", "f1_macro", "bal_acc"])
            show = [c for c in show if c in sub]
            print(sub.sort_values(sort, ascending=False)[show].round(3).to_string(index=False))
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.runs[0])), f"summary_{a.split}.csv")
        save_df(out, df)
        print(f"\nwrote {out}")

    if a.fuse:
        preds = [variant_preds(v, a.split)[0].set_index("id") for v in a.fuse]
        common = sorted(set.intersection(*(set(p.index) for p in preds)))
        y = preds[0].loc[common, "y"].values
        p = np.mean([q.loc[common, "pred"].values for q in preds], 0)
        r = report(y, p, a.threshold)
        lo, hi = bootstrap_ci(y, p, n_boot=a.n_boot)
        print(f"\nlate fusion of {len(preds)} ({a.split}, n={len(common)}): "
              f"CCC {r['ccc']:.3f} [{lo:.3f}, {hi:.3f}]  RMSE {r['rmse']:.2f}  MAE {r['mae']:.2f}")

    if a.compare:
        pa, pb = (variant_preds(v, a.split)[0].set_index("id") for v in a.compare)
        common = sorted(set(pa.index) & set(pb.index))
        diff, pval = paired_permutation(pa.loc[common, "y"].values, pa.loc[common, "pred"].values,
                                        pb.loc[common, "pred"].values)
        print(f"\nCCC(A) - CCC(B) = {diff:+.3f}, permutation p = {pval:.4f} (n={len(common)})")


if __name__ == "__main__":
    main()
