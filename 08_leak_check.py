"""Where does the signal in non-participant audio come from? (CPU, ~1 min)

    python 08_leak_check.py --config configs/data.yaml

For each feature set (default: wav2vec2-large layer-mean of participant / others / full recording):
  A. predict the PARTICIPANT's gender      -> high for "others" = participant voice leaks into others
  B. predict WoZ (id 300-492) vs AI (600+)  -> high = vectors encode recording condition / sub-corpus
  C. PHQ-8 CV CCC within WoZ-only and AI-only sessions (removes the sub-corpus confound)
Plus: PHQ-8 by sub-corpus, and how well the sub-corpus flag ALONE predicts PHQ-8.
All on train+dev (219 people), 5x5-fold CV.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from edaic.assemble import session_matrix
from edaic.config import load_config
from edaic.data import load_index
from edaic.metrics import ccc


def clf_auc(X, y, reps=5):
    if np.bincount(np.asarray(y, int), minlength=2).min() < 5:
        return np.nan
    aucs = []
    for r in range(reps):
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=3000))
        p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=r), method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs)


def reg_ccc(X, y, reps=5):
    vals = []
    for r in range(reps):
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-1, 5, 25)))
        p = cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=r))
        vals.append(ccc(y, p))
    return np.mean(vals), np.std(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--features", nargs="*", default=["w2v2_large", "w2v2_large_others", "w2v2_large_full"])
    ap.add_argument("--layer", default="mean")
    a = ap.parse_args()
    cfg = load_config(a.config)
    idx = load_index(cfg["paths"]["index"])
    lab = idx[idx.split.isin(["train", "dev"])].copy()
    lab["woz"] = (lab.id < 600).astype(int)
    lab["female"] = (lab.gender.astype(str).str.strip().str.lower() == "female").astype(int)

    print("== PHQ-8 by sub-corpus (train+dev)")
    print(lab.groupby("woz").phq_score.agg(["count", "mean", "std"]).rename(index={1: "WoZ", 0: "AI"}).round(2).to_string())
    y = lab.phq_score.to_numpy(float)
    m, s = reg_ccc(lab[["woz"]].to_numpy(float), y)
    print(f"PHQ-8 predicted from the WoZ/AI flag alone: CV CCC {m:.3f} ± {s:.3f}")

    rows = []
    for f in a.features:
        try:
            X, kept = session_matrix(lab.id.tolist(), cfg["paths"]["features"], [{"name": f, "layer": a.layer}])
        except Exception as e:
            print(f"{f}: cannot load ({e})")
            continue
        sub = lab.set_index("id").loc[kept]
        y, woz = sub.phq_score.to_numpy(float), sub.woz.to_numpy()
        row = {"features": f, "n": len(kept),
               "A_gender_auroc": clf_auc(X, sub.female.to_numpy()),
               "B_woz_vs_ai_auroc": clf_auc(X, woz),
               "phq_ccc_all": reg_ccc(X, y)[0]}
        for name, mask in (("WoZ", woz == 1), ("AI", woz == 0)):
            row[f"C_phq_ccc_{name}_only"] = reg_ccc(X[mask], y[mask])[0] if mask.sum() >= 40 else np.nan
        rows.append(row)
    print("\n== leak check (5x5-fold CV on train+dev)")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nA high for others -> participant voice in 'others'; B high -> recording condition encoded;"
          "\nC drops vs phq_ccc_all -> part of the signal was the sub-corpus, not the person.")


if __name__ == "__main__":
    main()
