"""Diagnose weak baselines: pipeline bug or weak signal? (CPU, ~1-2 min)

    python 06_diagnose.py --config configs/data.yaml

1. Sanity: labels aligned, no NaN/constant features, features differ between sessions.
2. Gender check: features must predict gender well (AUROC ~0.9+ for voice features).
   If not -> the features / segments are broken, not the model.
3. Cross-validated CCC on train+dev (219 people, 5x5-fold) - less noisy than one 56-person dev set.
   With and without gender-wise z-scoring and PCA.
4. Univariate Spearman correlations of named features with PHQ-8 (which cues carry signal?).
5. Variance check: pred std vs true std (ridge shrinks toward the mean -> low CCC).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
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


def gender_z(X, g, train_mask):
    """z-score features within each gender using train statistics."""
    X = X.copy()
    for val in np.unique(g):
        m = g == val
        ref = X[m & train_mask]
        if len(ref) < 5:
            continue
        mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0) + 1e-8
        X[m] = (X[m] - mu) / sd
    return X


def cv_ccc(X, y, g, use_gender_z=False, pca=None, reps=5):
    vals, stds = [], []
    for r in range(reps):
        pred = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            Xr = X
            if use_gender_z:
                mask = np.zeros(len(y), bool)
                mask[tr] = True
                Xr = gender_z(X, g, mask)
            steps = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
            if pca:
                steps.append(PCA(n_components=min(pca, len(tr) - 1)))
            model = make_pipeline(*steps, RidgeCV(alphas=np.logspace(-1, 5, 25)))
            pred[te] = model.fit(Xr[tr], y[tr]).predict(Xr[te])
        vals.append(ccc(y, pred))
        stds.append(pred.std())
    return np.mean(vals), np.std(vals), np.mean(stds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--w2v-layers", type=int, nargs="*", default=[3, 6, 7, 9, 12])
    a = ap.parse_args()
    cfg = load_config(a.config)
    feat = cfg["paths"]["features"]
    idx = load_index(cfg["paths"]["index"])
    lab = idx[idx.split.isin(["train", "dev"])].copy()
    ids = lab.id.tolist()

    sets = {"egemaps": [{"name": "egemaps"}], "praat": [{"name": "praat"}],
            "egemaps+praat": [{"name": "egemaps"}, {"name": "praat"}]}
    for L in a.w2v_layers:
        sets[f"w2v2_L{L}"] = [{"name": "w2v2_base", "layer": L}]

    print("== 1. sanity")
    print(f"train+dev sessions: {len(lab)}, PHQ mean {lab.phq_score.mean():.2f} sd {lab.phq_score.std():.2f}, "
          f"PHQ=0: {(lab.phq_score == 0).mean():.0%}, gender: {lab.gender.astype(str).str.strip().str.lower().value_counts().to_dict()}")
    data = {}
    for name, specs in sets.items():
        try:
            X, kept = session_matrix(ids, feat, specs)
        except Exception as e:
            print(f"{name}: cannot load ({e})")
            continue
        sub = lab.set_index("id").loc[kept]
        y = sub.phq_score.to_numpy(float)
        g = sub.gender.astype(str).str.strip().str.lower().to_numpy()
        nan = np.isnan(X).mean()
        const = int((np.nanstd(X, 0) < 1e-8).sum())
        dup = len(X) - len(np.unique(np.round(np.nan_to_num(X), 5), axis=0))
        print(f"{name:14s} X={X.shape}  NaN={nan:.1%}  constant cols={const}  duplicate rows={dup}")
        data[name] = (X, y, g, kept)

    print("\n== 2. gender from features (5-fold AUROC; voice features should give ~0.9+)")
    for name, (X, y, g, _) in data.items():
        gb = (g == "female").astype(int)
        if gb.min() == gb.max():
            print(f"{name}: only one gender")
            continue
        clf = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                            LogisticRegression(C=0.1, max_iter=2000))
        p = cross_val_predict(clf, X, gb, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        print(f"{name:14s} gender AUROC = {roc_auc_score(gb, p):.3f}")

    print("\n== 3. PHQ-8 regression, 5x5-fold CV on train+dev (CCC mean ± sd | pred std; true std "
          f"{lab.phq_score.std():.2f})")
    rows = []
    for name, (X, y, g, _) in data.items():
        for gz in (False, True):
            for pca in (None, 16):
                m, s, ps = cv_ccc(X, y, g, gz, pca)
                rows.append({"features": name, "gender_z": gz, "pca": pca or "-", "ccc": round(m, 3),
                             "ccc_sd": round(s, 3), "pred_std": round(ps, 2)})
    print(pd.DataFrame(rows).sort_values("ccc", ascending=False).to_string(index=False))

    print("\n== 4. strongest single features (Spearman with PHQ-8, train+dev)")
    for name in ("praat", "egemaps"):
        if name not in data:
            continue
        X, y, g, kept = data[name]
        names = np.load(f"{feat}/{name}/{int(kept[0])}.npz")["session_names"]
        res = []
        for j in range(X.shape[1]):
            ok = ~np.isnan(X[:, j])
            if ok.sum() > 15 and np.nanstd(X[ok, j]) > 0:
                r, p = spearmanr(X[ok, j], y[ok])
                res.append((str(names[j]), r, p))
        res = sorted(res, key=lambda t: -abs(t[1]))[:10]
        print(f"-- {name}")
        for n, r, p in res:
            print(f"   {n:40s} rho={r:+.3f}  p={p:.3g}")


if __name__ == "__main__":
    main()
