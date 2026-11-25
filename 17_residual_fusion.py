"""Multimodal residual fusion: text backbone + nonverbal residual (CPU, a few minutes).

    final = text_prediction + lambda * residual_prediction

1. Text backbone: ridge on the 9 zero-shot LLM features (same as llm_part_ridge).
2. Out-of-fold text predictions on the training data (inner 5-fold, repeated) -> residual r = y - OOF.
3. Residual model: ridge on standardised nonverbal features (median-imputed; PCA for large blocks, fitted per fold).
4. alpha (residual ridge) and lambda in {0, .25, .5, .75, 1} chosen by inner CV on the training data only;
   lambda = 0 means "text only", so the model can decide that the nonverbal block does not help.
5. Sessions whose nonverbal block is mostly missing (> max_missing NaN) get residual 0 (text only).

Evaluation (all selection inside the training folds):
    5x5-fold CV on train+dev (219)     -> cv_ccc, and the paired difference to E0 on the same folds
    train -> dev                       -> preds_dev.csv
    train -> test  (only with --with-test, once, after the design is frozen) -> preds_test.csv
Experiments (configs/multimodal.yaml): E0 text, E1 +audio, E2 +visual (speaker states), E3 +audio+visual,
E4 global visual pooling (control), and E3 with nonverbal rows shuffled across sessions (permutation control).

    python 17_residual_fusion.py --config configs/multimodal.yaml
    python 17_residual_fusion.py --config configs/multimodal.yaml --with-test       # final, once
    python 04_evaluate.py --split test --runs $EDAIC_RUNS/mm_residual
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from edaic.config import load_config, save_config
from edaic.data import load_index
from edaic.metrics import ccc, report, rmse
from edaic.models import build_sklearn
from edaic.utils import log, save_df, save_json


# ---------------------------------------------------------------- data
def load_block(feat_root, name, ids):
    """[N, D] matrix (NaN rows for sessions without the file) and feature names."""
    vecs, names, dim = {}, None, None
    for pid in ids:
        p = os.path.join(feat_root, name, f"{int(pid)}.npz")
        if os.path.exists(p):
            z = np.load(p)
            vecs[pid] = z["session"].astype(np.float64)
            dim = len(vecs[pid])
            if names is None:
                names = [str(s) for s in z["session_names"]] if "session_names" in z.files else None
    if dim is None:
        raise SystemExit(f"no features found for {name} in {feat_root}/{name}")
    X = np.full((len(ids), dim), np.nan)
    for i, pid in enumerate(ids):
        if pid in vecs:
            X[i] = vecs[pid]
    names = names or [f"{name}_{j}" for j in range(dim)]
    return X, [f"{name}:{n}" for n in names]


class Nonverbal:
    """Per-spec median imputation + scaling (+ PCA), concatenated, then ridge with a fixed alpha."""

    def __init__(self, specs):
        self.specs = specs

    def fit_transform(self, blocks):
        self.steps, out, self.names = [], [], []
        for spec, X in zip(self.specs, blocks):
            imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
            sc = StandardScaler().fit(imp.transform(X))
            Z = sc.transform(imp.transform(X))
            pca, s = None, None
            if spec.get("pca") and Z.shape[1] > int(spec["pca"]):
                pca = PCA(min(int(spec["pca"]), len(Z) - 1), random_state=0).fit(Z)
                Z = pca.transform(Z)
                s = Z.std(0, keepdims=True) + 1e-8      # components rescaled to unit variance like the other features
                Z = Z / s
                self.names += [f"{spec['name']}:pc{j + 1}" for j in range(Z.shape[1])]
            else:
                self.names += spec["_names"]
            self.steps.append((imp, sc, pca, s))
            out.append(Z)
        return np.concatenate(out, 1)

    def transform(self, blocks):
        out = []
        for (imp, sc, pca, s), X in zip(self.steps, blocks):
            Z = sc.transform(imp.transform(X))
            if pca is not None:
                Z = pca.transform(Z) / s
            out.append(Z)
        return np.concatenate(out, 1)


def missing_share(blocks):
    return np.isnan(np.concatenate(blocks, 1)).mean(1) if blocks else np.zeros(0)


# ---------------------------------------------------------------- model
def oof_text(fcfg, Xt, y, seed=0):
    """Out-of-fold text predictions on the training data (repeated k-fold, averaged)."""
    mcfg = {"type": "ridge", "alpha_logspace": fcfg["text_alpha_logspace"]}
    pred = np.zeros((int(fcfg["inner_reps"]), len(y)))
    for r in range(int(fcfg["inner_reps"])):
        for tr, te in KFold(int(fcfg["inner_folds"]), shuffle=True, random_state=1000 + seed * 10 + r).split(Xt):
            pred[r, te] = build_sklearn(mcfg).fit(Xt[tr], y[tr]).predict(Xt[te])
    return pred.mean(0)


def fit_model(fcfg, specs, Xt, blocks, y, seed=0):
    """Returns a predict(Xt, blocks) closure and the chosen (alpha, lambda)."""
    lo, hi = fcfg["y_range"]
    text = build_sklearn({"type": "ridge", "alpha_logspace": fcfg["text_alpha_logspace"]}).fit(Xt, y)
    if not specs:
        return (lambda Xt_, b_: np.clip(text.predict(Xt_), lo, hi)), {"alpha": None, "lambda": 0.0}, None

    oof = oof_text(fcfg, Xt, y, seed)
    res = y - oof
    use = missing_share(blocks) <= float(fcfg["max_missing"])
    alphas = np.logspace(*fcfg["alpha_logspace"])
    lambdas = np.array(fcfg["lambdas"], float)
    # inner CV for (alpha, lambda): residual model fitted on inner-train, applied to inner-test
    delta = np.zeros((len(alphas), len(y)))
    for tr, te in KFold(int(fcfg["inner_folds"]), shuffle=True, random_state=2000 + seed).split(Xt):
        tr_u = tr[use[tr]]
        nv = Nonverbal(specs)
        Ztr = nv.fit_transform([b[tr_u] for b in blocks])
        Zte = nv.transform([b[te] for b in blocks])
        for k, a in enumerate(alphas):
            d = Ridge(alpha=a).fit(Ztr, res[tr_u]).predict(Zte)
            delta[k, te] = np.where(use[te], d, 0.0)
    score = np.array([[ccc(y, np.clip(oof + lam * delta[k], lo, hi)) for lam in lambdas] for k in range(len(alphas))])
    best = np.max(score)
    cand = [(k, j) for k in range(len(alphas)) for j in range(len(lambdas)) if score[k, j] >= best - 1e-9]
    k, j = min(cand, key=lambda kj: (lambdas[kj[1]], -alphas[kj[0]]))  # ties: smaller lambda, stronger alpha
    a_best, lam = float(alphas[k]), float(lambdas[j])
    nv = Nonverbal(specs)
    Z = nv.fit_transform([b[use] for b in blocks])
    rr = Ridge(alpha=a_best).fit(Z, res[use])

    def predict(Xt_, b_):
        d = rr.predict(nv.transform(b_))
        d = np.where(missing_share(b_) <= float(fcfg["max_missing"]), d, 0.0)
        return np.clip(text.predict(Xt_) + lam * d, lo, hi)

    coef = pd.DataFrame({"feature": nv.names, "coef": rr.coef_}).sort_values("coef", key=np.abs, ascending=False)
    return predict, {"alpha": a_best, "lambda": lam, "inner_ccc": float(best),
                     "inner_ccc_text_only": float(score[:, 0].max())}, coef


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multimodal.yaml")
    ap.add_argument("--with-test", action="store_true", help="also predict test (only once, at the end)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiments")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    fcfg = cfg["fusion"]
    feat = cfg["paths"]["features"]
    out_root = os.path.join(cfg["paths"]["runs"], fcfg["name"])
    os.makedirs(out_root, exist_ok=True)
    save_config(cfg, os.path.join(out_root, "config.yaml"))

    idx = load_index(cfg["paths"]["index"])
    idx = idx[idx.phq_score.notna()]
    ids = {s: idx.loc[idx.split == s, "id"].astype(int).tolist() for s in ("train", "dev", "test")}
    all_ids = ids["train"] + ids["dev"] + ids["test"]
    y_all = dict(zip(idx.id.astype(int), idx.phq_score.astype(float)))

    Xt_all, _ = load_block(feat, fcfg["text"], all_ids)
    if np.isnan(Xt_all).any():
        bad = [all_ids[i] for i in np.flatnonzero(np.isnan(Xt_all).any(1))]
        raise SystemExit(f"text features missing for {bad[:10]}")
    blocks_all = {}
    for group, specs in fcfg["blocks"].items():
        for spec in specs:
            if spec["name"] not in blocks_all:
                X, names = load_block(feat, spec["name"], all_ids)
                blocks_all[spec["name"]] = (X, names)
                miss = np.isnan(X).all(1)
                log(f"{spec['name']:22s} {X.shape[1]:5d} features, {miss.sum():3d} sessions missing "
                    f"(train {miss[:len(ids['train'])].sum()}, test {miss[-len(ids['test']):].sum()})")
    pos = {pid: i for i, pid in enumerate(all_ids)}

    def rows(sel, perm=None):
        return np.array([pos[p] for p in sel]) if perm is None else perm[np.array([pos[p] for p in sel])]

    exps = dict(fcfg["experiments"])
    pc = fcfg.get("permutation_control")
    if pc:
        exps[f"{pc}_PERMUTED"] = exps[pc]
    if a.only:
        exps = {k: v for k, v in exps.items() if k in a.only or k == "E0_text"}
    rng = np.random.default_rng(0)
    perm_rows = rng.permutation(len(all_ids))

    cvids = ids["train"] + ids["dev"]
    ycv = np.array([y_all[p] for p in cvids])
    reps, folds = int(fcfg["cv_reps"]), int(fcfg["cv_folds"])
    summary, cv_rep_ccc = [], {}
    for exp, groups in exps.items():
        specs = [dict(s, _names=blocks_all[s["name"]][1]) for g in groups for s in fcfg["blocks"][g]]
        perm = perm_rows if exp.endswith("_PERMUTED") else None
        get = lambda sel: (Xt_all[rows(sel)], [blocks_all[s["name"]][0][rows(sel, perm)] for s in specs])

        # ---- outer 5x5 CV on train+dev
        Xt_cv, b_cv = get(cvids)
        rep_ccc, rep_rmse, rep_std, lams = [], [], [], []
        for r in range(reps):
            pred = np.zeros(len(ycv))
            for tr, te in KFold(folds, shuffle=True, random_state=r).split(Xt_cv):
                f, sel, _ = fit_model(fcfg, specs, Xt_cv[tr], [b[tr] for b in b_cv], ycv[tr], seed=r)
                pred[te] = f(Xt_cv[te], [b[te] for b in b_cv])
                lams.append(sel["lambda"])
            rep_ccc.append(ccc(ycv, pred))
            rep_rmse.append(rmse(ycv, pred))
            rep_std.append(float(pred.std()))
        cv_rep_ccc[exp] = np.array(rep_ccc)
        cvm = {"cv_ccc": float(np.mean(rep_ccc)), "cv_ccc_sd": float(np.std(rep_ccc)),
               "cv_rmse": float(np.mean(rep_rmse)), "cv_pred_std": float(np.mean(rep_std)),
               "cv_n": len(ycv), "cv_reps": reps, "cv_folds": folds,
               "cv_lambda_mean": float(np.mean(lams)), "cv_lambda_nonzero": float(np.mean(np.array(lams) > 0))}
        if "E0_text" in cv_rep_ccc and exp != "E0_text":
            d = cv_rep_ccc[exp] - cv_rep_ccc["E0_text"]
            cvm.update({"cv_delta_vs_E0": float(d.mean()), "cv_delta_vs_E0_sd": float(d.std())})
        save_json(os.path.join(out_root, exp, "metrics_cv.json"), cvm)

        # ---- train -> dev (and test)
        Xt_tr, b_tr = get(ids["train"])
        ytr = np.array([y_all[p] for p in ids["train"]])
        f, sel, coef = fit_model(fcfg, specs, Xt_tr, b_tr, ytr, seed=0)
        out = os.path.join(out_root, exp, "seed0")
        res_dev = {}
        for split in ["dev"] + (["test"] if a.with_test else []):
            Xs, bs = get(ids[split])
            p = f(Xs, bs)
            yy = np.array([y_all[q] for q in ids[split]])
            save_df(os.path.join(out, f"preds_{split}.csv"), pd.DataFrame({"id": ids[split], "y": yy, "pred": p}))
            if split == "dev":
                res_dev = report(yy, p)
        save_json(os.path.join(out, "selection.json"), sel)
        if coef is not None:
            save_df(os.path.join(out_root, exp, "residual_coefficients.csv"), coef)
        summary.append({"experiment": exp, "cv_ccc": cvm["cv_ccc"], "cv_sd": cvm["cv_ccc_sd"],
                        "cv_delta": cvm.get("cv_delta_vs_E0", np.nan), "cv_delta_sd": cvm.get("cv_delta_vs_E0_sd", np.nan),
                        "lambda_nonzero_folds": cvm["cv_lambda_nonzero"], "dev_ccc": res_dev.get("ccc", np.nan),
                        "dev_rmse": res_dev.get("rmse", np.nan), "train_lambda": sel["lambda"], "train_alpha": sel["alpha"]})
        log(f"{exp:32s} CV CCC {cvm['cv_ccc']:.3f} ± {cvm['cv_ccc_sd']:.3f}  "
            f"Δ vs E0 {cvm.get('cv_delta_vs_E0', 0):+.3f} ± {cvm.get('cv_delta_vs_E0_sd', 0):.3f}  "
            f"dev CCC {res_dev.get('ccc', np.nan):.3f}  (λ>0 in {100 * cvm['cv_lambda_nonzero']:.0f}% of folds; "
            f"train λ={sel['lambda']})")

    s = pd.DataFrame(summary)
    if len(s):
        base = s.loc[s.experiment == "E0_text", "dev_ccc"]
        s["dev_delta"] = s.dev_ccc - base.iloc[0] if len(base) else np.nan
    save_df(os.path.join(out_root, "summary_cv_dev.csv"), s)
    print("\n== residual fusion (selection on CV / dev only)")
    print(s.round(3).to_string(index=False))
    print(f"\nwrote {out_root}")
    if not a.with_test:
        print("design frozen? then run once with --with-test and: python 04_evaluate.py --split test --runs " + out_root)


if __name__ == "__main__":
    main()
