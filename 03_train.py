"""Train one experiment config on train, select on dev, write predictions for dev and test.

    python 03_train.py --config configs/exp_egemaps_ridge.yaml
    python 03_train.py --config configs/exp_w2v2_large_layers_ridge.yaml                     # layer sweep
    python 03_train.py --config configs/exp_ds_vgg16_ridge.yaml --set model.type=ridge_clf name=ds_vgg16_clf

Heads: ridge / svr (PHQ-8 regression), ridge_items (8 per-item ridges, summed), ridge_clf (PHQ-8 >= 10,
class-balanced), attnpool (sequences).
Output: $EDAIC_RUNS/<name>/<variant>/
    seed<k>/preds_{dev,test}.csv, metrics_dev.json   (train on train, evaluate on dev)
    metrics_cv.json                                   (sklearn heads: 5x5-fold CV on train+dev, 219 people;
                                                       the main model-selection number - dev alone is 56 people)
Test metrics are NOT printed here; use 04_evaluate.py --split test at the very end.
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from edaic.assemble import n_layers, segment_lists, session_matrix
from edaic.config import load_config, save_config
from edaic.data import load_index
from edaic.metrics import ccc, classif_report, report, rmse
from edaic.models import CLASSIFIERS, SKLEARN_MODELS, build_sklearn, train_attnpool
from edaic.utils import log, save_df, save_json, seed_everything


def expand_variants(cfg, ids):
    """Yield (variant_name, feature_specs). `layer: all` on one spec -> one variant per layer."""
    specs = cfg["features"]
    sweep = [i for i, s in enumerate(specs) if s.get("layer") == "all"]
    if not sweep:
        yield cfg.get("variant", "+".join(s["name"] + (f"_L{s['layer']}" if "layer" in s else "")
                                          + (f"_{s['key']}" if s.get("key") else "") for s in specs)), specs
        return
    i = sweep[0]
    for L in range(n_layers(cfg["paths"]["features"], specs[i]["name"], ids)):
        s2 = copy.deepcopy(specs)
        s2[i]["layer"] = L
        yield f"layer{L:02d}", s2


def fit_predict(mcfg, xtr, ytr, xs, y_range, is_clf):
    model = build_sklearn(mcfg).fit(xtr, ytr)
    out = []
    for x in xs:
        if not len(x):
            out.append((np.array([]), np.array([])))
        elif is_clf:
            out.append((model.decision_function(x), model.predict(x)))
        else:
            out.append((np.clip(model.predict(x), *y_range), None))
    return out


def cross_validate(mcfg, X, y, y_range, is_clf, reps=5, folds=5):
    """Repeated k-fold on train+dev. Returns mean/sd of the main metrics."""
    res = []
    for r in range(reps):
        split = (StratifiedKFold(folds, shuffle=True, random_state=r).split(X, y) if is_clf
                 else KFold(folds, shuffle=True, random_state=r).split(X))
        score, label = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in split:
            (s, l), = fit_predict(mcfg, X[tr], y[tr], [X[te]], y_range, is_clf)
            score[te] = s
            if is_clf:
                label[te] = l
        if is_clf:
            m = classif_report(y, score, label)
            res.append({k: m[k] for k in ("auroc", "f1_macro", "f1_pos", "bal_acc")})
        else:
            res.append({"ccc": ccc(y, score), "rmse": rmse(y, score), "pred_std": float(score.std())})
    df = pd.DataFrame(res)
    out = {f"cv_{k}": float(df[k].mean()) for k in df}
    out.update({f"cv_{k}_sd": float(df[k].std(ddof=0)) for k in df})
    out.update({"cv_n": int(len(y)), "cv_reps": reps, "cv_folds": folds})
    return out


def inner_split(n, y, frac, seed):
    """Indices (fit, early-stop) from the TRAINING part only, stratified by PHQ quantile bins."""
    rng = np.random.default_rng(seed)
    bins = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
    stop = []
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        k = max(1, int(round(frac * len(idx))))
        stop.extend(rng.choice(idx, k, replace=False))
    stop = np.array(sorted(stop))
    fit = np.setdiff1d(np.arange(n), stop)
    return fit, stop


def train_seq(xs, y, mcfg, tcfg, seed, y_range):
    """Train a sequence model with early stopping on an inner split of the given training data."""
    fit, stop = inner_split(len(xs), y, float(tcfg.get("early_stop_frac", 0.2)), seed)
    device = "cuda" if tcfg.get("device") == "cuda" else "cpu"
    return train_attnpool([xs[i] for i in fit], y[fit], [xs[i] for i in stop], y[stop],
                          mcfg, tcfg, seed, device, y_range)


def cross_validate_seq(xs, y, mcfg, tcfg, y_range, seeds, folds=5):
    """5-fold CV on train+dev for sequence models; seeds averaged inside each fold's prediction."""
    res = []
    for r, seed in enumerate(seeds):
        pred = np.zeros(len(y))
        for tr, te in KFold(folds, shuffle=True, random_state=r).split(np.zeros(len(y))):
            predict, _ = train_seq([xs[i] for i in tr], y[tr], mcfg, tcfg, seed, y_range)
            pred[te] = predict([xs[i] for i in te])
        res.append({"ccc": ccc(y, pred), "rmse": rmse(y, pred), "pred_std": float(pred.std())})
    df = pd.DataFrame(res)
    out = {f"cv_{k}": float(df[k].mean()) for k in df}
    out.update({f"cv_{k}_sd": float(df[k].std(ddof=0)) for k in df})
    out.update({"cv_n": int(len(y)), "cv_reps": len(seeds), "cv_folds": folds})
    return out


ITEMS = [f"item_{i}" for i in range(1, 9)]


def _items_fit_predict(mcfg, xtr, Ytr, xs):
    model = build_sklearn(mcfg).fit(xtr, Ytr)
    out = []
    for x in xs:
        if not len(x):
            out.append((np.zeros((0, 8)), np.array([])))
            continue
        items = np.clip(model.predict(x), 0, 3)
        out.append((items, items.sum(1)))
    return out


def run_items_variant(cfg, idx, variant, specs, out_root):
    """PHQ-8 per-item heads (ridge_items): 8 ridges, items clipped to 0-3 and summed to the total.
    Item labels exist for train + dev only; the test set is predicted with the total as reference."""
    mcfg, ecfg = cfg["model"], cfg.get("eval", {})
    thr = ecfg.get("threshold", 10)
    feat_root = cfg["paths"]["features"]
    has_items = idx[ITEMS].notna().all(1)
    ids = {s: idx.loc[(idx.split == s) & (has_items if s != "test" else True), "id"].tolist()
           for s in ("train", "dev", "test")}
    data = {s: session_matrix(ids[s], feat_root, specs) for s in ids}
    lab = idx.set_index("id")
    (xtr, idtr), (xdv, iddv), (xte, idte) = data["train"], data["dev"], data["test"]
    Ytr, Ydv = lab.loc[idtr, ITEMS].to_numpy(float), lab.loc[iddv, ITEMS].to_numpy(float)
    ttr, tdv = lab.loc[idtr, "phq_score"].to_numpy(float), lab.loc[iddv, "phq_score"].to_numpy(float)

    msg = ""
    if ecfg.get("cv", True):
        X, Y, T = np.concatenate([xtr, xdv]), np.concatenate([Ytr, Ydv]), np.concatenate([ttr, tdv])
        res, item_ccc = [], []
        for r in range(int(ecfg.get("cv_reps", 5))):
            P = np.zeros_like(Y)
            for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
                (P[te], _), = _items_fit_predict(mcfg, X[tr], Y[tr], [X[te]])
            tot = P.sum(1)
            res.append({"ccc": ccc(T, tot), "rmse": rmse(T, tot), "pred_std": float(tot.std())})
            item_ccc.append([ccc(Y[:, j], P[:, j]) for j in range(8)])
        df = pd.DataFrame(res)
        cvm = {f"cv_{k}": float(df[k].mean()) for k in df}
        cvm.update({f"cv_{k}_sd": float(df[k].std(ddof=0)) for k in df})
        names = idx.attrs.get("item_names") or ITEMS
        cvm["cv_item_ccc"] = dict(zip(ITEMS, np.mean(item_ccc, 0).round(4).tolist()))
        cvm.update({"cv_n": int(len(T)), "variant": variant, "target": "phq_items", "head": "ridge_items"})
        save_json(os.path.join(out_root, variant, "metrics_cv.json"), cvm)
        msg = f"CV CCC {cvm['cv_ccc']:.3f} ± {cvm['cv_ccc_sd']:.3f}"
        print("   per-item CV CCC: " + "  ".join(f"{k}={v:+.3f}" for k, v in cvm["cv_item_ccc"].items()))

    out = os.path.join(out_root, variant, "seed0")
    os.makedirs(out, exist_ok=True)
    (idv, pdv), (ite, pte) = _items_fit_predict(mcfg, xtr, Ytr, [xdv, xte])
    for split, ids_, tot, items in (("dev", iddv, pdv, idv), ("test", idte, pte, ite)):
        df = pd.DataFrame({"id": ids_, "y": lab.loc[ids_, "phq_score"].to_numpy(float) if len(ids_) else [],
                           "pred": tot})
        for j, c in enumerate(ITEMS):
            df[f"pred_{c}"] = items[:, j] if len(ids_) else []
        save_df(os.path.join(out, f"preds_{split}.csv"), df)
    m = report(tdv, pdv, thr)
    m.update({"variant": variant, "seed": 0, "n_train": len(idtr), "target": "phq_items", "head": "ridge_items"})
    save_json(os.path.join(out, "metrics_dev.json"), m)
    log(f"{variant:28s} {msg:28s} dev CCC {m['ccc']:.3f} (items summed)")


def run_variant(cfg, idx, variant, specs, out_root):
    mcfg, tcfg, ecfg = cfg["model"], cfg.get("train", {}), cfg.get("eval", {})
    if mcfg["type"] == "ridge_items":
        return run_items_variant(cfg, idx, variant, specs, out_root)
    is_clf = mcfg["type"] in CLASSIFIERS
    target = cfg.get("clf_target", "phq_binary") if is_clf else cfg.get("target", "phq_score")
    y_range = tuple(cfg.get("y_range", [0, 24]))
    thr = ecfg.get("threshold", 10)
    split_ids = {s: idx.loc[(idx.split == s) & idx[target].notna(), "id"].tolist() for s in ("train", "dev", "test")}
    ymap = dict(zip(idx["id"], idx[target]))
    feat_root = cfg["paths"]["features"]
    is_seq = mcfg["type"] not in SKLEARN_MODELS
    seeds = [0] if not is_seq else list(tcfg.get("seeds", [0]))

    if is_seq:
        if len(specs) != 1:
            raise ValueError("sequence models take exactly one feature spec (fusion is RP2)")
        data = {s: segment_lists(split_ids[s], feat_root, specs[0]) for s in split_ids}
    else:
        data = {s: session_matrix(split_ids[s], feat_root, specs) for s in split_ids}
    (xtr, idtr), (xdv, iddv), (xte, idte) = data["train"], data["dev"], data["test"]
    ytr = np.array([ymap[i] for i in idtr], float)
    ydv = np.array([ymap[i] for i in iddv], float)

    msg = ""
    if is_seq and ecfg.get("cv", True):
        xs, y = xtr + xdv, np.concatenate([ytr, ydv])
        cvm = cross_validate_seq(xs, y, mcfg, tcfg, y_range, list(tcfg.get("cv_seeds", [0, 1, 2])))
        cvm.update({"variant": variant, "target": target, "head": mcfg["type"]})
        save_json(os.path.join(out_root, variant, "metrics_cv.json"), cvm)
        msg = f"CV CCC {cvm['cv_ccc']:.3f} ± {cvm['cv_ccc_sd']:.3f}"
    if not is_seq and ecfg.get("cv", True):
        X = np.concatenate([xtr, xdv])
        y = np.concatenate([ytr, ydv])
        cvm = cross_validate(mcfg, X, y, y_range, is_clf, int(ecfg.get("cv_reps", 5)))
        cvm.update({"variant": variant, "target": target, "head": mcfg["type"]})
        save_json(os.path.join(out_root, variant, "metrics_cv.json"), cvm)
        msg = (f"CV AUROC {cvm['cv_auroc']:.3f} ± {cvm['cv_auroc_sd']:.3f}" if is_clf
               else f"CV CCC {cvm['cv_ccc']:.3f} ± {cvm['cv_ccc_sd']:.3f}")

    dev_main = []
    for seed in seeds:
        seed_everything(seed)
        out = os.path.join(out_root, variant, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        if is_seq:
            # early stopping on an inner split of TRAIN; dev is used only for evaluation
            predict, hist = train_seq(xtr, ytr, mcfg, tcfg, seed, y_range)
            save_df(os.path.join(out, "history.csv"), pd.DataFrame(hist))
            preds = {"dev": (iddv, predict(xdv), None),
                     "test": (idte, predict(xte) if len(idte) else np.array([]), None)}
        else:
            (sdv, ldv), (ste, lte) = fit_predict(mcfg, xtr, ytr, [xdv, xte], y_range, is_clf)
            preds = {"dev": (iddv, sdv, ldv), "test": (idte, ste, lte)}
        for split, (ids, p, lab) in preds.items():
            df = pd.DataFrame({"id": ids, "y": [ymap[i] for i in ids], "pred": p})
            if lab is not None:
                df["label"] = lab
            save_df(os.path.join(out, f"preds_{split}.csv"), df)
        if is_clf:
            m = classif_report(ydv, preds["dev"][1], preds["dev"][2])
            dev_main.append(m["auroc"])
        else:
            m = report(ydv, preds["dev"][1], thr if target == "phq_score" else None)
            dev_main.append(m["ccc"])
        m.update({"variant": variant, "seed": seed, "n_train": len(idtr), "target": target, "head": mcfg["type"]})
        save_json(os.path.join(out, "metrics_dev.json"), m)
    name = "AUROC" if is_clf else "CCC"
    log(f"{variant:28s} {msg:28s} dev {name} {np.mean(dev_main):.3f} (seeds={len(seeds)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.type=ridge_clf name=x_clf")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    idx = load_index(cfg["paths"]["index"])
    out_root = os.path.join(cfg["paths"]["runs"], cfg["name"])
    os.makedirs(out_root, exist_ok=True)
    save_config(cfg, os.path.join(out_root, "config.yaml"))
    log(f"experiment {cfg['name']} ({cfg['model']['type']}) -> {out_root}")
    for variant, specs in expand_variants(cfg, idx["id"].tolist()):
        run_variant(cfg, idx, variant, specs, out_root)


if __name__ == "__main__":
    main()
