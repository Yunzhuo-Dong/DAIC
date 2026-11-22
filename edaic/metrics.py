"""CCC / RMSE / MAE for PHQ-8 regression, plus screening metrics at PHQ-8 >= threshold."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def ccc(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = ((y - my) * (p - mp)).mean()
    den = vy + vp + (my - mp) ** 2
    return float(2 * cov / den) if den > 0 else 0.0


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def pearson(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else 0.0


def report(y, p, threshold=10.0):
    """All metrics for one set of predictions. y, p: PHQ-8 totals."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"n": int(len(y)), "ccc": ccc(y, p), "rmse": rmse(y, p), "mae": mae(y, p),
           "pearson": pearson(y, p), "pred_std": float(p.std())}
    if threshold is not None:
        yb, pb = (y >= threshold).astype(int), (p >= threshold).astype(int)
        out["f1_macro"] = float(f1_score(yb, pb, average="macro", zero_division=0))
        out["bal_acc"] = float(balanced_accuracy_score(yb, pb))
        out["auroc"] = float(roc_auc_score(yb, p)) if 0 < yb.sum() < len(yb) else float("nan")
    return out


def bootstrap_ci(y, p, metric=ccc, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI over participants."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(metric(y[i], p[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_permutation(y, p_a, p_b, metric=ccc, n_perm=5000, seed=0):
    """Two-sided p-value that model A and B differ in `metric` (swap predictions per participant)."""
    y, a, b = (np.asarray(v, float) for v in (y, p_a, p_b))
    obs = metric(y, a) - metric(y, b)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        s = rng.random(len(y)) < 0.5
        aa, bb = np.where(s, b, a), np.where(s, a, b)
        cnt += abs(metric(y, aa) - metric(y, bb)) >= abs(obs)
    return float(obs), float((cnt + 1) / (n_perm + 1))


def classif_report(y, score, label):
    """Binary screening metrics. y, label in {0,1}; score = decision value (higher = depressed)."""
    y, score, label = (np.asarray(v, float) for v in (y, score, label))
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "f1_macro": float(f1_score(y, label, average="macro", zero_division=0)),
           "f1_pos": float(f1_score(y, label, zero_division=0)),
           "bal_acc": float(balanced_accuracy_score(y, label))}
    out["auroc"] = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")
    return out
