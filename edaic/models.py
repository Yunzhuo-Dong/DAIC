"""Session-level regressors.

sklearn (fast baselines, session vectors):
    ridge      imputer -> scaler -> RidgeCV (alpha by leave-one-out on train)           -> PHQ-8 score
    ridge_clf  imputer -> scaler -> RidgeClassifierCV, class-balanced (alpha by LOO)  -> PHQ-8 >= 10
    svr     imputer -> scaler -> SVR (C, epsilon by 5-fold CV on train)
torch (segment sequences):
    attnpool  per-segment MLP -> gated attention pooling -> score
Reserved for RP2: cross-attention / gated fusion, LoRA LLM heads.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .metrics import ccc

SKLEARN_MODELS = ("ridge", "svr", "ridge_clf")
CLASSIFIERS = ("ridge_clf",)


def build_sklearn(mcfg):
    kind = mcfg["type"]
    pre = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
    if kind == "ridge":
        alphas = np.logspace(*mcfg.get("alpha_logspace", [-1, 5, 25]))
        return make_pipeline(*pre, RidgeCV(alphas=alphas, cv=mcfg.get("alpha_cv")))  # None = efficient LOO
    if kind == "ridge_clf":
        alphas = np.logspace(*mcfg.get("alpha_logspace", [-1, 5, 25]))
        return make_pipeline(*pre, RidgeClassifierCV(alphas=alphas, cv=mcfg.get("alpha_cv"), class_weight="balanced"))
    if kind == "svr":
        grid = {"svr__C": mcfg.get("C", [0.1, 1, 10]), "svr__epsilon": mcfg.get("epsilon", [0.5, 1, 2])}
        return GridSearchCV(make_pipeline(*pre, SVR(kernel=mcfg.get("kernel", "rbf"))), grid, cv=5,
                            scoring="neg_root_mean_squared_error")
    raise KeyError(kind)


# ---------------------------------------------------------------- torch
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_attnpool(in_dim, mcfg):
    torch, nn = _torch()

    class AttnPool(nn.Module):
        def __init__(self):
            super().__init__()
            h = int(mcfg.get("hidden", 128))
            p = float(mcfg.get("dropout", 0.3))
            self.norm = nn.LayerNorm(in_dim)
            self.enc = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(p))
            self.att_v = nn.Linear(h, h)
            self.att_u = nn.Linear(h, h)
            self.att_w = nn.Linear(h, 1)
            self.head = nn.Sequential(nn.Dropout(p), nn.Linear(h, 1))

        def forward(self, x, mask):                       # x [B, T, D], mask [B, T] True = real
            h = self.enc(self.norm(x))
            a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
            a = a.masked_fill(~mask, -1e4).softmax(-1)
            return self.head((a.unsqueeze(-1) * h).sum(1)).squeeze(-1), a

    return AttnPool()


def _pad(mats, max_len):
    torch, _ = _torch()
    B, D = len(mats), mats[0].shape[1]
    T = min(max(len(m) for m in mats), max_len)
    x = torch.zeros(B, T, D)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, m in enumerate(mats):
        m = m[:T]
        x[i, :len(m)] = torch.from_numpy(m)
        mask[i, :len(m)] = True
    return x, mask


def _ccc_loss(y, p):
    torch, _ = _torch()
    my, mp = y.mean(), p.mean()
    cov = ((y - my) * (p - mp)).mean()
    return 1 - 2 * cov / (y.var(unbiased=False) + p.var(unbiased=False) + (my - mp) ** 2 + 1e-8)


def train_attnpool(tr_x, tr_y, dv_x, dv_y, mcfg, tcfg, seed, device="cpu", y_range=(0, 24)):
    """Train on train, early-stop on dev CCC. Returns (predict_fn, history list)."""
    torch, _ = _torch()
    rng = np.random.default_rng(seed)
    # feature standardisation from training segments
    allseg = np.concatenate(tr_x)
    mu, sd = allseg.mean(0), allseg.std(0) + 1e-6
    norm = lambda mats: [((m - mu) / sd).astype(np.float32) for m in mats]
    tr_x, dv_x = norm(tr_x), norm(dv_x)
    lo, hi = y_range
    y_mu, y_sd = float(np.mean(tr_y)), float(np.std(tr_y) + 1e-6)

    model = build_attnpool(tr_x[0].shape[1], mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 1e-3)),
                            weight_decay=float(tcfg.get("weight_decay", 1e-2)))
    bs, max_len = int(tcfg.get("batch_size", 16)), int(tcfg.get("max_segments", 400))
    loss_kind = tcfg.get("loss", "mse")
    best, best_state, bad, hist = -np.inf, None, 0, []

    def predict(mats):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(mats), 32):
                x, m = _pad(mats[i:i + 32], max_len)
                out.append(model(x.to(device), m.to(device))[0].cpu().numpy())
        return np.clip(np.concatenate(out) * y_sd + y_mu, lo, hi)

    for ep in range(int(tcfg.get("epochs", 100))):
        model.train()
        order = rng.permutation(len(tr_x))
        losses = []
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            batch = []
            for j in idx:  # segment dropout as light augmentation
                m = tr_x[j]
                keep = rng.random(len(m)) > float(tcfg.get("segment_dropout", 0.1))
                batch.append(m[keep] if keep.sum() > 0 else m)
            x, mask = _pad(batch, max_len)
            y = torch.tensor((np.asarray(tr_y)[idx] - y_mu) / y_sd, dtype=torch.float32, device=device)
            p, _ = model(x.to(device), mask.to(device))
            loss = torch.nn.functional.mse_loss(p, y) if loss_kind == "mse" else _ccc_loss(y, p)
            if loss_kind == "mse+ccc":
                loss = torch.nn.functional.mse_loss(p, y) + _ccc_loss(y, p)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        dv_ccc = ccc(dv_y, predict(dv_x))
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "dev_ccc": dv_ccc})
        if dv_ccc > best:
            best, bad = dv_ccc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(tcfg.get("patience", 20)):
                break
    model.load_state_dict(best_state)
    return (lambda mats: predict(norm(mats))), hist
