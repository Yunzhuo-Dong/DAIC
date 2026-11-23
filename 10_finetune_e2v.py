"""LoRA fine-tuning of emotion2vec on participant speech -> PHQ-8 regression (GPU, ~e2v_env).

Two stages (the dev set is used to watch over-fitting, never to pick a reported score):
    # 1) develop: train on TRAIN, evaluate DEV after every epoch -> learning curve (history.csv)
    sbatch --export=ALL,CMD="python 10_finetune_e2v.py --config configs/finetune_e2v.yaml --stage dev" slurm/e2v_job.sh
    # 2) final: train on TRAIN+DEV for a fixed number of epochs (chosen from the curve), predict TEST
    sbatch --export=ALL,CMD="python 10_finetune_e2v.py --config configs/finetune_e2v.yaml --stage final --set train.epochs=30" slurm/e2v_job.sh

Data: per session, the participant's speech (diarization turns) is concatenated into one stream.
Training: every epoch draws `clips_per_session` random windows of `clip_seconds` per session (talkative
participants do not dominate), label = the session's PHQ-8 (standardised with train statistics).
Model: frozen emotion2vec + LoRA (rank r) on attention qkv/proj of the transformer blocks + linear head
on the time-mean of the last layer. Session prediction = mean over non-overlapping windows.
Outputs in $EDAIC_RUNS/<name>/<stage>/seed<k>/: preds_dev.csv, preds_test.csv, history.csv, lora_head.pt
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from edaic.config import load_config, save_config
from edaic.data import load_index, session_segments
from edaic.features.wav2vec2 import _resample
from edaic.metrics import ccc, report, rmse
from edaic.utils import log, save_df, save_json, seed_everything

SR = 16000


# ---------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B = 0 -> starts as the frozen model

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


def add_lora(model, targets, r, alpha, dropout, block_prefix="blocks."):
    """Wrap nn.Linear modules named <block_prefix>i.<...>.<target> with LoRA. Returns the number wrapped."""
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and full.startswith(block_prefix) and cname in targets:
                setattr(mod, cname, LoRALinear(child, r, alpha, dropout))
                n += 1
    return n


class Regressor(nn.Module):
    def __init__(self, backbone, dim, normalize, dropout=0.1):
        super().__init__()
        self.backbone, self.normalize = backbone, normalize
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, wav):                      # wav [B, T] at 16 kHz, all the same length
        if self.normalize:
            wav = F.layer_norm(wav, wav.shape[-1:])
        x = self.backbone.extract_features(wav, padding_mask=None)["x"]   # [B, frames, dim]
        return self.head(x.mean(1)).squeeze(-1)


def load_backbone(cfg, device):
    from funasr import AutoModel
    am = AutoModel(model=cfg["model"], device=device, disable_update=True)
    m = am.model
    mcfg = getattr(m, "cfg", {})
    normalize = bool(mcfg.get("normalize", False) if hasattr(mcfg, "get") else getattr(mcfg, "normalize", False))
    dim = mcfg.get("embed_dim", 1024) if hasattr(mcfg, "get") else 1024
    return m, int(dim), normalize


# ---------------------------------------------------------------- data
def participant_streams(idx, cfg, ids):
    """Concatenated participant speech per session (float16 numpy, 16 kHz)."""
    out = {}
    for r in idx[idx.id.isin(ids)].itertuples():
        _, segs, _ = session_segments(r.id, r.transcript_path, cfg, r.audio_path)
        a, sr = sf.read(r.audio_path, dtype="float32", always_2d=True)
        a = _resample(a.mean(1), sr)
        parts = [a[int(s * SR):int(e * SR)] for s, e in segs]
        x = np.concatenate(parts) if parts else a
        out[int(r.id)] = x.astype(np.float16)
    return out


def random_windows(stream, n, length, rng):
    if len(stream) <= length:
        return [np.resize(stream, length)] * n
    starts = rng.integers(0, len(stream) - length, n)
    return [stream[s:s + length] for s in starts]


def eval_windows(stream, length, max_windows):
    if len(stream) <= length:
        return [np.resize(stream, length)]
    starts = np.arange(0, len(stream) - length + 1, length)
    if len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows).round().astype(int)]
    return [stream[s:s + length] for s in starts]


@torch.no_grad()
def predict_sessions(model, streams, ids, length, max_windows, bs, device, mu, sd):
    model.eval()
    preds = []
    for pid in ids:
        w = np.stack(eval_windows(streams[pid], length, max_windows)).astype(np.float32)
        out = []
        for i in range(0, len(w), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out.append(model(torch.from_numpy(w[i:i + bs]).to(device)).float().cpu().numpy())
        preds.append(float(np.concatenate(out).mean()) * sd + mu)
    return np.clip(np.array(preds), 0, 24)


def ccc_loss(y, p):
    my, mp = y.mean(), p.mean()
    cov = ((y - my) * (p - mp)).mean()
    return 1 - 2 * cov / (y.var(unbiased=False) + p.var(unbiased=False) + (my - mp) ** 2 + 1e-8)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune_e2v.yaml")
    ap.add_argument("--stage", choices=["dev", "final"], required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N sessions per split")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    t, f = cfg["train"], cfg["finetune"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not t.get("allow_cpu", False):
        raise SystemExit("needs a GPU: submit via slurm/e2v_job.sh")

    idx = load_index(cfg["paths"]["index"])
    ids = {s: idx.loc[idx.split == s, "id"].astype(int).tolist() for s in ("train", "dev", "test")}
    if a.limit:
        ids = {k: v[:a.limit] for k, v in ids.items()}
    fit_ids = ids["train"] + (ids["dev"] if a.stage == "final" else [])
    y = dict(zip(idx.id.astype(int), idx.phq_score.astype(float)))
    out_root = os.path.join(cfg["paths"]["runs"], cfg["name"], a.stage)
    os.makedirs(out_root, exist_ok=True)
    save_config(cfg, os.path.join(out_root, "config.yaml"))

    t0 = time.time()
    streams = participant_streams(idx, cfg, ids["train"] + ids["dev"] + ids["test"])
    log(f"loaded participant speech of {len(streams)} sessions in {time.time() - t0:.0f} s "
        f"(median {np.median([len(v) for v in streams.values()]) / SR / 60:.1f} min)")
    L = int(float(t["clip_seconds"]) * SR)
    ytr = np.array([y[i] for i in fit_ids])
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-6)

    for seed in t.get("seeds", [0]):
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        backbone, dim, normalize = load_backbone(f, device)
        for p in backbone.parameters():
            p.requires_grad = False
        n_lora = add_lora(backbone, set(f["lora_targets"]), int(f["lora_r"]), float(f["lora_alpha"]),
                          float(f["lora_dropout"]), f.get("block_prefix", "blocks."))
        if n_lora == 0:
            raise SystemExit("no LoRA targets found - check finetune.lora_targets / block_prefix")
        model = Regressor(backbone, dim, normalize, float(f.get("head_dropout", 0.1))).to(device)
        lora_p = [p for n, p in model.named_parameters() if n.endswith(".A") or n.endswith(".B")]
        head_p = list(model.head.parameters())
        n_train = sum(p.numel() for p in lora_p + head_p)
        log(f"seed {seed}: LoRA on {n_lora} linear layers, trainable params {n_train / 1e6:.2f} M, normalize={normalize}")

        opt = torch.optim.AdamW([{"params": lora_p, "lr": float(t["lr_lora"])},
                                 {"params": head_p, "lr": float(t["lr_head"])}], weight_decay=float(t["weight_decay"]))
        bs, epochs, k = int(t["batch_size"]), int(t["epochs"]), int(t["clips_per_session"])
        steps_per_epoch = math.ceil(len(fit_ids) * k / bs)
        total, warm = steps_per_epoch * epochs, int(0.05 * steps_per_epoch * epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) *
                                                  0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        out = os.path.join(out_root, f"seed{seed}")
        hist = []
        for ep in range(1, epochs + 1):
            model.train()
            items = [(pid, w) for pid in fit_ids for w in random_windows(streams[pid], k, L, rng)]
            order = rng.permutation(len(items))
            losses, te = [], time.time()
            for i in range(0, len(order), bs):
                b = [items[j] for j in order[i:i + bs]]
                wav = torch.from_numpy(np.stack([w for _, w in b]).astype(np.float32)).to(device)
                tgt = torch.tensor([(y[pid] - mu) / sd for pid, _ in b], dtype=torch.float32, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    p = model(wav).float()
                loss = F.mse_loss(p, tgt) + (float(t.get("ccc_weight", 0.0)) * ccc_loss(tgt, p) if len(b) > 1 else 0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_p + head_p, 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
            row = {"epoch": ep, "train_loss": float(np.mean(losses)), "epoch_s": time.time() - te}
            if a.stage == "dev" and (ep % int(t.get("eval_every", 1)) == 0 or ep == epochs):
                pdv = predict_sessions(model, streams, ids["dev"], L, int(t["max_eval_windows"]), bs, device, mu, sd)
                ydv = np.array([y[i] for i in ids["dev"]])
                row.update({"dev_ccc": ccc(ydv, pdv), "dev_rmse": rmse(ydv, pdv), "dev_pred_std": float(pdv.std())})
            hist.append(row)
            log(" ".join(f"{k_}={v:.4f}" if isinstance(v, float) else f"{k_}={v}" for k_, v in row.items()))
            save_df(os.path.join(out, "history.csv"), pd.DataFrame(hist))

        # predictions of the LAST epoch (no selection on dev)
        for split in ("dev", "test"):
            p = predict_sessions(model, streams, ids[split], L, int(t["max_eval_windows"]), bs, device, mu, sd)
            save_df(os.path.join(out, f"preds_{split}.csv"),
                    pd.DataFrame({"id": ids[split], "y": [y[i] for i in ids[split]], "pred": p}))
            if split == "dev":
                m = report(np.array([y[i] for i in ids["dev"]]), p)
                m.update({"stage": a.stage, "seed": seed, "epochs": epochs,
                          "note": "dev was part of training" if a.stage == "final" else "train-only model"})
                save_json(os.path.join(out, "metrics_dev.json"), m)
        state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(state, os.path.join(out, "lora_head.pt"))
        if a.stage == "dev":
            h = pd.DataFrame(hist).dropna(subset=["dev_ccc"])
            log(f"seed {seed} done: last-epoch dev CCC {h.dev_ccc.iloc[-1]:.3f}; curve peak {h.dev_ccc.max():.3f} "
                f"at epoch {int(h.epoch[h.dev_ccc.idxmax()])} (for choosing the epoch count of the final stage only)")
        else:
            log(f"seed {seed} done ({epochs} epochs on train+dev); test predictions saved")
        del model, backbone, opt
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
