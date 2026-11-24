"""Plan F1: per-item supervised fine-tuning (LoRA) of Qwen3.8-27B to rate PHQ-8 items from the participant transcript.

Each training sample = (participant transcript, ONE PHQ-8 item) -> the true item score as a single digit 0-3.
163 train sessions x 8 items = 1304 samples; 1 epoch, effective batch 16 (~82 optimizer steps).

Prediction: one forward pass per (session, item); the probabilities of the tokens "0".."3" give an expected
item score E = sum_k k * p(k). The PHQ-8 prediction is the sum of the 8 expected scores (no generation, no parsing).

Stages (each writes $EDAIC_RUNS/llm_ft_<name>/<stage>/seed0/preds_<split>.csv for 04_evaluate.py):
    base   no training: the same per-item scoring with the original model     -> dev + test   (control)
    dev    train on TRAIN only, predict DEV                                    -> decision: dev CCC > 0.624 ?
    final  train on TRAIN + DEV, predict TEST (run only if `dev` passed)       -> test, evaluated once

    sbatch --export=ALL,CMD="python 14_finetune_llm.py --config configs/ft_qwen.yaml --stage base" slurm/llm_job.sh
    sbatch --export=ALL,CMD="python 14_finetune_llm.py --config configs/ft_qwen.yaml --stage dev" slurm/llm_job.sh
    sbatch --export=ALL,CMD="python 14_finetune_llm.py --config configs/ft_qwen.yaml --stage final" slurm/llm_job.sh
Quick pipeline test (4 sessions per split, 2 optimizer steps):  add  --limit 4
"""
import argparse
import math
import os
import random
import re
import time
from importlib import import_module

import numpy as np
import pandas as pd

from edaic.config import load_config, save_config
from edaic.data import load_index
from edaic.metrics import ccc, report
from edaic.utils import log, save_df, save_json

score = import_module("11_llm_score")  # transcript_text(), ITEMS
ITEMS = score.ITEMS
ITEM_COLS = [f"item_{i}" for i in range(1, 9)]
BAR_DEV = 0.624  # dev CCC of zero-shot items + ridge (llm_part_ridge)

SYSTEM = (
    "You are an experienced clinical psychologist. You read the transcript of a semi-structured interview "
    "and estimate the participant's answer to one item of the PHQ-8 depression questionnaire for the last "
    "two weeks: 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day. "
    "Answer with a single digit.")

USER = ("Interview transcript ({source}):\n\"\"\"\n{text}\n\"\"\"\n\n"
        "PHQ-8 item {k}: {item}\n"
        "How often has the participant been bothered by this over the last two weeks? Answer 0, 1, 2 or 3.")


# ---------------------------------------------------------------- data
def build_prompts(tok, cfg, rows):
    """One chat prompt (string) per (session, item); the transcript is computed once per session."""
    c = cfg["llm"]
    kw = dict(c.get("chat_template_kwargs", {}))
    out = []
    for r in rows:
        text, src = score.transcript_text(r, cfg, c["text_source"], int(c.get("max_chars", 40000)))
        for k, item in enumerate(ITEMS, 1):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.format(source=src, text=text, k=k, item=item)}]
            try:
                p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            except TypeError:
                p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            out.append((int(r.id), k, p))
    return out


def encode(tok, prompt, answer, max_len):
    """input_ids / labels for one sample; loss only on the answer digit and the end-of-turn token."""
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    if len(p) + len(a) > max_len:  # cut the middle of the prompt (keeps instructions and the item question)
        keep = max_len - len(a)
        p = p[:keep // 2] + p[-(keep - keep // 2):]
    return p + a, [-100] * len(p) + a


def end_of_turn(tok):
    for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        if tok.convert_tokens_to_ids(t) not in (None, tok.unk_token_id):
            return t
    return tok.eos_token or ""


# ---------------------------------------------------------------- model
def load_model(c, f, train):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(c["model"])
    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if f.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(c["model"], **kw)
    except (ValueError, KeyError):  # vision-language checkpoint
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(c["model"], **kw)
    model.config.use_cache = False
    if not train:
        model.eval()
        return tok, model

    from peft import LoraConfig, get_peft_model
    if f.get("load_in_4bit"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    excl = [e.lower() for e in f.get("exclude_modules", [])]
    leaves = sorted({n.split(".")[-1] for n, m in model.named_modules()
                     if isinstance(m, torch.nn.Linear) and not any(e in n.lower() for e in excl)})
    leaves = [l for l in leaves if not any(e in l.lower() for e in excl)]
    excl_re = "|".join(map(re.escape, excl)) or "^$"
    target = rf"^(?!.*({excl_re})).*\.({'|'.join(map(re.escape, leaves))})$"
    log(f"LoRA on linear layers: {leaves}")
    lcfg = LoraConfig(r=int(f["lora_r"]), lora_alpha=int(f["lora_alpha"]), lora_dropout=float(f["lora_dropout"]),
                      target_modules=target, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"trainable parameters: {n_tr / 1e6:.1f} M of {n_all / 1e9:.2f} B ({100 * n_tr / n_all:.2f} %)")
    return tok, model


def train(model, tok, samples, f, out_dir):
    """samples: list of (input_ids, labels). Plain loop: AdamW, linear warm-up, cosine decay, grad accumulation."""
    import torch
    rng = random.Random(int(f.get("seed", 0)))
    order = list(range(len(samples)))
    rng.shuffle(order)
    epochs, accum = int(f.get("epochs", 1)), int(f.get("grad_accum", 16))
    n_steps = max(1, math.ceil(len(samples) * epochs / accum))
    warm = max(1, int(round(float(f.get("warmup_ratio", 0.06)) * n_steps)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(f["lr"]), weight_decay=float(f.get("weight_decay", 0.0)),
                            betas=(0.9, 0.999))
    lr_at = lambda s: float(f["lr"]) * (min(1.0, (s + 1) / warm) if s < warm
                                         else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm))))
    dev = next(model.parameters()).device
    model.train()
    log(f"training: {len(samples)} samples x {epochs} epoch(s), effective batch {accum}, {n_steps} steps, "
        f"lr {f['lr']} (warm-up {warm} steps, cosine)")
    hist, step, run_loss, t0 = [], 0, [], time.time()
    seq = [i for _ in range(epochs) for i in order]
    for j, i in enumerate(seq):
        ids, lab = samples[i]
        x = torch.tensor([ids], device=dev)
        y = torch.tensor([lab], device=dev)
        loss = model(input_ids=x, labels=y).loss
        (loss / accum).backward()
        run_loss.append(loss.item())
        last = j == len(seq) - 1
        if (j + 1) % accum == 0 or last:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(params, float(f.get("max_grad_norm", 1.0))).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            hist.append({"step": step + 1, "loss": float(np.mean(run_loss)), "lr": lr_at(step), "grad_norm": gn,
                         "seconds": round(time.time() - t0, 1)})
            if (step + 1) % int(f.get("log_every", 1)) == 0 or last:
                h = hist[-1]
                log(f"step {h['step']:3d}/{n_steps}  loss {h['loss']:.4f}  lr {h['lr']:.2e}  "
                    f"grad-norm {gn:.3f}  ({h['seconds']:.0f} s)")
            run_loss, step = [], step + 1
    save_df(os.path.join(out_dir, "train_log.csv"), pd.DataFrame(hist))
    model.eval()
    return hist


def predict(model, tok, prompts, max_len):
    """Expected score (0-3) per (session, item) from the next-token probabilities of '0'..'3'."""
    import torch
    digit_ids = [tok(str(k), add_special_tokens=False)["input_ids"][0] for k in range(4)]
    dev = next(model.parameters()).device
    rows = []
    with torch.inference_mode():
        for n, (pid, k, p) in enumerate(prompts, 1):
            ids, _ = encode(tok, p, "", max_len)
            logits = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
            prob_all = torch.softmax(logits, -1)
            pr = prob_all[digit_ids]
            mass = pr.sum().item()
            pr = (pr / pr.sum()).cpu().numpy()
            rows.append({"id": pid, "item": k, **{f"p{j}": pr[j] for j in range(4)},
                         "expected": float((pr * np.arange(4)).sum()), "argmax": int(pr.argmax()),
                         "digit_mass": mass})
            if n % 80 == 0:
                log(f"  predicted {n}/{len(prompts)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ft_qwen.yaml")
    ap.add_argument("--stage", required=True, choices=["base", "dev", "final"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0, help="pipeline test: N sessions per split")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    c, f = cfg["llm"], cfg["ft"]
    random.seed(int(f.get("seed", 0)))
    np.random.seed(int(f.get("seed", 0)))
    import torch
    torch.manual_seed(int(f.get("seed", 0)))

    exp_dir = os.path.join(cfg["paths"]["runs"], f"llm_ft_{f['name']}")
    out_dir = os.path.join(exp_dir, a.stage)
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    idx = load_index(cfg["paths"]["index"])
    if not all(col in idx for col in ITEM_COLS):
        raise SystemExit("sessions.csv has no item_1..item_8 columns (Detailed_PHQ8_Labels.csv) - re-run 01_index.py")
    if a.limit:
        idx = idx.groupby("split").head(a.limit)
    labelled = idx[ITEM_COLS].notna().all(axis=1)
    train_splits = {"base": [], "dev": ["train"], "final": ["train", "dev"]}[a.stage]
    pred_splits = {"base": ["dev", "test"], "dev": ["dev"], "final": ["test"]}[a.stage]
    tr_rows = list(idx[idx.split.isin(train_splits) & labelled].itertuples())

    tok, model = load_model(c, f, train=bool(tr_rows))
    max_len = int(f.get("max_len", 8192))
    if tr_rows:
        eot = end_of_turn(tok)
        prompts = build_prompts(tok, cfg, tr_rows)
        lab = {int(r.id): [int(getattr(r, col)) for col in ITEM_COLS] for r in tr_rows}
        samples = [encode(tok, p, f"{lab[pid][k - 1]}{eot}", max_len) for pid, k, p in prompts]
        lens = np.array([len(s[0]) for s in samples])
        log(f"{len(tr_rows)} sessions ({'+'.join(train_splits)}) -> {len(samples)} samples; tokens per sample: "
            f"median {np.median(lens):.0f}, max {lens.max()}")
        dist = np.bincount(np.array([v for l in lab.values() for v in l]), minlength=4)
        log(f"label distribution 0/1/2/3: {dist.tolist()}")
        if a.limit:
            f = {**f, "grad_accum": max(1, len(samples) // 2)}
        train(model, tok, samples, f, out_dir)
        model.save_pretrained(os.path.join(out_dir, "adapter"))
        log(f"adapter saved -> {os.path.join(out_dir, 'adapter')}")

    item_names = ["interest", "mood", "sleep", "energy", "appetite", "selfworth", "concentration", "psychomotor"]
    summary = {"stage": a.stage, "trained_on": train_splits, "n_train_sessions": len(tr_rows)}
    for split in pred_splits:
        rows = list(idx[idx.split == split].itertuples())
        log(f"predicting {split}: {len(rows)} sessions x 8 items")
        pi = predict(model, tok, build_prompts(tok, cfg, rows), max_len)
        save_df(os.path.join(out_dir, f"items_{split}.csv"), pi)
        tot = pi.groupby("id").agg(pred=("expected", "sum"), pred_argmax=("argmax", "sum")).reset_index()
        df = idx[["id", "phq_score"]].merge(tot, on="id").rename(columns={"phq_score": "y"})
        save_df(os.path.join(out_dir, "seed0", f"preds_{split}.csv"), df[["id", "y", "pred", "pred_argmax"]])
        m = report(df.y, df.pred)
        m["ccc_argmax"] = float(ccc(df.y, df.pred_argmax))
        m["mean_pred"], m["mean_true"] = float(df.pred.mean()), float(df.y.mean())
        m["digit_mass_min"] = float(pi.digit_mass.min())
        if split != "test":
            wide = pi.pivot(index="id", columns="item", values="expected")
            lab_s = idx.set_index("id").loc[wide.index, ITEM_COLS]
            m["item_ccc"] = {item_names[k - 1]: float(ccc(lab_s[f"item_{k}"].values.astype(float), wide[k].values))
                             for k in range(1, 9) if lab_s[f"item_{k}"].notna().all()}
        summary[split] = m
        log(f"{a.stage} -> {split} (n={m['n']}): CCC {m['ccc']:.3f} (argmax {m['ccc_argmax']:.3f})  "
            f"RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  mean pred {m['mean_pred']:.2f} vs true {m['mean_true']:.2f}")
        if "item_ccc" in m:
            log("   per-item CCC: " + "  ".join(f"{k}={v:+.2f}" for k, v in m["item_ccc"].items()))
        if split == "dev" and a.stage == "dev":
            verdict = "PASSED -> run --stage final" if m["ccc"] > BAR_DEV else "NOT better -> stop fine-tuning"
            log(f"   bar (zero-shot items + ridge, dev): {BAR_DEV:.3f}  => {verdict}")
    save_json(os.path.join(out_dir, "metrics.json"), summary)
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
