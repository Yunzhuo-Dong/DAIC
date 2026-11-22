"""YAML configs with ${ENV_VAR} expansion and command-line overrides.

Usage in scripts:
    cfg = load_config("configs/exp_egemaps_ridge.yaml", overrides=["train.seeds=[0,1]"])
"""
import copy
import os

import yaml

# Default locations; slurm/env.sh and slurm/barnard_env.sh export these.
_WS = os.environ.get("WS", "/data/horse/ws/yudo868h-Yunzhuo_space")
DEFAULT_ENV = {
    "WS": _WS,
    "EDAIC_RAW": f"{_WS}/edaic_raw",
    "EDAIC_FEAT": f"{_WS}/edaic_features",
    "EDAIC_RUNS": f"{_WS}/edaic_runs",
    "HF_HOME": f"{_WS}/hf_cache",
    "EDAIC_CODE": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
}


def env(name):
    return os.environ.get(name, DEFAULT_ENV.get(name, ""))


def _expand(obj):
    if isinstance(obj, str):
        for k in DEFAULT_ENV:  # make defaults visible to expandvars
            os.environ.setdefault(k, DEFAULT_ENV[k])
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def _set_dotted(cfg, key, value):
    """features.0.layer=6 -> cfg['features'][0]['layer'] = 6 (numeric parts index lists).
    An index equal to the list length appends a new item: features.1.name=praat adds a 2nd feature."""
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if isinstance(node, list):
            i = int(p)
            if i == len(node):
                node.append({})
            node = node[i]
        else:
            node = node.setdefault(p, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()):
    """Load YAML. A top-level `include: other.yaml` (relative to this file) is merged first."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    inc = cfg.pop("include", None)
    if inc:
        base = load_config(os.path.join(os.path.dirname(os.path.abspath(path)), inc))
        cfg = _merge(base, cfg)
    cfg = copy.deepcopy(cfg)
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    return _expand(cfg)


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
