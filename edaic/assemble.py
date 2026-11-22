"""Turn cached feature files into model inputs.

A feature spec (in an experiment YAML) looks like:
    - name: egemaps                    # folder under $EDAIC_FEAT
    - name: w2v2_large
      layer: mean                      # one vector: mean | an int | "a-b" (mean of layers a..b) | all (sweep)
"""
import os

import numpy as np

from .utils import log


def feature_path(feat_root, name, pid):
    return os.path.join(feat_root, name, f"{int(pid)}.npz")


def n_layers(feat_root, name, ids):
    for pid in ids:
        p = feature_path(feat_root, name, pid)
        if os.path.exists(p):
            return int(np.load(p)["session_layers"].shape[0])
    raise FileNotFoundError(f"no files for {name} in {feat_root}")


def _pick_layers(x, layer, axis):
    """layer: int (one layer) | "mean" (average of all layers) | "a-b" (average of layers a..b)."""
    if isinstance(layer, str) and layer == "mean":
        return x.mean(axis=axis)
    if isinstance(layer, str) and "-" in layer:
        a, b = (int(v) for v in layer.split("-"))
        return np.take(x, range(a, b + 1), axis=axis).mean(axis=axis)
    return np.take(x, int(layer), axis=axis)


def _session_vec(z, spec):
    if spec.get("key"):                      # e.g. key: session_emotions (emotion2vec probabilities)
        return z[spec["key"]]
    if "session_layers" in z.files:
        layer = spec.get("layer", "mean")
        return _pick_layers(z["session_layers"], layer, axis=0)
    return z["session"]


def _segment_mat(z, spec):
    seg = z["segments"].astype(np.float32)
    if seg.ndim == 3:
        seg = _pick_layers(seg, spec.get("layer", "mean"), axis=1)
    return seg


def session_matrix(ids, feat_root, specs):
    """X [N, D] (early fusion = concatenation of all specs) and a list of kept ids."""
    rows, kept = [], []
    for pid in ids:
        parts = []
        for spec in specs:
            p = feature_path(feat_root, spec["name"], pid)
            if not os.path.exists(p):
                parts = None
                break
            parts.append(_session_vec(np.load(p), spec).astype(np.float32))
        if parts is None:
            continue
        rows.append(np.concatenate(parts))
        kept.append(pid)
    missing = len(ids) - len(kept)
    if missing:
        log(f"WARNING: {missing} sessions without features {[s['name'] for s in specs]} were skipped")
    return np.stack(rows), kept


def segment_lists(ids, feat_root, spec):
    """List of [n_seg, D] arrays for one spec (sequence models)."""
    mats, kept = [], []
    for pid in ids:
        p = feature_path(feat_root, spec["name"], pid)
        if not os.path.exists(p):
            continue
        m = _segment_mat(np.load(p), spec)
        m = m[~np.isnan(m).all(1)]
        if len(m):
            mats.append(np.nan_to_num(m))
            kept.append(pid)
    if len(kept) < len(ids):
        log(f"WARNING: {len(ids) - len(kept)} sessions without {spec['name']} segments were skipped")
    return mats, kept
