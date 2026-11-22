"""Small helpers: seeding, atomic writes (HORSE can throw transient I/O errors), logging."""
import json
import os
import random
import tempfile
import time

import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _try_write(path, write_fn):
    tmp = None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
        os.close(fd)
        write_fn(tmp)
        os.replace(tmp, path)
    except OSError:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _atomic(path, write_fn, retries=5):
    """Write via a temp file + rename. HORSE sometimes raises transient I/O errors (Errno 5):
    retry with back-off, then fall back to $HOME/edaic_fallback/<same path> instead of crashing."""
    for attempt in range(retries):
        try:
            _try_write(path, write_fn)
            return path
        except OSError as e:
            err = e
            time.sleep(2 * (attempt + 1))
    fallback = os.path.join(os.path.expanduser("~"), "edaic_fallback", os.path.abspath(path).lstrip("/"))
    _try_write(fallback, write_fn)
    log(f"WARNING: could not write {path} ({err}); saved to {fallback} instead")
    return fallback


def save_npz(path, **arrays):
    def w(tmp):
        with open(tmp, "wb") as f:  # file handle: numpy would otherwise append .npz
            np.savez(f, **arrays)
    _atomic(path, w)


def save_json(path, obj):
    def w(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=float)
    _atomic(path, w)


def save_df(path, df):
    _atomic(path, lambda tmp: df.to_csv(tmp, index=False))


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)
