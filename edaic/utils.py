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


def _atomic(path, write_fn, retries=3):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for attempt in range(retries):
        try:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
            os.close(fd)
            write_fn(tmp)
            os.replace(tmp, path)
            return
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


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
