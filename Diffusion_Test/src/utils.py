import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def current_time_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rotate_checkpoints(checkpoint_dir: str, keep: int) -> None:
    if keep <= 0:
        return
    files = []
    for name in os.listdir(checkpoint_dir):
        if name.endswith(".pt") and name.startswith("epoch_"):
            files.append(os.path.join(checkpoint_dir, name))
    files.sort(key=lambda p: os.path.getmtime(p))
    while len(files) > keep:
        to_remove = files.pop(0)
        os.remove(to_remove)
