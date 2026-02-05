import os
import random
from typing import Iterable, List, Tuple

import numpy as np
import torch
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def list_files(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".nii", ".nii.gz")):
                paths.append(os.path.join(dirpath, name))
    paths.sort()
    return paths


def load_arrays(path: str) -> Iterable[np.ndarray]:
    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        return [arr]
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        data = nib.load(path).get_fdata().astype(np.float32)
        if data.ndim == 2:
            return [data]
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim == 3:
            return [data[:, :, i] for i in range(data.shape[2])]
        return [np.squeeze(data)]
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return [arr]


def load_single_array(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim >= 3:
            return arr[..., arr.shape[-1] // 2]
        return arr
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        data = nib.load(path).get_fdata().astype(np.float32)
        if data.ndim == 2:
            return data
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim == 3:
            return data[:, :, data.shape[2] // 2]
        return np.squeeze(data)
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).unsqueeze(0)


def augment_tensor(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])
    angle = random.uniform(-10.0, 10.0)
    x = TF.rotate(x, angle=angle)
    return x


class InMemoryDataset(Dataset):
    def __init__(self, root: str, image_size: int, augment: bool, per_file: bool = False) -> None:
        self.paths = list_files(root)
        self.image_size = image_size
        self.augment = augment
        self.cache = []
        for path in self.paths:
            if per_file:
                arr = load_single_array(path)
                arr = normalize_array(arr)
                tensor = to_tensor(arr)
                self.cache.append(tensor)
            else:
                for arr in load_arrays(path):
                    arr = normalize_array(arr)
                    tensor = to_tensor(arr)
                    self.cache.append(tensor)

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.cache[idx]
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = TF.resize(x, [self.image_size, self.image_size], antialias=True)
        if self.augment:
            x = augment_tensor(x)
        return x


class UnpairedDataset(Dataset):
    def __init__(self, root_a: str, root_b: str, image_size: int, augment: bool) -> None:
        self.ds_a = InMemoryDataset(root_a, image_size, augment)
        self.ds_b = InMemoryDataset(root_b, image_size, augment)

    def __len__(self) -> int:
        return max(len(self.ds_a), len(self.ds_b))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.ds_a[idx % len(self.ds_a)]
        b = self.ds_b[random.randrange(len(self.ds_b))]
        return a, b
