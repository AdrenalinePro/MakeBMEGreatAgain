import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import TestConfig
from data import InMemoryDataset
from diffusion import DiffusionSchedule
from model import WaveletUNet
from utils import ensure_dir, get_device, set_seed
from pytorch_wavelets import DWTForward, DWTInverse


class SingleDomainDataset(Dataset):
    def __init__(self, root: str, image_size: int) -> None:
        self.ds = InMemoryDataset(root, image_size, augment=False)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.ds[idx]


def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=TestConfig.input_dir)
    parser.add_argument("--output-dir", type=str, default=TestConfig.output_dir)
    parser.add_argument("--checkpoint-path", type=str, default=TestConfig.checkpoint_path)
    parser.add_argument("--timesteps", type=int, default=TestConfig.timesteps)
    parser.add_argument("--start-step", type=int, default=TestConfig.start_step)
    parser.add_argument("--manifold-alpha", type=float, default=TestConfig.manifold_alpha)
    parser.add_argument("--output-ext", type=str, default=TestConfig.output_ext)
    parser.add_argument("--seed", type=int, default=TestConfig.seed)
    parser.add_argument("--batch-size", type=int, default=TestConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=TestConfig.num_workers)
    parser.add_argument("--image-size", type=int, default=TestConfig.image_size)
    args = parser.parse_args()
    return TestConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        timesteps=args.timesteps,
        start_step=args.start_step,
        manifold_alpha=args.manifold_alpha,
        output_ext=args.output_ext,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )


def apply_manifold_constraint(x: torch.Tensor, y: torch.Tensor, alpha: float, dwt: DWTForward, idwt: DWTInverse) -> torch.Tensor:
    ll_x, yh_x = dwt(x)
    ll_y, _ = dwt(y)
    ll_x = alpha * ll_x + (1.0 - alpha) * ll_y
    return idwt((ll_x, yh_x))


def to_uint8(array: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    if ref_max - ref_min < 1e-6:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array - ref_min) / (ref_max - ref_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def save_side_by_side(original: torch.Tensor, enhanced: torch.Tensor, path: str) -> None:
    orig_np = original.squeeze(0).cpu().numpy()
    enh_np = enhanced.squeeze(0).cpu().numpy()
    ref_min = float(orig_np.min())
    ref_max = float(orig_np.max())
    orig_u8 = to_uint8(orig_np, ref_min, ref_max)
    enh_u8 = to_uint8(enh_np, ref_min, ref_max)
    orig_img = Image.fromarray(orig_u8, mode="L")
    enh_img = Image.fromarray(enh_u8, mode="L")
    w, h = orig_img.size
    canvas = Image.new("L", (w * 2, h))
    canvas.paste(orig_img, (0, 0))
    canvas.paste(enh_img, (w, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def test() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device()
    dataset = SingleDomainDataset(cfg.input_dir, cfg.image_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    model = WaveletUNet().to(device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    schedule = DiffusionSchedule(cfg.timesteps, device)
    dwt = DWTForward(J=1, wave="bior1.3", mode="symmetric")
    idwt = DWTInverse(wave="bior1.3", mode="symmetric")
    ensure_dir(cfg.output_dir)
    with torch.no_grad():
        for idx, cond in enumerate(tqdm(loader, desc="inference")):
            cond = cond.to(device)
            t_start = min(cfg.start_step, cfg.timesteps - 1)
            t_batch = torch.full((cond.shape[0],), t_start, device=device, dtype=torch.long)
            xt = schedule.q_sample(cond, t_batch, torch.randn_like(cond))
            for t in reversed(range(t_start + 1)):
                xt = schedule.p_sample(model, xt, t, cond)
                xt = apply_manifold_constraint(xt, cond, cfg.manifold_alpha, dwt, idwt)
            for b in range(xt.shape[0]):
                path = os.path.join(cfg.output_dir, f"sample_{idx * cfg.batch_size + b:05d}.{cfg.output_ext}")
                save_side_by_side(cond[b], xt[b], path)


if __name__ == "__main__":
    test()
