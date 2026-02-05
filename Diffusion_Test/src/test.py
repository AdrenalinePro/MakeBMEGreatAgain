import argparse
import os
from typing import List
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
        self.ds = InMemoryDataset(root, image_size, augment=False, per_file=True)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        return self.ds[idx], self.ds.paths[idx]


def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=TestConfig.input_dir)
    parser.add_argument("--output-dir", type=str, default=TestConfig.output_dir)
    parser.add_argument("--checkpoint-path", type=str, default=TestConfig.checkpoint_path)
    parser.add_argument("--timesteps", type=int, default=TestConfig.timesteps)
    parser.add_argument("--start-step", type=int, default=TestConfig.start_step)
    parser.add_argument("--manifold-alpha", type=float, default=TestConfig.manifold_alpha)
    parser.add_argument("--ddim-steps", type=int, default=TestConfig.ddim_steps)
    parser.add_argument("--ddim-eta", type=float, default=TestConfig.ddim_eta)
    parser.add_argument("--pred-type", type=str, default=TestConfig.pred_type, choices=["eps", "x0"])
    parser.add_argument("--init-mode", type=str, default=TestConfig.init_mode, choices=["noise", "sdedit"])
    parser.add_argument("--output-ext", type=str, default=TestConfig.output_ext)
    parser.add_argument("--vis-frames", type=int, default=TestConfig.vis_frames)
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
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        pred_type=args.pred_type,
        init_mode=args.init_mode,
        output_ext=args.output_ext,
        vis_frames=args.vis_frames,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )


def apply_manifold_constraint(x: torch.Tensor, ll_y: torch.Tensor, alpha: float, dwt: DWTForward, idwt: DWTInverse) -> torch.Tensor:
    ll_x, yh_x = dwt(x)
    ll_x = alpha * ll_x + (1.0 - alpha) * ll_y
    return idwt((ll_x, yh_x))


def to_uint8(array: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    if ref_max - ref_min < 1e-6:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array - ref_min) / (ref_max - ref_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def strip_nii_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def tensor_to_image(x: torch.Tensor, ref_min: float, ref_max: float) -> Image.Image:
    arr = x.squeeze(0).detach().float().cpu().numpy()
    u8 = to_uint8(arr, ref_min, ref_max)
    return Image.fromarray(u8, mode="L")


def make_progress_canvas(frames: List[Image.Image]) -> Image.Image:
    if not frames:
        raise ValueError("frames must not be empty")
    w, h = frames[0].size
    canvas = Image.new("L", (w * len(frames), h))
    for i, img in enumerate(frames):
        canvas.paste(img, (i * w, 0))
    return canvas


def choose_capture_indices(num_iters: int, num_mid: int) -> List[int]:
    if num_iters <= 0 or num_mid <= 0:
        return []
    idx = np.linspace(0, num_iters - 1, num_mid + 2, dtype=np.int64)[1:-1]
    return sorted(set(idx.astype(int).tolist()))


def make_ddim_timesteps(t_start: int, steps: int) -> List[int]:
    if steps <= 1:
        return [t_start, 0] if t_start > 0 else [0]
    ts = np.linspace(0, t_start, steps, dtype=np.int64)
    ts = np.unique(ts)
    if ts[-1] != t_start:
        ts = np.append(ts, t_start)
    if ts[0] != 0:
        ts = np.insert(ts, 0, 0)
    return ts[::-1].astype(int).tolist()


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
    dwt = DWTForward(J=1, wave="bior1.3", mode="symmetric").to(device)
    idwt = DWTInverse(wave="bior1.3", mode="symmetric").to(device)
    ensure_dir(cfg.output_dir)
    with torch.inference_mode():
        for cond, paths in tqdm(loader, desc="inference"):
            cond = cond.to(device)
            bsz = cond.shape[0]
            vis_frames = max(int(cfg.vis_frames), 2)
            num_mid = vis_frames - 2

            ref_mins: List[float] = []
            ref_maxs: List[float] = []
            frames: List[List[Image.Image]] = []
            for b in range(bsz):
                orig_np = cond[b].squeeze(0).detach().float().cpu().numpy()
                ref_min = float(orig_np.min())
                ref_max = float(orig_np.max())
                ref_mins.append(ref_min)
                ref_maxs.append(ref_max)
                frames.append([Image.fromarray(to_uint8(orig_np, ref_min, ref_max), mode="L")])

            t_start = min(cfg.start_step, cfg.timesteps - 1)
            if t_start <= 0:
                x0_pred = cond
            else:
                t_batch = torch.full((bsz,), t_start, device=device, dtype=torch.long)
                if cfg.init_mode == "noise":
                    xt = torch.randn_like(cond)
                else:
                    xt = schedule.q_sample(cond, t_batch, torch.randn_like(cond))

                ll_cond, _ = dwt(cond)

                timesteps = make_ddim_timesteps(t_start, cfg.ddim_steps)
                num_iters = max(len(timesteps) - 1, 0)
                capture = set(choose_capture_indices(num_iters, num_mid))
                for iter_idx, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                    xt, x0_pred = schedule.ddim_step(model, xt, t, t_prev, cond, eta=cfg.ddim_eta, pred_type=cfg.pred_type)
                    xt = apply_manifold_constraint(xt, ll_cond, cfg.manifold_alpha, dwt, idwt)
                    if iter_idx in capture:
                        for b in range(bsz):
                            frames[b].append(tensor_to_image(x0_pred[b], ref_mins[b], ref_maxs[b]))

            for b in range(bsz):
                frames[b].append(tensor_to_image(x0_pred[b], ref_mins[b], ref_maxs[b]))

                rel = os.path.relpath(paths[b], cfg.input_dir)
                out_dir = os.path.join(cfg.output_dir, os.path.dirname(rel))
                ensure_dir(out_dir)
                stem = strip_nii_suffix(os.path.basename(rel))
                out_path = os.path.join(out_dir, f"{stem}.{cfg.output_ext}")
                canvas = make_progress_canvas(frames[b])
                canvas.save(out_path)


if __name__ == "__main__":
    test()
