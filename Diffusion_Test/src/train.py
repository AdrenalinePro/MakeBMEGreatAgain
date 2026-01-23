import argparse
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from data import UnpairedDataset
from diffusion import DiffusionSchedule
from losses import WaveletLLLoss, patch_nce_loss
from model import WaveletUNet
from utils import current_time_str, ensure_dir, get_device, rotate_checkpoints, set_seed


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-035t", type=str, default=TrainConfig.data_035t)
    parser.add_argument("--data-15t", type=str, default=TrainConfig.data_15t)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
    parser.add_argument("--timesteps", type=int, default=TrainConfig.timesteps)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--keep-checkpoints", type=int, default=TrainConfig.keep_checkpoints)
    parser.add_argument("--lambda-nce", type=float, default=TrainConfig.lambda_nce)
    parser.add_argument("--lambda-wave", type=float, default=TrainConfig.lambda_wave)
    parser.add_argument("--patch-nce-patches", type=int, default=TrainConfig.patch_nce_patches)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()
    return TrainConfig(
        data_035t=args.data_035t,
        data_15t=args.data_15t,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=args.image_size,
        timesteps=args.timesteps,
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
        lambda_nce=args.lambda_nce,
        lambda_wave=args.lambda_wave,
        patch_nce_patches=args.patch_nce_patches,
        seed=args.seed,
    )


def save_checkpoint(output_dir: str, epoch: int, model: WaveletUNet, optimizer: torch.optim.Optimizer) -> str:
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    ensure_dir(checkpoint_dir)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )
    latest = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        latest,
    )
    return checkpoint_dir


def train() -> None:
    cfg = parse_args()
    if cfg.seed < 0:
        cfg.seed = random.SystemRandom().randint(0, 2**31 - 1)
    set_seed(cfg.seed)
    device = get_device()
    dataset = UnpairedDataset(cfg.data_15t, cfg.data_035t, cfg.image_size, augment=True)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    model = WaveletUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    schedule = DiffusionSchedule(cfg.timesteps, device)
    wavelet_loss = WaveletLLLoss()
    ensure_dir(cfg.output_dir)
    epoch_times = []
    for epoch in range(1, cfg.num_epochs + 1):
        if cfg.seed < 0:
            cfg.seed = random.SystemRandom().randint(0, 2**31 - 1)
        set_seed(cfg.seed)
        model.train()
        start_time = time.time()
        losses = []
        for x15t, c035t in tqdm(loader, desc=f"epoch {epoch}/{cfg.num_epochs}"):
            x15t = x15t.to(device)
            c035t = c035t.to(device)
            t = torch.randint(0, cfg.timesteps, (x15t.shape[0],), device=device)
            noise = torch.randn_like(x15t)
            xt = schedule.q_sample(x15t, t, noise)
            pred_eps = model(xt, t, c035t)
            loss_diff = schedule.loss_diff(pred_eps, noise)
            x0_pred = schedule.predict_x0(xt, t, pred_eps)
            features_pred = model.encode_features(x0_pred, t)
            features_c = model.encode_features(c035t, t)
            loss_nce = patch_nce_loss(features_pred[1:], features_c[1:], cfg.patch_nce_patches)
            loss_wave = wavelet_loss(x0_pred, c035t)
            total = loss_diff + cfg.lambda_nce * loss_nce + cfg.lambda_wave * loss_wave
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            losses.append(total.item())
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (cfg.num_epochs - epoch)
        print(
            f"epoch {epoch} loss {sum(losses)/len(losses):.4f} time {epoch_time:.1f}s eta {eta:.1f}s now {current_time_str()}"
        )
        if epoch % cfg.save_every == 0 or epoch == cfg.num_epochs:
            checkpoint_dir = save_checkpoint(cfg.output_dir, epoch, model, optimizer)
            rotate_checkpoints(checkpoint_dir, cfg.keep_checkpoints)


if __name__ == "__main__":
    train()
