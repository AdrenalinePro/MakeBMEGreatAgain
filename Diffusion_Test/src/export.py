import argparse
import os

import torch

from config import ExportConfig
from model import WaveletUNet
from utils import ensure_dir, get_device


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default=ExportConfig.checkpoint_path)
    parser.add_argument("--output-path", type=str, default=ExportConfig.output_path)
    parser.add_argument("--image-size", type=int, default=ExportConfig.image_size)
    args = parser.parse_args()
    return ExportConfig(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        image_size=args.image_size,
    )


def export_model() -> None:
    cfg = parse_args()
    device = get_device()
    model = WaveletUNet().to(device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dummy_x = torch.randn(1, 1, cfg.image_size, cfg.image_size, device=device)
    dummy_c = torch.randn(1, 1, cfg.image_size, cfg.image_size, device=device)
    dummy_t = torch.zeros(1, dtype=torch.long, device=device)
    traced = torch.jit.trace(model, (dummy_x, dummy_t, dummy_c))
    ensure_dir(os.path.dirname(cfg.output_path))
    traced.save(cfg.output_path)


if __name__ == "__main__":
    export_model()
