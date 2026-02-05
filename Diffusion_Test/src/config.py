from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_035t: str = "data/0_35T"
    data_15t: str = "data/1_5T"
    output_dir: str = "outputs"
    batch_size: int = 6
    num_epochs: int = 200
    learning_rate: float = 6e-5
    min_learning_rate: float = 1e-6
    warmup_steps: int = 500
    weight_decay: float = 1e-4
    num_workers: int = 12
    image_size: int = 352
    timesteps: int = 800
    save_every: int = 5
    keep_checkpoints: int = 10
    lambda_nce: float = 0.6
    lambda_wave: float = 4
    patch_nce_patches: int = 256
    seed: int = 42


@dataclass
class TestConfig:
    input_dir: str = "data/0_35T"
    output_dir: str = "outputs/inference"
    checkpoint_path: str = "outputs/checkpoints/latest.pt"
    timesteps: int = 800
    start_step: int = 400
    manifold_alpha: float = 0.8
    ddim_steps: int = 100
    ddim_eta: float = 0.0
    pred_type: str = "eps"
    init_mode: str = "sdedit"
    output_ext: str = "png"
    vis_frames: int = 6
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 8
    image_size: int = 352


@dataclass
class ExportConfig:
    checkpoint_path: str = "outputs/checkpoints/latest.pt"
    output_path: str = "outputs/exported_model.pt"
    image_size: int = 352
