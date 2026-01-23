from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_035t: str = "data/0_35T"
    data_15t: str = "data/1_5T"
    output_dir: str = "outputs"
    batch_size: int = 4
    num_epochs: int = 60
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    image_size: int = 352
    timesteps: int = 1000
    save_every: int = 5
    keep_checkpoints: int = 10
    lambda_nce: float = 1.0
    lambda_wave: float = 10.0
    patch_nce_patches: int = 256
    seed: int = 42


@dataclass
class TestConfig:
    input_dir: str = "data/0_35T"
    output_dir: str = "outputs/inference"
    checkpoint_path: str = "outputs/checkpoints/latest.pt"
    timesteps: int = 1000
    start_step: int = 500
    manifold_alpha: float = 0.8
    output_ext: str = "png"
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 2
    image_size: int = 352


@dataclass
class ExportConfig:
    checkpoint_path: str = "outputs/checkpoints/latest.pt"
    output_path: str = "outputs/exported_model.pt"
    image_size: int = 352
