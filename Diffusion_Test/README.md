- # WGD-MRI (Wavelet-Guided Diffusion) — English README

  This project implements a wavelet-based conditional diffusion model for neonatal brain MRI enhancement. The primary goal is to enhance low-field (0.35T) MRI into a cleaner, 1.5T-like appearance while preserving anatomical structure.

  ## Overview

  **Task**: Unpaired conditional enhancement (0.35T → 1.5T style).

  **Key idea**:

  - Train a diffusion denoiser on the target domain (1.5T) so the model learns the clean 1.5T distribution.
  - Use structure constraints to preserve content from the source domain (0.35T) even though the data is *unpaired*.

  ## Model

  **Backbone**: Wavelet-Diffusion U-Net ([model.py](file:///f:/Diffusion_Test/src/model.py))

  - U-Net with wavelet down/up sampling (DWT/IDWT) to reduce aliasing and keep detail during multi-scale processing.
  - Self-attention blocks at intermediate resolutions.
  - Input is concatenated as `[x_t, cond]` where `cond` is the 0.35T image and `x_t` is the noisy latent at diffusion step `t`.

  **Diffusion schedule**: Linear beta schedule ([diffusion.py](file:///f:/Diffusion_Test/src/diffusion.py))

  - Training optimizes the standard noise-prediction objective (`eps` prediction).
  - Inference uses DDIM sampling only.

  ## Losses (Training Objective)

  The training loss is a weighted sum ([train.py](file:///f:/Diffusion_Test/src/train.py), [losses.py](file:///f:/Diffusion_Test/src/losses.py)):

  - **Diffusion loss**: MSE between predicted noise and true noise on 1.5T samples.
  - **PatchNCE loss**: Contrastive feature matching between generated output and the 0.35T condition to keep structure.
  - **Wavelet LL loss**: L1 distance between low-frequency (LL) wavelet components of output and condition.

  During training, the script prints epoch averages for:
  `total / diff / nce / wave` plus the current learning rate.

  ## Setup

  Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

  ## Data

  Place data under:

  - `data/0_35T` for conditional inputs (0.35T)
  - `data/1_5T` for target domain samples (1.5T)

  Supported formats: `.nii`, `.nii.gz`, `.png/.jpg/...`, `.npy`

  Notes:

  - Training uses axial slicing for NIfTI volumes.
  - Inference processes each file and exports a single comparison canvas image per input file.

  ## Training

  Example:

  ```bash
  python src/train.py --data-035t data/0_35T --data-15t data/1_5T --output-dir outputs
  ```

  Random seed per epoch (useful for augmentation randomness):

  ```bash
  python src/train.py --data-035t data/0_35T --data-15t data/1_5T --output-dir outputs --seed -1
  ```

  ### Learning rate schedule (Warmup + Cosine Annealing)

  The training loop uses:

  - **Warmup**: linearly increase LR from `0` to `learning_rate` over `warmup_steps`.
  - **Cosine decay**: decay LR from `learning_rate` down to `min_learning_rate` towards the end of training.

  Relevant arguments:

  - `--learning-rate` (max LR, default is defined in config)
  - `--min-learning-rate` (default `1e-6`)
  - `--warmup-steps` (default `1000`)

  Example:

  ```bash
  python src/train.py --learning-rate 5e-5 --min-learning-rate 1e-6 --warmup-steps 1000
  ```

  ## Inference / Enhancement

  Inference uses **DDIM only** ([test.py](file:///f:/Diffusion_Test/src/test.py)).

  Example:

  ```bash
  python src/test.py \
    --input-dir data/0_35T \
    --checkpoint-path outputs/checkpoints/latest.pt \
    --output-dir outputs/inference \
    --pred-type eps \
    --init-mode sdedit \
    --ddim-steps 200 \
    --start-step 200
  ```

  Key arguments:

  - `--timesteps`: diffusion timesteps (must match training)
  - `--start-step`: SDEdit starting step (lower = less noise; higher = stronger enhancement but harder)
  - `--ddim-steps`: number of DDIM reverse steps
  - `--ddim-eta`: noise level for DDIM (0 = deterministic)
  - `--pred-type`: model output type (`eps` recommended; training is noise-prediction)
  - `--init-mode`: `sdedit` (noise the input then denoise) or `noise` (start from pure noise)
  - `--manifold-alpha`: LL wavelet manifold constraint strength during sampling

  Output:

  - A single canvas image per input file showing the input on the left and the final output on the right, with intermediate snapshots in between.

  ## Current Known Issues / Work in Progress

  - **Noisy outputs at high `start-step`**: when `start-step` is large, the model may produce outputs where anatomy is visible but residual noise remains. This is typically addressed by longer training, better LR scheduling, and tuning `lambda_nce / lambda_wave`.
  - **Hyperparameter sensitivity**: the balance between diffusion loss and structure constraints can lead to “structure preserved but denoising weak” if auxiliary losses dominate.
  - **Timesteps vs. sampling steps tradeoff**: using large training timesteps but too few DDIM reverse steps can leave residual noise.

  
