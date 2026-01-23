import torch
import torch.nn.functional as F


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device) -> None:
        self.timesteps = timesteps
        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return (xt - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

    def p_sample(self, model, xt: torch.Tensor, t: int, cond: torch.Tensor) -> torch.Tensor:
        t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        eps = model(xt, t_batch, cond)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
        mean = coef1 * (xt - coef2 * eps)
        if t == 0:
            return mean
        noise = torch.randn_like(xt)
        sigma = torch.sqrt(self.betas[t])
        return mean + sigma * noise

    def loss_diff(self, pred_eps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_eps, noise)
