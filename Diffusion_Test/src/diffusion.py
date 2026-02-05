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

    def _predict_eps_x0(self, model, xt: torch.Tensor, t: int, cond: torch.Tensor, pred_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        if pred_type == "x0":
            x0_pred = model(xt, t_batch, cond)
            alpha_bar = self.alpha_bars[t_batch].view(-1, 1, 1, 1)
            eps = (xt - torch.sqrt(alpha_bar) * x0_pred) / torch.sqrt(1.0 - alpha_bar)
            return eps, x0_pred
        eps = model(xt, t_batch, cond)
        x0_pred = self.predict_x0(xt, t_batch, eps)
        return eps, x0_pred

    def ddpm_step(self, model, xt: torch.Tensor, t: int, cond: torch.Tensor, pred_type: str = "eps") -> tuple[torch.Tensor, torch.Tensor]:
        eps, x0_pred = self._predict_eps_x0(model, xt, t, cond, pred_type)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
        mean = coef1 * (xt - coef2 * eps)
        if t == 0:
            return mean, x0_pred
        noise = torch.randn_like(xt)
        alpha_bar_prev = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=xt.device, dtype=xt.dtype)
        beta_tilde = self.betas[t] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        return mean + torch.sqrt(beta_tilde) * noise, x0_pred

    def ddim_step(
        self, model, xt: torch.Tensor, t: int, t_prev: int, cond: torch.Tensor, eta: float = 0.0, pred_type: str = "eps"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps, x0_pred = self._predict_eps_x0(model, xt, t, cond, pred_type)
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=xt.device, dtype=xt.dtype)
        if t == 0 or t_prev < 0:
            return x0_pred, x0_pred

        sigma = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
        )
        noise = torch.randn_like(xt) if eta > 0.0 else torch.zeros_like(xt)
        direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + direction + sigma * noise
        return x_prev, x0_pred

    def loss_diff(self, pred_eps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_eps, noise)
