from typing import List

import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class WaveletLLLoss:
    def __init__(self, wave: str = "bior1.3", mode: str = "symmetric") -> None:
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ll_x, _ = self.dwt(x)
        ll_y, _ = self.dwt(y)
        return F.l1_loss(ll_x, ll_y)


def patch_nce_loss(features_q: List[torch.Tensor], features_k: List[torch.Tensor], num_patches: int, temperature: float = 0.07) -> torch.Tensor:
    losses = []
    for fq, fk in zip(features_q, features_k):
        b, c, h, w = fq.shape
        n = h * w
        fq = fq.view(b, c, n).permute(0, 2, 1)
        fk = fk.view(b, c, n).permute(0, 2, 1)
        idx = torch.randint(0, n, (b, min(num_patches, n)), device=fq.device)
        fq_s = torch.gather(fq, 1, idx.unsqueeze(-1).expand(-1, -1, c))
        fk_s = torch.gather(fk, 1, idx.unsqueeze(-1).expand(-1, -1, c))
        fq_s = F.normalize(fq_s, dim=-1)
        fk_norm = F.normalize(fk, dim=-1)
        logits = torch.bmm(fq_s, fk_norm.transpose(1, 2)) / temperature
        target = idx
        loss = F.cross_entropy(logits.view(-1, n), target.view(-1))
        losses.append(loss)
    return torch.stack(losses).mean()
