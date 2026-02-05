from typing import List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        half = dim // 2
        if half <= 0:
            inv_freq = torch.empty(0)
        else:
            denom = max(half - 1, 1)
            inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / denom)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.float().unsqueeze(1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).view(b, c, h * w).permute(0, 2, 1)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.permute(0, 2, 1).view(b, c, h, w)
        return x + y


class DWTBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, wave: str = "bior1.3", mode: str = "symmetric") -> None:
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.proj = nn.Conv2d(in_channels * 4, out_channels, 1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_l, y_h = self.dwt(x)
        y_h = y_h[0]
        lh = y_h[:, :, 0]
        hl = y_h[:, :, 1]
        hh = y_h[:, :, 2]
        y = torch.cat([y_l, lh, hl, hh], dim=1)
        y = self.proj(y)
        y = self.act(self.norm(y))
        return y


class IDWTBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, wave: str = "bior1.3", mode: str = "symmetric") -> None:
        super().__init__()
        self.idwt = DWTInverse(wave=wave, mode=mode)
        self.proj = nn.Conv2d(in_channels, out_channels * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        c = y.shape[1] // 4
        ll, lh, hl, hh = torch.chunk(y, 4, dim=1)
        y_h = torch.stack([lh, hl, hh], dim=2)
        out = self.idwt((ll, [y_h]))
        return out


class FeatureEncoder(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(1, base_channels, 3, padding=1)
        self.level1_dwt = DWTBlock(base_channels, base_channels * 2)
        self.level1_rb1 = ResBlock(base_channels * 2, base_channels * 2, base_channels * 4)
        self.level1_rb2 = ResBlock(base_channels * 2, base_channels * 2, base_channels * 4)
        self.level2_dwt = DWTBlock(base_channels * 2, base_channels * 4)
        self.level2_rb1 = ResBlock(base_channels * 4, base_channels * 4, base_channels * 4)
        self.level2_rb2 = ResBlock(base_channels * 4, base_channels * 4, base_channels * 4)
        self.level2_sa = SelfAttention(base_channels * 4)
        self.level3_dwt = DWTBlock(base_channels * 4, base_channels * 8)
        self.level3_rb1 = ResBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        self.level3_rb2 = ResBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        self.level3_sa = SelfAttention(base_channels * 8)
        self.level4_dwt = DWTBlock(base_channels * 8, base_channels * 8)
        self.level4_rb1 = ResBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        self.level4_rb2 = ResBlock(base_channels * 8, base_channels * 8, base_channels * 4)
        self.level4_sa = SelfAttention(base_channels * 8)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> List[torch.Tensor]:
        h = self.in_conv(x)
        h = self.level1_dwt(h)
        h = self.level1_rb1(h, t_embed)
        h = self.level1_rb2(h, t_embed)
        f1 = h
        h = self.level2_dwt(h)
        h = self.level2_rb1(h, t_embed)
        h = self.level2_rb2(h, t_embed)
        h = self.level2_sa(h)
        f2 = h
        h = self.level3_dwt(h)
        h = self.level3_rb1(h, t_embed)
        h = self.level3_rb2(h, t_embed)
        h = self.level3_sa(h)
        f3 = h
        h = self.level4_dwt(h)
        h = self.level4_rb1(h, t_embed)
        h = self.level4_rb2(h, t_embed)
        h = self.level4_sa(h)
        f4 = h
        return [f1, f2, f3, f4]


class WaveletUNet(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(2, base_channels, 3, padding=1)
        self.level1_dwt = DWTBlock(base_channels, base_channels * 2)
        self.level1_rb1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.level1_rb2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.level2_dwt = DWTBlock(base_channels * 2, base_channels * 4)
        self.level2_rb1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.level2_rb2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.level2_sa = SelfAttention(base_channels * 4)
        self.level3_dwt = DWTBlock(base_channels * 4, base_channels * 8)
        self.level3_rb1 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.level3_rb2 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.level3_sa = SelfAttention(base_channels * 8)
        self.level4_dwt = DWTBlock(base_channels * 8, base_channels * 8)
        self.level4_rb1 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.level4_rb2 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.level4_sa = SelfAttention(base_channels * 8)
        self.mid_rb1 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.mid_rb2 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.mid_sa = SelfAttention(base_channels * 8)
        self.mid_rb3 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.up4_idwt = IDWTBlock(base_channels * 8, base_channels * 8)
        self.up4_rb1 = ResBlock(base_channels * 16, base_channels * 8, time_dim)
        self.up4_rb2 = ResBlock(base_channels * 8, base_channels * 8, time_dim)
        self.up4_sa = SelfAttention(base_channels * 8)
        self.up3_idwt = IDWTBlock(base_channels * 8, base_channels * 4)
        self.up3_rb1 = ResBlock(base_channels * 12, base_channels * 4, time_dim)
        self.up3_rb2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.up3_sa = SelfAttention(base_channels * 4)
        self.up2_idwt = IDWTBlock(base_channels * 4, base_channels * 2)
        self.up2_rb1 = ResBlock(base_channels * 6, base_channels * 2, time_dim)
        self.up2_rb2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.up2_sa = SelfAttention(base_channels * 2)
        self.up1_idwt = IDWTBlock(base_channels * 2, base_channels)
        self.up1_rb1 = ResBlock(base_channels * 3, base_channels, time_dim)
        self.up1_rb2 = ResBlock(base_channels, base_channels, time_dim)
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.feature_encoder = FeatureEncoder(base_channels)

    def encode_features(self, x: torch.Tensor, t: torch.Tensor) -> List[torch.Tensor]:
        t_embed = self.time_embed(t)
        return self.feature_encoder(x, t_embed)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        h = self.in_conv(torch.cat([x, cond], dim=1))
        h = self.level1_dwt(h)
        h = self.level1_rb1(h, t_embed)
        h = self.level1_rb2(h, t_embed)
        s1 = h
        h = self.level2_dwt(h)
        h = self.level2_rb1(h, t_embed)
        h = self.level2_rb2(h, t_embed)
        h = self.level2_sa(h)
        s2 = h
        h = self.level3_dwt(h)
        h = self.level3_rb1(h, t_embed)
        h = self.level3_rb2(h, t_embed)
        h = self.level3_sa(h)
        s3 = h
        h = self.level4_dwt(h)
        h = self.level4_rb1(h, t_embed)
        h = self.level4_rb2(h, t_embed)
        h = self.level4_sa(h)
        s4 = h
        h = self.mid_rb1(h, t_embed)
        h = self.mid_rb2(h, t_embed)
        h = self.mid_sa(h)
        h = self.mid_rb3(h, t_embed)
        h = self.up4_idwt(h)
        h, s4 = self._match_size(h, s4)
        h = torch.cat([h, s4], dim=1)
        h = self.up4_rb1(h, t_embed)
        h = self.up4_rb2(h, t_embed)
        h = self.up4_sa(h)
        h = self.up3_idwt(h)
        h, s3 = self._match_size(h, s3)
        h = torch.cat([h, s3], dim=1)
        h = self.up3_rb1(h, t_embed)
        h = self.up3_rb2(h, t_embed)
        h = self.up3_sa(h)
        h = self.up2_idwt(h)
        h, s2 = self._match_size(h, s2)
        h = torch.cat([h, s2], dim=1)
        h = self.up2_rb1(h, t_embed)
        h = self.up2_rb2(h, t_embed)
        if h.shape[-2] * h.shape[-1] <= 88 * 88:
            h = self.up2_sa(h)
        h = self.up1_idwt(h)
        h, s1 = self._match_size(h, s1)
        h = torch.cat([h, s1], dim=1)
        h = self.up1_rb1(h, t_embed)
        h = self.up1_rb2(h, t_embed)
        h = self.out_conv(self.out_act(self.out_norm(h)))
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return h

    @staticmethod
    def _match_size(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if a.shape[-2:] == b.shape[-2:]:
            return a, b
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return a, b
