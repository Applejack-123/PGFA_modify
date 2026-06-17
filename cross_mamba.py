import torch
import torch.nn as nn
from mamba_ssm import Mamba

class CrossModalMamba(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.rgb_mamba    = Mamba(d_model)
        self.ske_mamba    = Mamba(d_model)
        self.cross_gate   = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fusion_mamba = Mamba(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, rgb, skeleton):
        r = self.norm1(self.rgb_mamba(rgb) + rgb)
        s = self.norm1(self.ske_mamba(skeleton) + skeleton)

        gate  = self.cross_gate(torch.cat([r, s], dim=-1))
        fused = gate * r + (1 - gate) * s

        out = self.norm2(self.fusion_mamba(fused) + fused)
        return out


class MambaFusion(nn.Module):
    def __init__(self, d_model=512, num_layers=4):
        super().__init__()
        self.fusion_layers = nn.ModuleList([
            CrossModalMamba(d_model) for _ in range(num_layers)
        ])

    def forward(self, rgb, ske):
        for layer in self.fusion_layers:
            fused = layer(rgb, ske)
            rgb = fused
            ske = fused
        return fused