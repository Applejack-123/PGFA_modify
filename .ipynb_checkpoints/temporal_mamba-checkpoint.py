# module/temporal_mamba.py

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class TemporalMamba(nn.Module):

    def __init__(
        self,
        feature_dim=512,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(feature_dim)
        self.mamba = Mamba(
            d_model=feature_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):

        residual = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = residual + self.dropout(x)
        x = self.norm2(x)

        return x