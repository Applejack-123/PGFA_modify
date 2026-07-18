# safe_rgb_temporal_adapter.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBAdapter(nn.Module):
    """
    Identity-safe RGB temporal adapter.

    Input : rgb_feat [B, T, 512]
    Output: rgb_feat [B, T, 512]

    It starts almost exactly as identity:
        out ≈ x

    Compared with the previous medium adapter:
        1. No final LayerNorm by default.
        2. Last projection layers are zero-initialized.
        3. Residual gates are initialized very small.
        4. Temporal conv only works as a small residual correction.
    """

    def __init__(
        self,
        dim=512,
        hidden=128,
        dropout=0.1,
        kernel_size=3,
        init_gate=-6.0,
        use_temporal=True,
        final_norm=False,
    ):
        super().__init__()

        self.use_temporal = use_temporal

        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.pwconv = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop2 = nn.Dropout(dropout)

        self.gate_channel = nn.Parameter(torch.tensor(float(init_gate)))
        self.gate_temporal = nn.Parameter(torch.tensor(float(init_gate)))

        self.out_norm = nn.LayerNorm(dim) if final_norm else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        """
        Args:
            x: [B, T, 512]

        Returns:
            x: [B, T, 512]
        """
        if x.dim() != 3:
            raise ValueError(f"SafeRGBTemporalAdapter expects [B,T,D], got {tuple(x.shape)}")

        h = self.norm1(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.drop1(h)
        h = self.fc2(h)
        x = x + torch.sigmoid(self.gate_channel) * h

        if self.use_temporal:
            h = self.norm2(x).transpose(1, 2)
            h = self.dwconv(h)
            h = F.gelu(h)
            h = self.pwconv(h)
            h = h.transpose(1, 2)
            h = self.drop2(h)
            x = x + torch.sigmoid(self.gate_temporal) * h

        return self.out_norm(x)
