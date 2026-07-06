# rgb_only_module.py
# -*- coding: utf-8 -*-
"""
RGB-only adapter module.

Purpose:
    Convert pretrained RGB / VideoCLIP features into a trainable RGB-only token feature.

Input:
    rgb_feat: [B, T, 512]

Output:
    rgb_tokens: [B, T, 512]

Typical usage:
    rgb_tokens = self.rgb_only(rgb_feat)       # [B, T, 512]
    rgb_feature = rgb_tokens.mean(dim=1)       # [B, 512]
    acc_batch, pred = get_acc(rgb_feature, text_bank_512, unseen_label, label)

Recommended:
    Train this module with the same text prototype loss as your skeleton/fusion branch,
    then save its predictions for oracle analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Lightweight RMSNorm for token features.
    Input / output:
        x: [B, T, D]
    """

    def __init__(self, dim: int = 512, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RGBTokenMLPBlock(nn.Module):
    """
    Token-wise MLP block.
    It does not change shape.

    Input:
        x: [B, T, D]
    Output:
        x: [B, T, D]
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_rmsnorm: bool = False,
        init_gamma: float = 0.1,
    ):
        super().__init__()

        self.norm = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Residual scale. A small non-zero init helps training faster than pure identity.
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(self.norm(x))
        return x + torch.tanh(self.gamma) * h


class RGBTemporalMixBlock(nn.Module):
    """
    Lightweight temporal mixing block.
    It uses depthwise Conv1d over T and keeps output shape unchanged.

    Input:
        x: [B, T, D]
    Output:
        x: [B, T, D]
    """

    def __init__(
        self,
        dim: int = 512,
        kernel_size: int = 3,
        dropout: float = 0.1,
        init_gamma: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.norm = nn.LayerNorm(dim)

        self.dw_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
        )

        self.pw_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        h = self.norm(x)
        h = h.transpose(1, 2)      # [B, D, T]
        h = self.dw_conv(h)
        h = F.gelu(h)
        h = self.pw_conv(h)
        h = h.transpose(1, 2)      # [B, T, D]
        h = self.dropout(h)

        return x + torch.tanh(self.gamma) * h


class RGBOnlyModule(nn.Module):
    """
    RGB-only feature adapter.

    This module keeps sequence output:
        [B, T, 512] -> [B, T, 512]

    It contains:
        1. input projection / adapter
        2. optional temporal mixing
        3. token-wise MLP adaptation
        4. final norm

    Args:
        dim:
            feature dimension, default 512.
        hidden_dim:
            MLP hidden dimension.
        depth:
            number of RGBTokenMLPBlock blocks.
        temporal_depth:
            number of RGBTemporalMixBlock blocks.
        dropout:
            dropout rate.
        use_temporal:
            whether to use temporal Conv1d mixing over T.
        pool:
            default pooling method when calling encode().
            "mean" or "attn".
            forward() always returns [B, T, 512].
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 1024,
        depth: int = 2,
        temporal_depth: int = 1,
        dropout: float = 0.1,
        use_temporal: bool = True,
        pool: str = "mean",
    ):
        super().__init__()

        if pool not in ["mean", "attn"]:
            raise ValueError(f"pool must be 'mean' or 'attn', but got {pool}")

        self.dim = dim
        self.pool = pool
        self.use_temporal = use_temporal

        self.input_norm = nn.LayerNorm(dim)

        # A light input adapter. Shape remains [B, T, D].
        self.input_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.input_gamma = nn.Parameter(torch.tensor(0.1))

        if use_temporal:
            self.temporal_blocks = nn.ModuleList([
                RGBTemporalMixBlock(
                    dim=dim,
                    kernel_size=3,
                    dropout=dropout,
                    init_gamma=0.1,
                )
                for _ in range(temporal_depth)
            ])
        else:
            self.temporal_blocks = nn.ModuleList([])

        self.mlp_blocks = nn.ModuleList([
            RGBTokenMLPBlock(
                dim=dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                init_gamma=0.1,
            )
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(dim)

        # Attention pooling is only used by encode(pool="attn").
        # forward() still returns [B, T, 512].
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, rgb_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_feat: [B, T, 512]

        Returns:
            rgb_tokens: [B, T, 512]
        """
        if rgb_feat.dim() != 3:
            raise ValueError(
                f"RGBOnlyModule expects rgb_feat shape [B,T,D], "
                f"but got {tuple(rgb_feat.shape)}"
            )

        x = rgb_feat

        # Input adapter.
        h = self.input_proj(self.input_norm(x))
        x = x + torch.tanh(self.input_gamma) * h

        for block in self.temporal_blocks:
            x = block(x)

        for block in self.mlp_blocks:
            x = block(x)

        x = self.out_norm(x)

        return x

    def encode(self, rgb_feat: torch.Tensor, pool: str = None) -> torch.Tensor:
        """
        Convert RGB tokens into one clip-level feature for text matching.

        Args:
            rgb_feat: [B, T, 512]
            pool: "mean" or "attn". If None, use self.pool.

        Returns:
            rgb_feature: [B, 512]
        """
        pool = self.pool if pool is None else pool

        tokens = self.forward(rgb_feat)  # [B, T, D]

        if pool == "mean":
            return tokens.mean(dim=1)

        if pool == "attn":
            score = self.attn_pool(tokens)             # [B, T, 1]
            weight = torch.softmax(score, dim=1)       # [B, T, 1]
            return (tokens * weight).sum(dim=1)        # [B, D]

        raise ValueError(f"Unsupported pool: {pool}")


class RGBOnlyWrapper(nn.Module):
    """
    Optional wrapper for RGB-only zero-shot training.

    It returns both:
        rgb_tokens: [B, T, 512]
        rgb_feature: [B, 512]
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 1024,
        depth: int = 2,
        temporal_depth: int = 1,
        dropout: float = 0.1,
        use_temporal: bool = True,
        pool: str = "mean",
    ):
        super().__init__()

        self.rgb_encoder = RGBOnlyModule(
            dim=dim,
            hidden_dim=hidden_dim,
            depth=depth,
            temporal_depth=temporal_depth,
            dropout=dropout,
            use_temporal=use_temporal,
            pool=pool,
        )
        self.pool = pool

    def forward(self, rgb_feat: torch.Tensor):
        """
        Args:
            rgb_feat: [B, T, 512]

        Returns:
            dict:
                rgb_tokens: [B, T, 512]
                rgb_feature: [B, 512]
        """
        rgb_tokens = self.rgb_encoder(rgb_feat)

        if self.pool == "mean":
            rgb_feature = rgb_tokens.mean(dim=1)
        elif self.pool == "attn":
            score = self.rgb_encoder.attn_pool(rgb_tokens)
            weight = torch.softmax(score, dim=1)
            rgb_feature = (rgb_tokens * weight).sum(dim=1)
        else:
            raise ValueError(f"Unsupported pool: {self.pool}")

        return {
            "rgb_tokens": rgb_tokens,
            "rgb_feature": rgb_feature,
        }


if __name__ == "__main__":
    B, T, D = 4, 8, 512
    rgb = torch.randn(B, T, D).cuda()

    model = RGBOnlyModule(
        dim=512,
        hidden_dim=1024,
        depth=2,
        temporal_depth=1,
        dropout=0.1,
        use_temporal=True,
        pool="mean",
    ).cuda()

    tokens = model(rgb)
    pooled = model.encode(rgb)

    print("input:", rgb.shape)
    print("tokens:", tokens.shape)
    print("pooled:", pooled.shape)

    wrapper = RGBOnlyWrapper(pool="attn").cuda()
    out = wrapper(rgb)
    print("wrapper rgb_tokens:", out["rgb_tokens"].shape)
    print("wrapper rgb_feature:", out["rgb_feature"].shape)
