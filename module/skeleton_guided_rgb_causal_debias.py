# module/skeleton_guided_rgb_causal_debias.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """
    Simple token-wise MLP block for [B, T, D] features.
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, D]

        Returns:
            out: [B, T, D]
        """

        return self.norm(x + self.net(x))


class SkeletonGuidedRGBCausalDebias(nn.Module):
    """
    Skeleton-guided RGB causal debiasing module.

    Motivation:
        RGB contains strong visual confounders:
            background, appearance, illumination, scene, camera view.

        Skeleton contains relatively invariant body-motion structure.

        Therefore:
            skeleton feature is used as causal motion anchor;
            RGB feature is decomposed into:
                rgb_causal   : action-relevant RGB feature
                rgb_spurious : RGB-specific non-causal feature

    Input:
        rgb_feat : [B, T, D]
        skl_feat : [B, T, D]

    Output:
        output dict:
            rgb_causal      : [B, T, D]
            rgb_spurious    : [B, T, D]
            debiased_rgb    : [B, T, D]
            fusion_feat     : [B, T, D]
            loss_causal     : scalar
            loss_align      : scalar
            loss_orth       : scalar
            loss_recon      : scalar
            loss_spurious   : scalar
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_temporal_conv: bool = True,
        fusion_mode: str = "gate",
        detach_skeleton_anchor: bool = False,
        eps: float = 1e-6
    ):
        super().__init__()

        assert fusion_mode in ["gate", "add", "concat"], \
            "fusion_mode must be one of ['gate', 'add', 'concat']"

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_temporal_conv = use_temporal_conv
        self.fusion_mode = fusion_mode
        self.detach_skeleton_anchor = detach_skeleton_anchor
        self.eps = eps

        # -------------------------------------------------
        # RGB causal / spurious decomposition
        # -------------------------------------------------
        self.rgb_causal_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            MLPBlock(dim, hidden_dim, dropout),
            MLPBlock(dim, hidden_dim, dropout)
        )

        self.rgb_spurious_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            MLPBlock(dim, hidden_dim, dropout),
            MLPBlock(dim, hidden_dim, dropout)
        )

        # -------------------------------------------------
        # Skeleton anchor projection
        # -------------------------------------------------
        self.skl_anchor_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        # -------------------------------------------------
        # Optional temporal modeling
        # Depth-wise Conv1d over temporal dimension
        # -------------------------------------------------
        if use_temporal_conv:
            self.rgb_causal_temporal = nn.Sequential(
                nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=1,
                    groups=dim,
                    bias=False
                ),
                nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=1,
                    bias=True
                ),
                nn.GELU()
            )

            self.rgb_spurious_temporal = nn.Sequential(
                nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=1,
                    groups=dim,
                    bias=False
                ),
                nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=1,
                    bias=True
                ),
                nn.GELU()
            )
        else:
            self.rgb_causal_temporal = None
            self.rgb_spurious_temporal = None

        # -------------------------------------------------
        # Debiased RGB projection
        # -------------------------------------------------
        self.debias_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        # -------------------------------------------------
        # Fusion
        # -------------------------------------------------
        if fusion_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim)
            )

        elif fusion_mode == "concat":
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim)
            )

        else:
            self.fusion_proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim)
            )

    def _temporal_conv(self, x, conv):
        """
        Args:
            x: [B, T, D]
            conv: Conv1d module

        Returns:
            out: [B, T, D]
        """

        if conv is None:
            return x

        residual = x
        x = x.transpose(1, 2)       # [B, D, T]
        x = conv(x)
        x = x.transpose(1, 2)       # [B, T, D]

        return x + residual

    def forward(
        self,
        rgb_feat,
        skl_feat,
        return_loss: bool = True
    ):
        """
        Args:
            rgb_feat: [B, T, D]
            skl_feat: [B, T, D]
            return_loss: whether to return auxiliary causal losses

        Returns:
            dict
        """

        assert rgb_feat.dim() == 3, \
            f"rgb_feat should be [B,T,D], but got {rgb_feat.shape}"

        assert skl_feat.dim() == 3, \
            f"skl_feat should be [B,T,D], but got {skl_feat.shape}"

        assert rgb_feat.shape == skl_feat.shape, \
            f"rgb_feat and skl_feat should have same shape, but got {rgb_feat.shape} and {skl_feat.shape}"

        B, T, D = rgb_feat.shape

        # -------------------------------------------------
        # Skeleton causal anchor
        # -------------------------------------------------
        skl_anchor = self.skl_anchor_proj(skl_feat)

        if self.detach_skeleton_anchor:
            skl_anchor_for_loss = skl_anchor.detach()
        else:
            skl_anchor_for_loss = skl_anchor

        # -------------------------------------------------
        # RGB decomposition
        # -------------------------------------------------
        rgb_causal = self.rgb_causal_encoder(rgb_feat)
        rgb_spurious = self.rgb_spurious_encoder(rgb_feat)

        rgb_causal = self._temporal_conv(
            rgb_causal,
            self.rgb_causal_temporal
        )

        rgb_spurious = self._temporal_conv(
            rgb_spurious,
            self.rgb_spurious_temporal
        )

        # Normalize before constraints
        rgb_causal_norm = F.normalize(
            rgb_causal,
            dim=-1,
            eps=self.eps
        )

        rgb_spurious_norm = F.normalize(
            rgb_spurious,
            dim=-1,
            eps=self.eps
        )

        skl_anchor_norm = F.normalize(
            skl_anchor_for_loss,
            dim=-1,
            eps=self.eps
        )

        # -------------------------------------------------
        # Debiased RGB
        # -------------------------------------------------
        # Keep causal part and suppress spurious part.
        debiased_rgb = rgb_feat + self.debias_proj(rgb_causal - rgb_spurious)

        # -------------------------------------------------
        # Fusion with skeleton
        # -------------------------------------------------
        if self.fusion_mode == "gate":
            gate = self.gate(
                torch.cat(
                    [debiased_rgb, skl_feat],
                    dim=-1
                )
            )  # [B,T,D]

            fusion_feat = gate * debiased_rgb + (1.0 - gate) * skl_feat
            fusion_feat = self.fusion_proj(fusion_feat)

        elif self.fusion_mode == "concat":
            fusion_feat = self.fusion_proj(
                torch.cat(
                    [debiased_rgb, skl_feat],
                    dim=-1
                )
            )

        else:
            fusion_feat = self.fusion_proj(
                debiased_rgb + skl_feat
            )

        output = {
            "rgb_causal": rgb_causal,
            "rgb_spurious": rgb_spurious,
            "debiased_rgb": debiased_rgb,
            "fusion_feat": fusion_feat
        }

        if return_loss:
            loss_dict = self.compute_losses(
                rgb_feat=rgb_feat,
                rgb_causal=rgb_causal_norm,
                rgb_spurious=rgb_spurious_norm,
                skl_anchor=skl_anchor_norm
            )
            output.update(loss_dict)

        return output

    def compute_losses(
        self,
        rgb_feat,
        rgb_causal,
        rgb_spurious,
        skl_anchor
    ):
        """
        Compute auxiliary causal losses.

        Args:
            rgb_feat:     original RGB feature [B,T,D]
            rgb_causal:   normalized causal feature [B,T,D]
            rgb_spurious: normalized spurious feature [B,T,D]
            skl_anchor:   normalized skeleton anchor [B,T,D]

        Returns:
            dict of scalar losses
        """

        # -------------------------------------------------
        # 1. causal alignment loss
        # Make RGB causal feature close to skeleton motion anchor.
        # -------------------------------------------------
        loss_align = 1.0 - torch.sum(
            rgb_causal * skl_anchor,
            dim=-1
        ).mean()

        # -------------------------------------------------
        # 2. orthogonal loss
        # Make RGB spurious feature decorrelated with skeleton anchor.
        # -------------------------------------------------
        spurious_skl_corr = torch.sum(
            rgb_spurious * skl_anchor,
            dim=-1
        )  # [B,T]

        loss_orth = (spurious_skl_corr ** 2).mean()

        # -------------------------------------------------
        # 3. causal-spurious separation loss
        # Make causal and spurious components different.
        # -------------------------------------------------
        causal_spurious_corr = torch.sum(
            rgb_causal * rgb_spurious,
            dim=-1
        )  # [B,T]

        loss_spurious = (causal_spurious_corr ** 2).mean()

        # -------------------------------------------------
        # 4. reconstruction-style stability loss
        # Avoid collapse.
        # Since rgb_causal/rgb_spurious here are normalized,
        # only use a weak constraint.
        # -------------------------------------------------
        rgb_feat_norm = F.normalize(
            rgb_feat,
            dim=-1,
            eps=self.eps
        )

        recon = F.normalize(
            rgb_causal + rgb_spurious,
            dim=-1,
            eps=self.eps
        )

        loss_recon = F.mse_loss(
            recon,
            rgb_feat_norm
        )

        # -------------------------------------------------
        # Total auxiliary causal loss
        # -------------------------------------------------
        loss_causal = (
            loss_align
            + 0.5 * loss_orth
            + 0.5 * loss_spurious
            + 0.1 * loss_recon
        )

        return {
            "loss_causal": loss_causal,
            "loss_align": loss_align,
            "loss_orth": loss_orth,
            "loss_spurious": loss_spurious,
            "loss_recon": loss_recon
        }


class RGBOnlyCausalDebias(nn.Module):
    """
    Fallback version when skeleton feature is unavailable.

    It decomposes RGB into causal/spurious parts,
    but does not use skeleton guidance.

    Input:
        rgb_feat: [B,T,D]

    Output:
        debiased_rgb: [B,T,D]
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()

        self.rgb_causal_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            MLPBlock(dim, hidden_dim, dropout),
            MLPBlock(dim, hidden_dim, dropout)
        )

        self.rgb_spurious_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            MLPBlock(dim, hidden_dim, dropout),
            MLPBlock(dim, hidden_dim, dropout)
        )

        self.debias_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, rgb_feat, return_loss=True):
        """
        Args:
            rgb_feat: [B,T,D]

        Returns:
            dict
        """

        rgb_causal = self.rgb_causal_encoder(rgb_feat)
        rgb_spurious = self.rgb_spurious_encoder(rgb_feat)

        debiased_rgb = rgb_feat + self.debias_proj(
            rgb_causal - rgb_spurious
        )

        output = {
            "rgb_causal": rgb_causal,
            "rgb_spurious": rgb_spurious,
            "debiased_rgb": debiased_rgb
        }

        if return_loss:
            rgb_causal_norm = F.normalize(rgb_causal, dim=-1)
            rgb_spurious_norm = F.normalize(rgb_spurious, dim=-1)

            corr = torch.sum(
                rgb_causal_norm * rgb_spurious_norm,
                dim=-1
            )

            loss_spurious = (corr ** 2).mean()

            output["loss_causal"] = loss_spurious
            output["loss_spurious"] = loss_spurious

        return output


if __name__ == "__main__":
    B = 8
    T = 8
    D = 512

    rgb_feat = torch.randn(B, T, D).cuda()
    skl_feat = torch.randn(B, T, D).cuda()

    model = SkeletonGuidedRGBCausalDebias(
        dim=D,
        hidden_dim=1024,
        dropout=0.1,
        use_temporal_conv=True,
        fusion_mode="gate",
        detach_skeleton_anchor=False
    ).cuda()

    out = model(
        rgb_feat=rgb_feat,
        skl_feat=skl_feat,
        return_loss=True
    )

    print("rgb_causal:", out["rgb_causal"].shape)
    print("rgb_spurious:", out["rgb_spurious"].shape)
    print("debiased_rgb:", out["debiased_rgb"].shape)
    print("fusion_feat:", out["fusion_feat"].shape)

    print("loss_causal:", out["loss_causal"].item())
    print("loss_align:", out["loss_align"].item())
    print("loss_orth:", out["loss_orth"].item())
    print("loss_spurious:", out["loss_spurious"].item())
    print("loss_recon:", out["loss_recon"].item())