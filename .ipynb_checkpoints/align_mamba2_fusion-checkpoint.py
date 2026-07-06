# align_mamba_fusion.py
# -*- coding: utf-8 -*-
import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba

from Causal import EnhancedCausalIntervention


def build_mamba(dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
    if Mamba is None:
        raise ImportError("Cannot find mamba_ssm.")
    try:
        return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
    except TypeError:
        try:
            return Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        except TypeError:
            return Mamba(dim, d_state=d_state, expand=expand)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class MambaResidualBlock(nn.Module):
    def __init__(self, dim=512, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mamba = build_mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class ModalityExpertLinear(nn.Module):
    """
    output = shared_expert(x) + modality_specific_expert_m(x)
    modality_ids: 0 skeleton, 1 rgb, 2 text
    """

    def __init__(self, dim=512, num_modalities=3, bias=True):
        super().__init__()
        self.num_modalities = num_modalities
        self.shared = nn.Linear(dim, dim, bias=bias)
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim, bias=bias) for _ in range(num_modalities)
        ])

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        if modality_ids.dim() == 1:
            modality_ids = modality_ids.unsqueeze(0).expand(B, L)
        modality_ids = modality_ids.to(device=x.device, dtype=torch.long)

        out_shared = self.shared(x)
        out_specific = torch.zeros_like(out_shared)
        for m in range(self.num_modalities):
            mask = modality_ids == m
            if mask.any():
                out_specific[mask] = self.experts[m](x[mask])
        return out_shared + out_specific


class ModalityAwareMambaBlock(nn.Module):
    def __init__(
        self,
        dim=512,
        num_modalities=3,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        ffn_ratio=2.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.moe_in = ModalityExpertLinear(dim, num_modalities=num_modalities)
        self.mamba = build_mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.moe_out = ModalityExpertLinear(dim, num_modalities=num_modalities)

        self.norm2 = RMSNorm(dim)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.moe_in(h, modality_ids)
        h = self.mamba(h)
        h = self.moe_out(h, modality_ids)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SinkhornOTLoss(nn.Module):
    def __init__(self, eps=0.05, n_iters=30):
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x.float(), dim=-1)
        y = F.normalize(y.float(), dim=-1)
        B, Tx, D = x.shape
        Ty = y.shape[1]
        cost = 1.0 - torch.bmm(x, y.transpose(1, 2))
        cost = cost.clamp_min(0.0)
        log_K = -cost / self.eps
        log_a = cost.new_full((B, Tx), -math.log(Tx))
        log_b = cost.new_full((B, Ty), -math.log(Ty))
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        for _ in range(self.n_iters):
            u = log_a - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
            v = log_b - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)
        log_P = log_K + u.unsqueeze(2) + v.unsqueeze(1)
        P = torch.exp(log_P)
        return (P * cost).sum(dim=(1, 2)).mean()


class MMDLoss(nn.Module):
    def __init__(self, gamma: Optional[float] = None):
        super().__init__()
        self.gamma = gamma

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        D = x.shape[-1]
        gamma = self.gamma if self.gamma is not None else 1.0 / D
        dist = torch.cdist(x.float(), y.float(), p=2).pow(2)
        return torch.exp(-gamma * dist)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = self._kernel(x, x).mean(dim=(1, 2))
        k_yy = self._kernel(y, y).mean(dim=(1, 2))
        k_xy = self._kernel(x, y).mean(dim=(1, 2))
        return (k_xx + k_yy - 2.0 * k_xy).mean()


class BalancedGatedPooling(nn.Module):
    def __init__(self, dim=512, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, Ts: int, Tr: int):
        skel_tokens = x[:, :Ts, :]
        rgb_tokens = x[:, Ts:Ts + Tr, :]
        skel_pool = skel_tokens.mean(dim=1)
        rgb_pool = rgb_tokens.mean(dim=1)
        gate_input = torch.cat([
            skel_pool,
            rgb_pool,
            torch.abs(skel_pool - rgb_pool),
            skel_pool * rgb_pool,
        ], dim=-1)
        gate = self.gate(gate_input)
        fused = gate * skel_pool + (1.0 - gate) * rgb_pool
        fused = self.norm(fused)
        return fused, gate


class AlignMamba2Fusion(nn.Module):
    """
    Lightweight Skeleton + RGB fusion.

    Causal is a side branch:
        base_feature = pool(x)
        causal_feature = pool(causal(x))
        fusion_feature = base_feature + sigmoid(causal_mix) * (causal_feature - base_feature)
    """

    def __init__(
        self,
        skel_dim: int = 512,
        rgb_dim: int = 512,
        text_dim: int = 512,
        dim: int = 512,
        num_classes: Optional[int] = None,
        unimodal_depth: int = 1,
        fusion_depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        lambda_ot: float = 0.001,
        lambda_mmd: float = 0.01,
        use_text_in_fusion: bool = False,
        use_causal: bool = False,
        causal_mix_init: float = -6.0,
    ):
        super().__init__()
        self.dim = dim
        self.lambda_ot = lambda_ot
        self.lambda_mmd = lambda_mmd
        self.use_text_in_fusion = use_text_in_fusion
        self.use_causal = use_causal

        self.skel_proj = nn.Linear(skel_dim, dim)
        self.rgb_proj = nn.Linear(rgb_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)

        self.fusion_layers = nn.ModuleList([
            ModalityAwareMambaBlock(
                dim=dim,
                num_modalities=3,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(fusion_depth)
        ])
        self.out_norm = nn.LayerNorm(dim)

        self.cls_head = nn.Linear(dim, num_classes) if num_classes is not None else None
        self.ot_loss = SinkhornOTLoss(eps=0.05, n_iters=30)
        self.mmd_loss = MMDLoss(gamma=None)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.balanced_pool = BalancedGatedPooling(dim=dim, dropout=dropout)

        if self.use_causal:
            self.causal = EnhancedCausalIntervention(
                dim=dim,
                hidden_dim=dim * 2,
                num_global_confounders=16,
                num_modal_confounders=8,
                num_modalities=3,
                dropout=dropout,
                init_alpha=0.0,
                init_conf_scale=0.0,
            )
        else:
            self.causal = None

        # Will be missing when loading old checkpoints; use strict=False.
        self.causal_mix = nn.Parameter(torch.tensor(float(causal_mix_init)))

    def _prepare_text_sequence(
        self,
        text_feat: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if text_feat is None:
            return None

        if text_feat.dim() == 2:
            if text_feat.size(0) == batch_size:
                return text_feat.unsqueeze(1)
            if labels is None:
                return None
            labels = labels.long().to(text_feat.device)
            return text_feat[labels].unsqueeze(1)

        if text_feat.dim() == 3:
            if text_feat.size(0) == batch_size:
                return text_feat
            if labels is None:
                return None
            labels = labels.long().to(text_feat.device)
            return text_feat[labels]

        raise ValueError(f"Invalid text_feat shape: {tuple(text_feat.shape)}")

    def encode_text_bank(self, text_bank: torch.Tensor) -> torch.Tensor:
        if text_bank.dim() == 2:
            t = text_bank.unsqueeze(1)
        elif text_bank.dim() == 3:
            t = text_bank
        else:
            raise ValueError(f"Invalid text_bank shape: {tuple(text_bank.shape)}")
        t = self.text_proj(t)
        t = t.mean(dim=1)
        return t

    def compute_similarity(self, fusion_feature: torch.Tensor, text_bank: torch.Tensor) -> torch.Tensor:
        text_emb = self.encode_text_bank(text_bank)
        fusion_feature = F.normalize(fusion_feature, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        return logit_scale * fusion_feature @ text_emb.t()

    def _pool_tokens(self, x: torch.Tensor, Ts: int, Tr: int):
        if self.use_text_in_fusion:
            return x.mean(dim=1), None
        return self.balanced_pool(x, Ts=Ts, Tr=Tr)

    def forward(
        self,
        skeleton_feat: torch.Tensor,
        rgb_feat: torch.Tensor,
        text_feat: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_align_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        B, Ts, _ = skeleton_feat.shape
        _, Tr, _ = rgb_feat.shape
        device = skeleton_feat.device

        skel = self.skel_proj(skeleton_feat)
        rgb = self.rgb_proj(rgb_feat)

        text_seq = self._prepare_text_sequence(text_feat, labels, B)
        if text_seq is not None:
            text = self.text_proj(text_seq.to(device))
        else:
            text = None

        if self.training and return_align_loss and text is not None:
            loss_ot = self.ot_loss(skel, text) + self.ot_loss(rgb, text)
            loss_mmd = self.mmd_loss(skel, text) + self.mmd_loss(rgb, text)
            loss_align = self.lambda_ot * loss_ot + self.lambda_mmd * loss_mmd
        else:
            loss_ot = torch.zeros([], device=device)
            loss_mmd = torch.zeros([], device=device)
            loss_align = torch.zeros([], device=device)

        fusion_tokens = [skel, rgb]
        modality_ids = [
            torch.zeros(Ts, dtype=torch.long, device=device),
            torch.ones(Tr, dtype=torch.long, device=device),
        ]

        if self.use_text_in_fusion:
            if text is None:
                raise ValueError("use_text_in_fusion=True requires text_feat.")
            Tt = text.shape[1]
            fusion_tokens.append(text)
            modality_ids.append(torch.full((Tt,), 2, dtype=torch.long, device=device))

        x = torch.cat(fusion_tokens, dim=1)
        modality_ids = torch.cat(modality_ids, dim=0)

        for layer in self.fusion_layers:
            x = layer(x, modality_ids)
        x = self.out_norm(x)

        # Main path: keep the original strong feature.
        base_feature, gate = self._pool_tokens(x, Ts=Ts, Tr=Tr)

        causal_out = None
        causal_feature = None
        causal_gate = None
        if self.use_causal:
            causal_out = self.causal(x, modality_ids=modality_ids)
            x_causal = causal_out["feature"]
            causal_feature, causal_gate = self._pool_tokens(x_causal, Ts=Ts, Tr=Tr)
            mix = torch.sigmoid(self.causal_mix)
            fusion_feature = base_feature + mix * (causal_feature - base_feature)
        else:
            mix = torch.zeros([], device=device)
            fusion_feature = base_feature

        logits = self.cls_head(fusion_feature) if self.cls_head is not None else None

        ret = {
            "fusion_feature": fusion_feature,
            "base_feature": base_feature.detach(),
            "logits": logits,
            "loss_align": loss_align,
            "loss_ot": loss_ot,
            "loss_mmd": loss_mmd,
            "fusion_gate": gate,
            "causal_mix": mix.detach(),
        }

        if causal_out is not None:
            ret.update({
                "causal_feature_pooled": causal_feature,
                "causal_gate": causal_gate,
                "causal_token_feature": causal_out["feature"],
                "x_clean": causal_out["x_clean"],
                "causal_feature": causal_out["causal_feature"],
                "invariant_feature": causal_out["invariant_feature"],
                "counterfactual_feature": causal_out["counterfactual_feature"],
                "confounder_effect": causal_out["confounder_effect"],
                "confounder_gate": causal_out["confounder_gate"],
                "causal_delta": causal_out["delta"],
                "causal_alpha": causal_out["alpha"],
                "conf_scale": causal_out["conf_scale"],
            })

        return ret


if __name__ == "__main__":
    B, Ts, Tr, D = 4, 50, 8, 512
    C = 60
    skeleton = torch.randn(B, Ts, D).cuda()
    rgb = torch.randn(B, Tr, D).cuda()
    text_bank = torch.randn(C, D).cuda()
    labels = torch.randint(0, C, (B,)).cuda()
    model = AlignMamba2Fusion(use_causal=True).cuda()
    out = model(skeleton, rgb, text_bank, labels)
    print("fusion_feature:", out["fusion_feature"].shape)
    print("base_feature:", out["base_feature"].shape)
    print("causal_mix:", out["causal_mix"].item())
    print("logits:", model.compute_similarity(out["fusion_feature"], text_bank).shape)
