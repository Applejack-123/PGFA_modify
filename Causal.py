# Causal.py
# -*- coding: utf-8 -*-
import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedCausalIntervention(nn.Module):
    """
    Token-level causal intervention for multimodal fusion tokens.

    Input:
        x: [B, T, D]
        modality_ids: [T] or [B, T]
            0 -> skeleton
            1 -> rgb
            2 -> text

    Design:
        1. modality-aware confounder dictionary
        2. gated backdoor adjustment
        3. temporal mediator over T
        4. identity-safe residual output
    """

    def __init__(
        self,
        dim: int = 512,
        hidden_dim: int = 1024,
        num_global_confounders: int = 16,
        num_modal_confounders: int = 8,
        num_modalities: int = 3,
        temporal_kernel: int = 3,
        dropout: float = 0.1,
        init_alpha: float = 0.0,
        init_conf_scale: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities

        self.global_confounders = nn.Parameter(torch.randn(num_global_confounders, dim))
        self.modality_confounders = nn.Parameter(
            torch.randn(num_modalities, num_modal_confounders, dim)
        )
        nn.init.normal_(self.global_confounders, std=0.02)
        nn.init.normal_(self.modality_confounders, std=0.02)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.conf_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.conf_scale = nn.Parameter(torch.tensor(float(init_conf_scale)))

        padding = temporal_kernel // 2
        self.dw_temporal_conv = nn.Conv1d(
            dim, dim, kernel_size=temporal_kernel, padding=padding, groups=dim
        )
        self.pw_temporal_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.temporal_norm = nn.LayerNorm(dim)

        self.mediator = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.invariant_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.delta_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.dropout = nn.Dropout(dropout)

    def _expand_modality_ids(
        self,
        modality_ids: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if modality_ids is None:
            return None
        modality_ids = modality_ids.to(device=device, dtype=torch.long)
        if modality_ids.dim() == 1:
            modality_ids = modality_ids.unsqueeze(0).expand(batch_size, seq_len)
        if modality_ids.shape != (batch_size, seq_len):
            raise ValueError(
                f"modality_ids should be [T] or [B,T], got {tuple(modality_ids.shape)}."
            )
        return modality_ids

    def _global_confounder_attention(self, q: torch.Tensor) -> torch.Tensor:
        B, T, D = q.shape
        k = self.k_proj(self.global_confounders)  # [Kg, D]
        v = self.v_proj(self.global_confounders)  # [Kg, D]
        attn = torch.matmul(q, k.t()) / math.sqrt(D)  # [B, T, Kg]
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)  # [B, T, D]

    def _modality_aware_confounder_attention(
        self,
        q: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = q.shape
        q_flat = q.reshape(B * T, D)
        ids_flat = modality_ids.reshape(B * T)
        conf_effect_flat = torch.zeros_like(q_flat)

        for m in range(self.num_modalities):
            mask = ids_flat == m
            if not mask.any():
                continue
            bank = torch.cat(
                [self.global_confounders, self.modality_confounders[m]],
                dim=0,
            )
            k = self.k_proj(bank)
            v = self.v_proj(bank)
            q_m = q_flat[mask]
            attn = torch.matmul(q_m, k.t()) / math.sqrt(D)
            attn = F.softmax(attn, dim=-1)
            conf_effect_flat[mask] = torch.matmul(attn, v)

        return conf_effect_flat.reshape(B, T, D)

    def backdoor_adjustment(
        self,
        x: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        modality_ids = self._expand_modality_ids(modality_ids, B, T, x.device)
        q = self.q_proj(x)

        if modality_ids is None:
            conf_effect = self._global_confounder_attention(q)
        else:
            conf_effect = self._modality_aware_confounder_attention(q, modality_ids)

        gate_input = torch.cat([x, conf_effect, x - conf_effect], dim=-1)
        conf_gate = self.conf_gate(gate_input)
        conf_scale = torch.tanh(self.conf_scale)
        x_clean = x - conf_scale * conf_gate * conf_effect
        return x_clean, conf_effect, conf_gate, conf_scale

    def temporal_mediator(self, x_clean: torch.Tensor) -> torch.Tensor:
        h = x_clean.transpose(1, 2)  # [B, D, T]
        h = self.dw_temporal_conv(h)
        h = F.gelu(h)
        h = self.pw_temporal_conv(h)
        h = h.transpose(1, 2)  # [B, T, D]
        h = self.temporal_norm(x_clean + self.dropout(h))
        mediator = self.mediator(h)
        return x_clean + mediator

    def invariant_learning(self, causal_feature: torch.Tensor) -> torch.Tensor:
        return self.invariant_proj(causal_feature)

    def counterfactual_intervention(
        self,
        x: torch.Tensor,
        conf_effect: torch.Tensor,
        conf_gate: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x.shape
        if self.training and B > 1:
            perm = torch.randperm(B, device=x.device)
            cf_conf_effect = conf_effect[perm]
        else:
            cf_conf_effect = -conf_effect

        conf_scale = torch.tanh(self.conf_scale)
        cf_x_clean = x - conf_scale * conf_gate * cf_conf_effect
        cf_causal = self.temporal_mediator(cf_x_clean)
        cf_invariant = self.invariant_learning(cf_causal)
        return cf_invariant

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x_clean, conf_effect, conf_gate, conf_scale = self.backdoor_adjustment(
            x, modality_ids=modality_ids
        )
        causal_feature = self.temporal_mediator(x_clean)
        invariant_feature = self.invariant_learning(causal_feature)
        delta = self.delta_proj(invariant_feature)

        alpha = torch.tanh(self.alpha)
        out = x + alpha * self.dropout(delta)

        cf_feature = self.counterfactual_intervention(
            x=x,
            conf_effect=conf_effect.detach(),
            conf_gate=conf_gate.detach(),
        )

        return {
            "feature": out,
            "x_clean": x_clean,
            "causal_feature": causal_feature,
            "invariant_feature": invariant_feature,
            "counterfactual_feature": cf_feature,
            "confounder_effect": conf_effect,
            "confounder_gate": conf_gate,
            "delta": delta,
            "alpha": alpha.detach(),
            "conf_scale": conf_scale.detach(),
        }


def causal_regularization(
    causal_out: Dict[str, torch.Tensor],
    lambda_cf: float = 0.0,
    lambda_conf: float = 0.0,
    lambda_delta: float = 0.0,
):
    """
    Optional causal regularization. Initial recommendation: keep all lambdas 0.
    """
    device = causal_out["feature"].device
    loss = torch.zeros([], device=device)
    loss_dict = {}

    if lambda_cf > 0:
        inv = F.normalize(causal_out["invariant_feature"].mean(dim=1), dim=-1)
        cf = F.normalize(causal_out["counterfactual_feature"].mean(dim=1), dim=-1)
        cf_loss = 1.0 - F.cosine_similarity(inv, cf, dim=-1).mean()
        loss = loss + lambda_cf * cf_loss
        loss_dict["cf_loss"] = cf_loss.detach()

    if lambda_conf > 0:
        conf_loss = causal_out["confounder_effect"].pow(2).mean()
        loss = loss + lambda_conf * conf_loss
        loss_dict["conf_loss"] = conf_loss.detach()

    if lambda_delta > 0:
        delta_loss = causal_out["delta"].pow(2).mean()
        loss = loss + lambda_delta * delta_loss
        loss_dict["delta_loss"] = delta_loss.detach()

    loss_dict["causal_reg"] = loss.detach()
    return loss, loss_dict


CausalIntervention = EnhancedCausalIntervention
Causal = EnhancedCausalIntervention


if __name__ == "__main__":
    B, T, D = 4, 58, 512
    x = torch.randn(B, T, D).cuda()
    modality_ids = torch.cat([
        torch.zeros(50, dtype=torch.long),
        torch.ones(8, dtype=torch.long),
    ]).cuda()
    model = EnhancedCausalIntervention(dim=D).cuda()
    out = model(x, modality_ids=modality_ids)
    print("feature:", out["feature"].shape)
    print("alpha:", out["alpha"].item())
    print("conf_scale:", out["conf_scale"].item())
