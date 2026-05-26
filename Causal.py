import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalIntervention(nn.Module):

    def __init__(
        self,
        dim=512,
        hidden_dim=512,
        num_confounders=32,
        dropout=0.1,
    ):
        super().__init__()
        self.dim = dim
        # ------------------------------------------------
        # Learnable Confounder Dictionary
        # ------------------------------------------------
        self.confounders = nn.Parameter(
            torch.randn(num_confounders, dim)
        )
        nn.init.normal_(self.confounders, std=0.02)

        # ------------------------------------------------
        # Backdoor Attention
        # ------------------------------------------------
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # ------------------------------------------------
        # Frontdoor Mediator
        # ------------------------------------------------
        self.mediator = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

        # ------------------------------------------------
        # Invariant Projection
        # ------------------------------------------------
        self.invariant_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------
        # Counterfactual Branch
        # ------------------------------------------------
        self.cf_generator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # ------------------------------------------------
        # Fusion Gate
        # ------------------------------------------------
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # ------------------------------------------------
        # norms
        # ------------------------------------------------
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def backdoor_adjustment(self, x):

        """
        Remove confounder effect
        """

        B, T, D = x.shape
        
        q = self.q_proj(x)
        conf = self.confounders.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(conf)
        v = self.v_proj(conf)

        attn = torch.matmul(
            q,
            k.transpose(-1, -2)
        ) / math.sqrt(D)

        attn = F.softmax(attn, dim=-1)

        conf_effect = torch.matmul(attn, v)

        # remove confounder
        x_clean = x#x_clean = x - conf_effect

        return x_clean, conf_effect

    def frontdoor_intervention(self, x_clean):

        """
        reconstruct mediator pathway
        """
        mediator = self.mediator(x_clean)
        causal_feature = x_clean + mediator

        return causal_feature

    def counterfactual_feature(self, x):

        noise = torch.randn_like(x) * 0.05
        cf_x = x + noise
        cf_x = self.cf_generator(cf_x)
        
        return cf_x

    def invariant_learning(self, x):
        return self.invariant_proj(x)

    def forward(self, x):
        # ----------------------------------------
        # backdoor removal
        # ----------------------------------------
        x_clean, conf_effect = self.backdoor_adjustment(x)
        
        # ----------------------------------------
        # frontdoor reconstruction
        # ----------------------------------------
        causal_feature = self.frontdoor_intervention(x_clean)

        # ----------------------------------------
        # invariant representation
        # ----------------------------------------
        invariant_feature = self.invariant_learning(
            causal_feature
        )

        # ----------------------------------------
        # counterfactual branch
        # ----------------------------------------
        cf_feature = self.counterfactual_feature(
            invariant_feature
        )

        # ----------------------------------------
        # gated fusion
        # ----------------------------------------
        gate = self.gate(
            torch.cat(
                [invariant_feature, x],
                dim=-1
            )
        )

        out = gate * invariant_feature + \
              (1 - gate) * x
        out = self.norm1(out)
        out = self.dropout(out)

        return {
            "feature": out,
            "causal_feature": causal_feature,
            "invariant_feature": invariant_feature,
            "counterfactual_feature": cf_feature,
            "confounder_effect": conf_effect,
        }