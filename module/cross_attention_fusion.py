# module/cross_attention_fusion.py
import torch
import torch.nn as nn
import math

class CrossAttentionFusion(nn.Module):
    """
    跨模态交叉注意力融合模块
    支持 3D 输入输出 (N, T, D)
    """
    def __init__(self, 
                 feature_dim=512,      # 输入特征维度（骨架和RGB都是512）
                 num_heads=8,          # 多头注意力的头数
                 dropout=0.1):         # dropout比率
        super(CrossAttentionFusion, self).__init__()
        
        self.feature_dim = feature_dim  # 512
        self.num_heads = num_heads      # 8
        self.head_dim = feature_dim // num_heads  # 512/8=64
        self.scale = math.sqrt(self.head_dim)  # 开方64=8
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Q、K、V投影层（用于交叉注意力）
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer Normalization
        self.norm_query = nn.LayerNorm(feature_dim)
        self.norm_key_value = nn.LayerNorm(feature_dim)
        self.norm_out = nn.LayerNorm(feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # 拼接后的投影层
        self.concat_proj = nn.Linear(feature_dim * 2, feature_dim)
    
    def forward(self, skeleton_feat, rgb_feat):
        """
        Args:
            skeleton_feat: [batch_size, seq_len, feature_dim] 骨架特征
            rgb_feat: [batch_size, seq_len, feature_dim] RGB特征
            
        Returns:
            fused_feat: [batch_size, seq_len, feature_dim] 融合后的特征
        """
        batch_size, seq_len, _ = skeleton_feat.shape
        
        # ========== 1. 单向交叉注意力 ==========
        # 已经是 [B, T, D]，不需要 unsqueeze
        skeleton_seq = skeleton_feat  # [B, T, D]
        rgb_seq = rgb_feat            # [B, T, D]
        
        # LayerNorm
        skeleton_norm = self.norm_query(skeleton_seq)
        rgb_norm = self.norm_key_value(rgb_seq)
        
        # 计算Q、K、V
        Q = self.query_proj(skeleton_norm)  # [B, T, D]
        K = self.key_proj(rgb_norm)         # [B, T, D]
        V = self.value_proj(rgb_norm)       # [B, T, D]
        
        # 分割成多头
        # [B, T, D] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, T, T]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_out = torch.matmul(attn_weights, V)  # [B, num_heads, T, head_dim]
        
        # 合并多头
        # [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        attn_out = self.out_proj(attn_out)
        
        # 残差连接
        attended_feat = skeleton_seq + self.dropout(attn_out)
        attended_feat = self.norm_out(attended_feat)  # [B, T, D]
        
        # ========== 2. 可学习权重拼接融合 ==========
        # 拼接原始骨架特征和注意力增强后的特征（沿特征维度）
        concat_feat = torch.cat([skeleton_seq, attended_feat], dim=-1)  # [B, T, 2*D]
        
        # 投影回原始维度
        concat_feat = self.concat_proj(concat_feat)  # [B, T, D]
        
        # 可学习权重融合（逐时间步）
        weight = torch.sigmoid(self.fusion_weight)
        fused_feat = weight * skeleton_seq + (1 - weight) * concat_feat  # [B, T, D]
        
        return fused_feat