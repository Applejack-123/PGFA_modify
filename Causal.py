import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
#from mamba_ssm import Mamba
#from models.Predict import MGCN_block

import torch
import torch.nn as nn
import numpy as np

class CausalIntervention(nn.Module):
    """
    适配 (N, D) 格式的因果干预模块
    保持因果干预的核心功能：混淆因子估计 + 环境变量调整
    """
    
    def __init__(self, feature_dim=512, state_size=64, device='cuda'):
        """
        Args:
            feature_dim: 输入特征维度（512）
            state_size: 隐状态维度
            device: 设备
        """
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.state_size = state_size
        
        # 可学习的混淆因子原型
        # 不再需要 num_nodes，改为可学习的混淆因子个数
        self.num_confounders = 8  # 混淆因子数量
        self.S = nn.Parameter(torch.randn(self.num_confounders, state_size) * 0.1)
        
        # 混淆因子估计网络（从特征估计混淆因子）
        self.confounder_net = nn.Sequential(
            nn.Linear(feature_dim, 2 * state_size),
            nn.GELU(),
            nn.Linear(2 * state_size, state_size)
        )
        
        # 环境变量编码网络（从特征编码环境）
        self.env_net = nn.Sequential(
            nn.Linear(feature_dim, 2 * state_size),
            nn.ReLU(),
            nn.Linear(2 * state_size, state_size),
            nn.Dropout(0.1)
        )
        
        # 门控网络（融合混淆因子和环境变量）
        self.gate_net = nn.Sequential(
            nn.Linear(2 * state_size, state_size),
            nn.Sigmoid()
        )
        
        # 投影回原始维度
        self.proj = nn.Linear(state_size, feature_dim)
        
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, adj=None):
        """
        因果干预前向传播
        
        Args:
            x: (N, D) 输入特征
            adj: 邻接矩阵 (N, N) 可选，用于样本间关系
            
        Returns:
            x_hat: (N, D) 干预后的特征
        """
        B, D = x.shape  # B: batch_size, D: feature_dim (512)
        
        # ========== 1. 估计混淆因子 ==========
        # 从输入特征估计混淆因子
        confounder = self.confounder_net(x)  # (B, state_size)
        
        # ========== 2. 估计环境变量 ==========
        # 从输入特征估计环境变量
        env = self.env_net(x)  # (B, state_size)
        
        # ========== 3. 使用可学习的混淆因子原型（可选） ==========
        # 计算特征与混淆因子原型的相似度
        # 归一化
        x_norm = F.normalize(x, dim=-1)
        S_norm = F.normalize(self.S, dim=-1)
        
        # 相似度矩阵 (B, num_confounders)
        sim = torch.matmul(x_norm, S_norm.t()) / self.temperature
        sim_weights = torch.softmax(sim, dim=-1)  # (B, num_confounders)
        
        # 加权混淆因子原型
        proto_confounder = torch.matmul(sim_weights, self.S)  # (B, state_size)
        
        # ========== 4. 融合环境变量和混淆因子 ==========
        # 如果提供了邻接矩阵，可以通过图传播环境变量
        if adj is not None:
            # adj: (B, B) 样本间相似度/关系
            adj_tensor = torch.tensor(adj.astype(np.float32)).to(self.device)
            # 环境变量通过图传播
            env_propagated = torch.matmul(adj_tensor, env)  # (B, state_size)
        else:
            env_propagated = env
        
        # 融合环境变量和混淆因子
        S_combined = torch.cat([env_propagated, proto_confounder], dim=-1)  # (B, 2*state_size)
        gate = self.gate_net(S_combined)  # (B, state_size)
        
        # 门控融合
        S_adjusted = env_propagated * gate + proto_confounder * (1 - gate)  # (B, state_size)
        
        # ========== 5. 计算调整项 ==========
        adjustment = self.proj(S_adjusted)  # (B, feature_dim)
        
        # ========== 6. 后门调整 ==========
        x_hat = x + x * adjustment
        
        return x_hat


class SpatioTemporalEncoder(nn.Module):
    """
    保留原结构，但 N=1（单节点）
    输入 (B, T, D)，输出 (B, T, d_model)
    """
    def __init__(self, hidden_dim, num_nodes, d_model, device):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.num_nodes = 1  # 固定为 1
        
        # 原代码保持不变
        self.mgcn = MGCN_block(device, in_channels=1, K=2, nb_chev_filter=64, nb_time_filter=64, time_strides=1, len_input=12)
        
        self.spatial_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(3)
        ])
        self.temporal_blocks = nn.ModuleList([
            Mamba(d_model=64)
            for _ in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        self.linear_1 = nn.Linear(hidden_dim, d_model)
        self.linear_2 = nn.Linear(d_model, hidden_dim)
        
        self.device = device
    
    def forward(self, x, adj=None):
        """
        Args:
            x: (B, T, D)  D=512
        Returns:
            output: (B, T, d_model)
        """
        B, T, D = x.shape
        N = 1  # 固定节点数
        
        # (B, T, D) -> (B, N, T, D)
        x = x.unsqueeze(1)  # (B, 1, T, D)
        
        # 输入投影
        x = self.linear_1(x)  # (B, 1, T, d_model)
        
        # 调整维度给 mgcn: (B, N, d_model, T)
        x = x.permute(0, 1, 3, 2)  # (B, 1, d_model, T)
        
        # 创建虚拟邻接矩阵
        adj = np.eye(N).astype(np.float32)
        adj_tensor = torch.from_numpy(adj).float().to(x.device)
        edge_index = adj_tensor.nonzero().t().contiguous()
        
        # MGCN 处理
        x = self.mgcn(x, edge_index)  # (B, 1, d_model, T)
        
        # 移除节点维度
        x = x.squeeze(1)  # (B, d_model, T)
        x = x.permute(0, 2, 1)  # (B, T, d_model)
        
        # 后续的 GCN + Mamba 层（N=1）
        for spatial_conv, temporal_block, norm in zip(self.spatial_convs, 
                                                      self.temporal_blocks, 
                                                      self.norms):
            residual = x
            
            # 添加节点维度
            x_with_node = x.unsqueeze(1)  # (B, 1, T, hidden_dim)
            
            # 空间卷积（N=1 退化为 MLP）
            x_spatial = x_with_node.permute(0, 2, 1, 3).reshape(B*T, N, self.hidden_dim)
            x_spatial = spatial_conv(x_spatial, edge_index)
            x_spatial = x_spatial.view(B, T, N, self.hidden_dim).permute(0, 2, 1, 3).squeeze(1)
            
            # 时序 Mamba
            x_temporal = temporal_block(x.reshape(B*N, T, self.hidden_dim)).reshape(B, N, T, self.hidden_dim).squeeze(1)
            
            x = x_spatial + x_temporal
            x = norm(x + residual)
        
        # 输出投影
        output = self.linear_2(x.unsqueeze(1)).squeeze(1)  # (B, T, d_model)
        
        return output

class CausalModule(nn.Module):
    def __init__(self, hidden_dim, num_nodes, his_len, pred_len, d_model, state_size, device):
        super().__init__()
        self.his_len = his_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.device = device
        self.linear = nn.Linear(hidden_dim, d_model)
        self.encoder = SpatioTemporalEncoder(hidden_dim, num_nodes, d_model, device)
        self.causal_layer = CausalIntervention(num_nodes, state_size, hidden_dim, device)
        self.decoder = nn.Linear(hidden_dim, d_model)
        
    def forward(self, fused_feat, t_feat, matrix):
        encoded = self.encoder(t_feat, matrix)
        corrected=self.decoder(encoded)
        pred_3=corrected.squeeze(-1)[..., :3]
        pred_6=corrected.squeeze(-1)[..., :6]
        pred_12=corrected.squeeze(-1)


        m_encoded = self.encoder(fused_feat, matrix) 
        m_corrected = self.causal_layer(m_encoded, matrix) 
        m_corrected=self.decoder(m_corrected)

        m_pred_3 = m_corrected.squeeze(-1)[..., :3]
        m_pred_6 = m_corrected.squeeze(-1)[..., :6]
        m_pred_12 = m_corrected.squeeze(-1)
        
        return pred_3, pred_6, pred_12, m_pred_3, m_pred_6, m_pred_12 