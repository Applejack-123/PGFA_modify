# module/skeleton_mamba_encoder.py
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba


def build_mamba(dim, d_state=16, d_conv=4, expand=2):
    if Mamba is None:
        raise ImportError(
            "没有找到 mamba_ssm。请先安装 mamba-ssm，或者确认你的环境里存在 "
            "`from mamba_ssm.modules.mamba_simple import Mamba`。"
        )
    try:
        return Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            )
    except TypeError:
        try:
            return Mamba(
                dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                )
        except TypeError:
            return Mamba(
                dim,
                d_state=d_state,
                expand=expand,
                )


# =========================================================
# 2. NTU RGB+D 25-joint graph
# =========================================================
def edge2mat(link, num_node):
    A = torch.zeros(num_node, num_node)
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    """
    D^{-1}A 形式的有向图归一化。
    """
    Dl = A.sum(dim=0)
    Dn = torch.zeros_like(A)

    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i].pow(-1)

    AD = A @ Dn
    return AD


class NTUGraph:
    """
    NTU RGB+D 25 个关节点的骨架图。
    输入默认关节点顺序必须是 NTU 25-joint 顺序。
    """
    def __init__(self, num_node=25):
        self.num_node = num_node

        self.self_link = [(i, i) for i in range(num_node)]

        # NTU RGB+D 25 joints, 1-based index
        inward_ori_index = [
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
            (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
            (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
            (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
            (22, 23), (23, 8), (24, 25), (25, 12)
        ]

        # 转成 0-based index
        self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        self.outward = [(j, i) for (i, j) in self.inward]

        I = edge2mat(self.self_link, num_node)
        In = normalize_digraph(edge2mat(self.inward, num_node))
        Out = normalize_digraph(edge2mat(self.outward, num_node))

        # A: [3, 25, 25]
        # 3 个子图: self / inward / outward
        self.A = torch.stack([I, In, Out], dim=0)



class UnitGCN(nn.Module):
    """
    空间图卷积。
    输入:  [N, C, T, V]
    输出:  [N, C_out, T, V]
    """

    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super().__init__()

        if not torch.is_tensor(A):
            A = torch.tensor(A, dtype=torch.float32)

        self.register_buffer("A_base", A.float())

        self.num_subset = A.shape[0]
        self.adaptive = adaptive

        if adaptive:
            self.PA = nn.Parameter(torch.zeros_like(self.A_base))
        else:
            self.register_parameter("PA", None)

        self.edge_importance = nn.Parameter(torch.ones_like(self.A_base))

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.num_subset,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out")
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        """
        x: [N, C, T, V]
        """
        N, C, T, V = x.shape

        A = self.A_base

        if self.adaptive:
            A = A + self.PA

        A = A * self.edge_importance

        # [N, C_out * K, T, V]
        x = self.conv(x)

        # [N, K, C_out, T, V]
        x = x.view(N, self.num_subset, -1, T, V)

        # 图卷积聚合
        # [N, K, C_out, T, V] x [K, V, V] -> [N, C_out, T, V]
        x = torch.einsum("nkctv,kvw->nctw", x, A)

        x = self.bn(x)
        x = self.relu(x)

        return x


class UnitTCN(nn.Module):
    """
    时间卷积，保持 T 不变。
    输入:  [N, C, T, V]
    输出:  [N, C, T, V]
    """

    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout, inplace=True),
        )

        nn.init.kaiming_normal_(self.net[0].weight, mode="fan_out")
        nn.init.constant_(self.net[1].weight, 1.0)
        nn.init.constant_(self.net[1].bias, 0.0)

    def forward(self, x):
        return self.net(x)


class STGCNBlock(nn.Module):
    """
    Spatial GCN + Temporal Conv block。
    这里不做 temporal stride，所以 T=50 会一直保留。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        dropout=0.1,
        adaptive=True,
    ):
        super().__init__()

        self.gcn = UnitGCN(
            in_channels=in_channels,
            out_channels=out_channels,
            A=A,
            adaptive=adaptive,
        )

        self.tcn = UnitTCN(
            channels=out_channels,
            kernel_size=3,
            dropout=dropout,
        )

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


# =========================================================
# 4. Temporal Mamba Encoder
# =========================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        ) * self.weight


class TemporalMambaBlock(nn.Module):
    """
    输入:  [B, T, D]
    输出:  [B, T, D]
    """

    def __init__(
        self,
        dim=512,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.mamba = build_mamba(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(dim)

        hidden_dim = dim * 4

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        x = x + self.drop1(self.mamba(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


# =========================================================
# 5. Full Skeleton Encoder
# =========================================================

class SkeletonMambaEncoder(nn.Module):
    """
    推荐使用的完整骨架编码器。

    输入:
        x: [B, 3, T, 25, 2]

    输出:
        feat: [B, T, 512]

    例如:
        x.shape    = [128, 3, 50, 25, 2]
        feat.shape = [128, 50, 512]
    """
    def __init__(
        self,
        in_channels=3,
        num_point=25,
        num_person=2,
        embed_dim=512,
        dropout=0.1,
        adaptive_graph=True,
        temporal_depth=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        person_pool="mean",
    ):
        super().__init__()

        assert person_pool in ["mean", "max"]

        self.in_channels = in_channels
        self.num_point = num_point
        self.num_person = num_person
        self.embed_dim = embed_dim
        self.person_pool = person_pool

        graph = NTUGraph(num_node=num_point)
        A = graph.A

        # 对原始骨架数据做 BN
        # 输入会被整理成 [B, M*V*C, T]
        self.data_bn = nn.BatchNorm1d(num_person * num_point * in_channels)

        # 不做 stride，保证 T 不变
        channels = [in_channels, 64, 128, 256, embed_dim]

        self.st_gcn = nn.Sequential(
            STGCNBlock(
                channels[0],
                channels[1],
                A,
                dropout=dropout,
                adaptive=adaptive_graph,
            ),
            STGCNBlock(
                channels[1],
                channels[2],
                A,
                dropout=dropout,
                adaptive=adaptive_graph,
            ),
            STGCNBlock(
                channels[2],
                channels[3],
                A,
                dropout=dropout,
                adaptive=adaptive_graph,
            ),
            STGCNBlock(
                channels[3],
                channels[4],
                A,
                dropout=dropout,
                adaptive=adaptive_graph,
            ),
        )

        self.temporal_encoder = nn.Sequential(
            *[
                TemporalMambaBlock(
                    dim=embed_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    dropout=dropout,
                )
                for _ in range(temporal_depth)
            ]
        )

        self.out_norm = RMSNorm(embed_dim)

    def forward(self, x):
        """
        x: [B, C, T, V, M]
        """
        if x.dim() != 5:
            raise ValueError(
                f"输入必须是 [B, C, T, V, M]，但当前 x.shape={x.shape}"
            )

        B, C, T, V, M = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"输入通道数错误，期望 C={self.in_channels}，但得到 C={C}"
            )

        if V != self.num_point:
            raise ValueError(
                f"关节点数错误，期望 V={self.num_point}，但得到 V={V}"
            )

        if M != self.num_person:
            raise ValueError(
                f"人数维度错误，期望 M={self.num_person}，但得到 M={M}"
            )

        # -------------------------------------------------
        # 原始输入:
        # x: [B, C, T, V, M]
        #
        # ST-GCN 常用整理方式:
        # [B, C, T, V, M]
        # -> [B, M, V, C, T]
        # -> [B, M*V*C, T]
        # -> BatchNorm1d
        # -> [B*M, C, T, V]
        # -------------------------------------------------

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(B, M * V * C, T)

        x = self.data_bn(x)

        x = x.view(B, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B * M, C, T, V)

        # -------------------------------------------------
        # Spatial GCN
        # x: [B*M, C, T, V]
        # -> [B*M, 512, T, V]
        # -------------------------------------------------

        x = self.st_gcn(x)

        # -------------------------------------------------
        # 关节点池化
        # [B*M, 512, T, V] -> [B*M, 512, T]
        # -------------------------------------------------

        x = x.mean(dim=-1)

        # -------------------------------------------------
        # 恢复人物维度
        # [B*M, 512, T] -> [B, M, 512, T]
        # -------------------------------------------------

        x = x.view(B, M, self.embed_dim, T)

        # -------------------------------------------------
        # 多人池化
        # [B, M, 512, T] -> [B, 512, T]
        # -------------------------------------------------

        if self.person_pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x.max(dim=1).values

        # -------------------------------------------------
        # 转成 Mamba 需要的序列格式
        # [B, 512, T] -> [B, T, 512]
        # -------------------------------------------------

        x = x.permute(0, 2, 1).contiguous()

        # -------------------------------------------------
        # Temporal Mamba
        # [B, T, 512] -> [B, T, 512]
        # -------------------------------------------------

        x = self.temporal_encoder(x)
        x = self.out_norm(x)

        return x


# =========================================================
# 6. Test
# =========================================================

if __name__ == "__main__":
    model = SkeletonMambaEncoder(
        in_channels=3,
        num_point=25,
        num_person=2,
        embed_dim=512,
        temporal_depth=2,
        dropout=0.1,
    )

    x = torch.randn(128, 3, 50, 25, 2)

    y = model(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)