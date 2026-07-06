import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.tgcn import ConvTemporalGraphical
from .utils.graph import Graph

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    支持两种输出：
    1. return_sequence=False: 原始输出 [N, D]
    2. return_sequence=True : 序列输出 [N, T, D]，用于 Mamba
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        hidden_dim,
        graph_args,
        edge_importance_weighting,
        return_sequence=True,
        keep_temporal=True,
        **kwargs
    ):
        super().__init__()

        self.return_sequence = return_sequence
        self.keep_temporal = keep_temporal

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # 如果 keep_temporal=True，则所有 temporal stride 都设为 1
        # 输入 T=50，输出仍然 T=50
        s_down = 1 if keep_temporal else 2

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),

            # 原来这里是 stride=2，会把 T 下采样
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, s_down, **kwargs),

            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),

            # 原来这里也是 stride=2，会再次把 T 下采样
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, s_down, **kwargs),

            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x, ignore_joint=[]):
        """
        Args:
            x: [N, C, T, V, M]

        Returns:
            if return_sequence=True:
                [N, T, D]
            else:
                [N, D]
        """

        # data normalization
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()    # [N, M, V, C, T]
        x = x.view(N * M, V * C, T)                  # [N*M, V*C, T]
        x = self.data_bn(x)

        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()    # [N, M, C, T, V]
        x = x.view(N * M, C, T, V)                   # [N*M, C, T, V]

        # 获取未被 mask 掉的节点序列
        all_joint = set(range(V))
        remain_joint = list(all_joint - set(ignore_joint))
        remain_joint = sorted(remain_joint)

        x = x[:, :, :, remain_joint]

        # ST-GCN backbone
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance, remain_joint)

        # 此时：
        # x: [N*M, hidden_dim, T_out, V_remain]
        # 如果 keep_temporal=True 且输入 T=50，则 T_out=50

        if self.return_sequence:
            # 只池化关节维 V，不池化时间维 T
            x = x.mean(dim=3)                        # [N*M, D, T]

            # 多人 M 维度平均
            x = x.view(N, M, -1, x.size(-1))         # [N, M, D, T]
            x = x.mean(dim=1)                        # [N, D, T]

            # 转成 Mamba 需要的格式 [B, T, D]
            x = x.permute(0, 2, 1).contiguous()      # [N, T, D]

            return x

        else:
            # 原始 ST-GCN 输出方式：[N, D]
            x = F.avg_pool2d(x, x.size()[2:])        # [N*M, D, 1, 1]
            x = x.view(N, M, -1).mean(dim=1)         # [N, D]

            return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, remain_joint):
        
        A = A[:,remain_joint,:]
        A = A[:,:,remain_joint]
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A