# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module): #对图进行空域卷积

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements. #空洞卷积的参数
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 kernel_size, #这个kernel_size是空域的kernel_size，为1,2,3
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d( #2D卷积对channel做全覆盖
            in_channels,
            # 第二维的大小是由输出维度和空域卷积核大小共同决定的，这里是out_channels*3
            #由此也决定了参数数量：in_channels*outchannels*kernel_size，直观的理解就是kernel_size个权重矩阵

            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1), #卷积是在T和V维度上进行的，这里指时域和空域范围都1
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        # 拆成了5维的数据，n,k,c,t,v，其中k=3, c=out_channels,现在每个节点都对应由三个不同的权重矩阵卷积得到的特征矩阵
        #同一个节点的三个不同权重矩阵的数据，有的用于自身节点的计算，有的是该节点作为其他节点邻居节点时的权重矩阵
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        #这一步是对邻域特征的聚合
        #A中的三个矩阵分别代表自身，近重心节点，远重心节点
        x = torch.einsum('nkctv,kvw->nctw', (x, A)) #A的维度是kvw, v=w，在重叠的维度上求和，也就是k和v

        return x.contiguous(), A
