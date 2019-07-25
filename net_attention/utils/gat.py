import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, dropout, alpha = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.alpha = alpha

        self.a = nn.Parameter(torch.zeros(size=(2*in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        N, C, T, V = x.size()
        x = x.permute(0,2,3,1)
        A = torch.sum(A, 0)

        x1 = x.repeat(1, 1, 1, V).view(N, T, V, V, C)
        x2 = x.repeat(1, 1, V, 1).view(N, T, V, V, C)
        a_input = torch.cat([x1, x2], dim = 4)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(4))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)
        attention = F.softmax(attention, dim=3)
        attention = F.dropout(attention, self.dropout, training=self.training)

        x = torch.bmm(attention.contiguous().view(N * T, V, V), x.contiguous().view(N * T, V, C))

        return x.view(N, T, V, C).permute(0,3,1,2)
