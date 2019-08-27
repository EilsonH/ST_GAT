import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGAT(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super(SpatialGAT, self).__init__()
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size = (in_channels, out_channels)))
        nn.init.xavier_uniform_(self.W.data, gain = 1.414)
        self.alpha = nn.Parameter(torch.zeros(size=(out_channels // 4, out_channels)))
        nn.init.xavier_uniform_(self.alpha.data, gain=1.414)
        self.phi = nn.Parameter(torch.zeros(size=(out_channels // 4, out_channels)))
        nn.init.xavier_uniform_(self.phi.data, gain=1.414)

    def forward(self, x, A):

        N, _, T, V = x.size()
        x = x.permute(0,2,3,1).contiguous().view(N * T, V, -1) # the size -1 is inferred from other dimensions
        x = torch.matmul(x, self.W)
        A = torch.sum(A, 0)

        x1 = torch.einsum('hc, nvc -> nvh', (self.alpha, x))
        x2 = torch.einsum('hc, nvc -> nvh', (self.phi, x)).transpose(1, 2)

        attention = torch.bmm(x1, x2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = F.softmax(attention, dim = 2) + A

        x = torch.bmm(attention, x)
        return x.view(N, T, V, -1).permute(0,3,1,2)

class TemporalGAT(nn.Module):

    def __init__(self, in_channels, kernel, stride, dropout = 0.5):
        super(TemporalGAT, self).__init__()
        self.dropout = dropout
        self.kernel = kernel
        self.stride = stride

        self.W = nn.Parameter(torch.zeros(size=(in_channels, in_channels)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.k = torch.tensor([1 / self.kernel for i in range(kernel)], requires_grad = True)
        self.alpha = nn.Parameter(torch.zeros(size = (in_channels, in_channels // 4)))
        self.phi = nn.Parameter(torch.zeros(size=(in_channels, in_channels // 4)))

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        x = torch.matmul(x, self.W)
        W = self.att_W(T)
        I = self.att_I(T)

        x1 = torch.matmul(x, self.alpha)
        x2 = torch.matmul(x, self.phi).permute(0, 1, 3, 2)

        attention = F.dropout(torch.matmul(x1, x2) / x2.size(-1), self.dropout, training = self.training)
        attention = F.softmax(torch.matmul(W, attention), dim = -1) + torch.matmul(W, I)

        x = torch.matmul(attention, x)
        return x.permute(0, 3, 2, 1)

    def att_I(self, t):
        out = torch.zeros(t, t)
        j = 0

        for i in range(t):
            e1, e2 = j - self.kernel // 2, j + self.kernel // 2 + 1
            out[i][max(0, e1): min(t, e2)] = self.k[max(0, -e1) : min(self.kernel, self.kernel // 2 + t - j)]
            j += 1

        return out.cuda()

    def att_W(self, t):
        out = torch.zeros((math.ceil(t / self.stride)), t)
        j = 0

        for i in range(out.size()[0]):
            out[i][j] = 1
            j += self.stride

        return out.cuda()
