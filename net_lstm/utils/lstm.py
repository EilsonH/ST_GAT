import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

class ST_LSTM(nn.Module):
    def __init__(self, nodes, in_channel, out_channel, out = None,
                forget_bias = 1.0, ln = True, first = False):
        super(ST_LSTM, self).__init__()
        self.nodes = nodes

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.forget_bias = forget_bias
        self.layer_norm = ln

        if first == False:
            out = in_channel
        self.conv_xg = nn.Conv1d(out, out_channel, 1)
        self.conv_xi = nn.Conv1d(out, out_channel, 1)
        self.conv_xf = nn.Conv1d(out, out_channel, 1)
        self.conv_xg_ = nn.Conv1d(out, out_channel, 1)
        self.conv_xi_ = nn.Conv1d(out, out_channel, 1)
        self.conv_xf_ = nn.Conv1d(out, out_channel, 1)
        self.conv_xo = nn.Conv1d(out, out_channel, 1)

        self.conv_hg = nn.Conv1d(out_channel, out_channel, 1)
        self.conv_hi = nn.Conv1d(out_channel, out_channel, 1)
        self.conv_hf = nn.Conv1d(out_channel, out_channel, 1)
        self.conv_ho = nn.Conv1d(out_channel, out_channel, 1)

        self.conv_c = nn.Conv1d(out_channel, out_channel, 1)

        self.conv_mg = nn.Conv1d(in_channel, out_channel, 1)
        self.conv_mi = nn.Conv1d(in_channel, out_channel, 1)
        self.conv_mf = nn.Conv1d(in_channel, out_channel, 1)
        self.conv_mm = nn.Conv1d(in_channel, out_channel, 1)

        self.conv_mo = nn.Conv1d(out_channel, out_channel, 1)

        self.W_1x1 = nn.Conv1d(2 * out_channel, out_channel, 1)

        if self.layer_norm:
            self.ln_xg = nn.LayerNorm([out_channel, self.nodes])
            self.ln_xi = nn.LayerNorm([out_channel, self.nodes])
            self.ln_xf = nn.LayerNorm([out_channel, self.nodes])
            self.ln_xg_ = nn.LayerNorm([out_channel, self.nodes])
            self.ln_xi_ = nn.LayerNorm([out_channel, self.nodes])
            self.ln_xf_ = nn.LayerNorm([out_channel, self.nodes])
            self.ln_xo = nn.LayerNorm([out_channel, self.nodes])

            self.ln_hg = nn.LayerNorm([out_channel, self.nodes])
            self.ln_hi = nn.LayerNorm([out_channel, self.nodes])
            self.ln_hf = nn.LayerNorm([out_channel, self.nodes])
            self.ln_ho = nn.LayerNorm([out_channel, self.nodes])

            self.ln_c = nn.LayerNorm([out_channel, self.nodes])

            self.ln_mg = nn.LayerNorm([out_channel, self.nodes])
            self.ln_mi = nn.LayerNorm([out_channel, self.nodes])
            self.ln_mf = nn.LayerNorm([out_channel, self.nodes])
            self.ln_mm = nn.LayerNorm([out_channel, self.nodes])

            self.ln_mo = nn.LayerNorm([out_channel, self.nodes])

    def forward(self, x, h, c, m):
        '''
        x shape:(N, C, V)
        '''

        xg, xi, xf, xg_, xi_, xf_, xo = self.conv_xg(x), self.conv_xi(x), self.conv_xf(x), self.conv_xg_(x), self.conv_xi_(x), self.conv_xf_(x), self.conv_xo(x)
        hg, hi, hf, ho = self.conv_hg(h.cuda()), self.conv_hi(h.cuda()), self.conv_hf(h.cuda()), self.conv_ho(h.cuda())
        mg, mi, mf, mm = self.conv_mg(m.cuda()), self.conv_mi(m.cuda()), self.conv_mf(m.cuda()), self.conv_mm(m.cuda())

        if self.layer_norm:
            xg, xi, xf, xg_, xi_, xf_, xo = self.ln_xg(xg), self.ln_xi(xi), self.ln_xf(xf), self.ln_xg_(
                xg_), self.ln_xi_(xi_), self.ln_xf_(xf_), self.ln_xo(xo)
            hg, hi, hf, ho = self.ln_hg(hg), self.ln_hi(hi), self.ln_hf(hf), self.ln_ho(ho)
            mg, mi, mf, mm = self.ln_mg(mg), self.ln_mi(mi), self.ln_mf(mf), self.ln_mm(mm)


        g, i, f = torch.tanh(xg + hg), torch.sigmoid(xi + hi), torch.sigmoid(xf + hf + self.forget_bias)
        c_ = f * c.cuda() + i * g
        g_, i_, f_ = torch.tanh(xg_ + mg), torch.sigmoid(xi_ + mi), torch.sigmoid(xf_ + mf + self.forget_bias)
        m_ = f_ * mm + i_ * g_
        co = self.conv_c(c_)
        mo = self.conv_mo(m_)
        if self.layer_norm:
            co = self.ln_c(co)
            mo = self.ln_mo(mo)

        o = torch.sigmoid(xo + ho + co + mo)
        cm = torch.cat([co, mo], dim = 1)
        temp = self.W_1x1(cm)
        h_ = o * torch.tanh(temp)

        return h_, c_, m_

class RNN(nn.Module):
    def __init__(self, nodes, out_c, num_layers = 4, hidden_channel = [128, 64, 64, 64]):
        super(RNN, self).__init__()

        self.lstm = nn.ModuleList()
        self.num_layers = num_layers
        self.nodes = nodes
        self.hidden_channel = hidden_channel

        assert num_layers == len(hidden_channel)
        for i in range(num_layers):
            if i == 0:
                self.lstm.append(ST_LSTM(self.nodes, hidden_channel[i - 1], hidden_channel[i], first=True, out=out_c))
            else:
                self.lstm.append(ST_LSTM(self.nodes, hidden_channel[i - 1], hidden_channel[i]))

        self.conv_back = nn.Sequential(
            nn.Conv2d(hidden_channel[-1], out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True))

    def forward(self, x):
        '''
        x shape: (N, C, T, V)
        '''

        N, _, T, V = x.size()
        hidden, cell, memory, result = self.initHidden(N, T, V)
        for t in range(T):
            for (l, lstm) in enumerate(self.lstm):
                frame = x[:, :, t] if l == 0 else hidden[l - 1]
                hidden[l], cell[l], memory = lstm(frame, hidden[l], cell[l], memory)
            result[:, :, t] = hidden[-1]

        output = self.conv_back(result)
        return output

    def initHidden(self, N, T, V):
        c, h = [], []
        m = torch.zeros(N, self.hidden_channel[-1], V)
        o = torch.zeros(N, self.hidden_channel[-1], T, V).cuda()
        for i in range(self.num_layers):
            c.append(torch.zeros(N, self.hidden_channel[i], V))
            h.append(torch.zeros(N, self.hidden_channel[i], V))

        return h, c, m, o

