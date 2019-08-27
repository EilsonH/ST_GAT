import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_LSTM(nn.Module):
    def __init__(self, nodes, in_channel, out_channel,
                forget_bias = 1.0, ln = True):
        super(ST_LSTM, self).__init__()
        self.nodes = nodes

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.forget_bias = forget_bias
        self.layer_norm = ln

        self.conv_x = nn.Conv1d(in_channel, 7 * out_channel, 1)
        self.conv_h = nn.Conv1d(out_channel, 4 * out_channel, 1)
        self.conv_c = nn.Conv1d(out_channel, out_channel, 1)
        self.conv_m_in = nn.Conv1d(in_channel, 4 * out_channel, 1)
        self.conv_m_out = nn.Conv1d(out_channel, out_channel, 1)

        self.W_1x1 = nn.Conv1d(2 * out_channel, out_channel, 1)

        if self.layer_norm:
            self.ln_x = nn.LayerNorm([7 * out_channel, self.nodes])
            self.ln_h = nn.LayerNorm([4 * out_channel, self.nodes])
            self.ln_c = nn.LayerNorm([out_channel, self.nodes])
            self.ln_m_in = nn.LayerNorm([4 * out_channel, self.nodes])
            self.ln_m_out = nn.LayerNorm([out_channel, self.nodes])

    def forward(self, x, h, c, m):
        '''
        x shape:(N, C, V)
        '''

        batch = x.shape()[0]
        if h is None:
            h = torch.zeros(batch, self.out_channel, self.nodes)
        if c is None:
            c = torch.zeros(batch, self.out_channel, self.nodes)
        if m is None:
            m = torch.zeros(batch, self.out_channel, self.nodes)

        xs, hs, ms = self.conv_x(x), self.conv_h(h), self.conv_m_in(m)

        if self.layer_norm:
            xs, hs, ms = self.ln_x(xs), self.ln_h(hs), self.ln_m_in(ms)

        xg, xi, xf, xg_, xi_, xf_, xo = torch.chunk(xs, 7, dim = 1)
        hg, hi, hf, ho = torch.chunk(hs, 4, dim = 1)
        mg, mi, mf, mm = torch.chunk(ms, 4, dim = 1)

        g, i, f = torch.tanh(xg + hg), torch.sigmoid(xi + hi), torch.sigmoid(xf + hf + self.forget_bias)
        c_ = f * c + i * g
        g_, i_, f_ = torch.tanh(xg_ + mg), torch.sigmoid(xi_ + mi), torch.sigmoid(xf_ + mf + self.forget_bias)
        m_ = f_ * mm + i_ * g_
        co = self.conv_c(c_)
        mo = self.conv_m_out(m_)
        if self.layer_norm:
            co, mo = self.ln_c(co), self.ln_m_out(mo)

        o = torch.sigmoid(xo + ho + co + mo)
        cm = torch.concat([co, mo], dim = 1)
        temp = self.W_1x1(cm)
        h_ = o * torch.tanh(temp)

        return h_, c_, m_

class RNN(nn.Module):
    def __init__(self, nodes, out_c, num_layers = 4, hidden_channel = [128, 64, 64, 64]):
        super(RNN, self).__init__()

        self.hidden = []
        self.cell = []
        self.memory = None
        self.lstm = nn.ModuleList()
        self.num_layers = num_layers
        self.nodes = nodes

        assert num_layers == len(hidden_channel)
        for i in range(num_layers):
            lstm.append(ST_LSTM(self.nodes, hidden_channel[i - 1], hidden_channel[i]))
            cell.append(None)
            hidden.append(None)

        self.conv_back = nn.Sequential(
            nn.Conv2d(hidden_channel[-1], out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True))

    def forward(self, x):
        '''
        x shape: (N, C, T, V)
        '''

        batch, _, sequence = x.size()[:3]
        result = torch.zeros(batch, self.hidden[-1], sequence, self.nodes)
        for t in range(sequence):
            for (l, lstm) in enumerate(self.lstm):
                frame = x[:, :, t] if l == 0 else self.hidden[l - 1]
                self.hidden[l], self.cell[l], self.memory = lstm(frame, self.hidden[l], self.cell[l], self.memory)
            result[:, :, t] = self.hidden[-1]

        output = self.conv_back(result)
        return output
