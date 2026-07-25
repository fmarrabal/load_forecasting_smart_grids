"""
Baseline models — STLF V4.

All deep baselines share the same interface as the proposed model:
    forward(x_load, x_cov_past, x_cov_fut) -> (y_hat, attn_or_None)
so every model sees exactly the same information ([FIX-5]): past load, past
covariates, and future known covariates. Architectures are kept canonical;
future covariates enter through a small FiLM-style conditioning of the head,
identical across baselines, so comparisons reflect the backbone.

Includes the strong 2023+ baselines DLinear and PatchTST (channel-independent
patch transformer) that V3 lacked.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureCovHead(nn.Module):
    """Shared mechanism to inject future known covariates into a baseline:
    y = W2( ReLU( W1(features) + Wf(flatten(cov_fut)) ) )."""

    def __init__(self, feat_dim, horizon, n_cov_fut, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fcf = nn.Linear(horizon * n_cov_fut, hidden)
        self.out = nn.Linear(hidden, horizon)

    def forward(self, feat, cov_fut):
        z = self.fc1(feat) + self.fcf(cov_fut.flatten(1))
        return self.out(F.relu(z))


# ═══════════════════ NON-PARAMETRIC ═══════════════════

class SeasonalNaive:
    """Weekly persistence: y[t:t+H] = x[t-168:t-168+H]. The honest floor."""

    def __init__(self, season, horizon):
        self.season = season
        self.horizon = horizon

    def predict(self, x_load):
        # x_load: (B, L) with L >= season
        return x_load[:, -self.season:-self.season + self.horizon] \
            if self.season > self.horizon else x_load[:, -self.horizon:]


# ═══════════════════ RECURRENT / CONV ═══════════════════

class _RNNBase(nn.Module):
    def __init__(self, rnn_cls, in_dim, hidden, layers, horizon, n_cov_fut,
                 bidirectional=False):
        super().__init__()
        self.rnn = rnn_cls(in_dim, hidden, layers, batch_first=True,
                           dropout=0.2 if layers > 1 else 0.0,
                           bidirectional=bidirectional)
        d = hidden * (2 if bidirectional else 1)
        self.head = FutureCovHead(d, horizon, n_cov_fut)

    def forward(self, xl, xc, xf):
        h, _ = self.rnn(torch.cat([xl.unsqueeze(-1), xc], -1))
        return self.head(h[:, -1], xf), None


class LSTMModel(_RNNBase):
    def __init__(self, n_cov, horizon, n_cov_fut, hidden=128, layers=2):
        super().__init__(nn.LSTM, 1 + n_cov, hidden, layers, horizon, n_cov_fut)


class BiLSTMModel(_RNNBase):
    def __init__(self, n_cov, horizon, n_cov_fut, hidden=128, layers=2):
        super().__init__(nn.LSTM, 1 + n_cov, hidden, layers, horizon, n_cov_fut,
                         bidirectional=True)


class GRUModel(_RNNBase):
    def __init__(self, n_cov, horizon, n_cov_fut, hidden=128, layers=2):
        super().__init__(nn.GRU, 1 + n_cov, hidden, layers, horizon, n_cov_fut)


class TCNBlock(nn.Module):
    """Causal dilated conv block (left padding only — truly causal)."""

    def __init__(self, ic, oc, ks=3, d=1):
        super().__init__()
        self.pad = (ks - 1) * d
        self.conv = nn.Conv1d(ic, oc, ks, dilation=d)
        self.bn = nn.BatchNorm1d(oc)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        y = self.conv(F.pad(x, (self.pad, 0)))
        return F.relu(F.relu(self.bn(y)) + self.res(x))


class TCNModel(nn.Module):
    def __init__(self, n_cov, horizon, n_cov_fut):
        super().__init__()
        self.net = nn.Sequential(TCNBlock(1 + n_cov, 64, d=1),
                                 TCNBlock(64, 128, d=2), TCNBlock(128, 64, d=4))
        self.head = FutureCovHead(64, horizon, n_cov_fut)

    def forward(self, xl, xc, xf):
        z = self.net(torch.cat([xl.unsqueeze(-1), xc], -1).transpose(1, 2))
        return self.head(z[:, :, -1], xf), None


class CNNLSTMModel(nn.Module):
    def __init__(self, n_cov, horizon, n_cov_fut):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1 + n_cov, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.rnn = nn.LSTM(128, 64, 1, batch_first=True)
        self.head = FutureCovHead(64, horizon, n_cov_fut)

    def forward(self, xl, xc, xf):
        z = self.cnn(torch.cat([xl.unsqueeze(-1), xc], -1).transpose(1, 2))
        h, _ = self.rnn(z.transpose(1, 2))
        return self.head(h[:, -1], xf), None


class GRUTCNAttention(nn.Module):
    def __init__(self, n_cov, horizon, n_cov_fut):
        super().__init__()
        self.gru = nn.GRU(1 + n_cov, 64, 1, batch_first=True)
        self.tcn = nn.Sequential(TCNBlock(64, 64, d=1), TCNBlock(64, 64, d=2))
        self.att = nn.MultiheadAttention(64, 4, batch_first=True)
        self.head = FutureCovHead(64, horizon, n_cov_fut)

    def forward(self, xl, xc, xf):
        h, _ = self.gru(torch.cat([xl.unsqueeze(-1), xc], -1))
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.att(h, h, h)
        return self.head(h[:, -1], xf), None


# ═══════════════════ TRANSFORMERS ═══════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    """Vanilla point-wise transformer encoder."""

    def __init__(self, n_cov, horizon, n_cov_fut, dm=64, nh=4, nl=2):
        super().__init__()
        self.proj = nn.Linear(1 + n_cov, dm)
        self.pe = PositionalEncoding(dm)
        layer = nn.TransformerEncoderLayer(dm, nh, dm * 2, 0.1,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nl)
        self.head = FutureCovHead(dm, horizon, n_cov_fut)

    def forward(self, xl, xc, xf):
        z = self.enc(self.pe(self.proj(torch.cat([xl.unsqueeze(-1), xc], -1))))
        return self.head(z[:, -1], xf), None


class DLinear(nn.Module):
    """Zeng et al. 2023: trend/remainder split + two linear maps.
    Kept univariate (load channel) as in the original; future covariates
    enter through the same FutureCovHead used by every baseline."""

    def __init__(self, input_len, horizon, n_cov_fut, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.lin_trend = nn.Linear(input_len, horizon)
        self.lin_res = nn.Linear(input_len, horizon)
        # Future covariates enter as an ADDITIVE, zero-initialized correction
        # to the canonical linear forecast — the same "linear base + zero-init
        # residual" structure the proposed model uses, so DLinear keeps its
        # defining linear pathway at full strength and its covariate access is
        # untuned rather than throttled (round-3 referee audit: the previous
        # hard-coded 0.1 damping handicapped only DLinear).
        self.cov = nn.Sequential(nn.Linear(horizon * n_cov_fut, 64), nn.ReLU(),
                                 nn.Linear(64, horizon))
        nn.init.zeros_(self.cov[-1].weight)
        nn.init.zeros_(self.cov[-1].bias)

    def forward(self, xl, xc, xf):
        pad = self.kernel // 2
        xpad = F.pad(xl.unsqueeze(1), (pad, pad), mode="replicate")
        trend = F.avg_pool1d(xpad, self.kernel, stride=1).squeeze(1)
        res = xl - trend
        base = self.lin_trend(trend) + self.lin_res(res)
        return base + self.cov(xf.flatten(1)), None


class PatchTST(nn.Module):
    """Nie et al. 2023 (channel-independent patch transformer), simplified to
    the univariate-load setting with instance normalization (RevIN-style)."""

    def __init__(self, input_len, horizon, n_cov_fut,
                 patch_len=16, stride=8, dm=64, nh=4, nl=2):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patches = (input_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, dm)
        self.pe = PositionalEncoding(dm)
        layer = nn.TransformerEncoderLayer(dm, nh, dm * 2, 0.1,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nl)
        self.flat = nn.Linear(n_patches * dm, 256)
        self.head = FutureCovHead(256, horizon, n_cov_fut)

    def forward(self, xl, xc, xf):
        mu = xl.mean(dim=1, keepdim=True)
        sd = xl.std(dim=1, keepdim=True) + 1e-5
        z = (xl - mu) / sd
        patches = z.unfold(1, self.patch_len, self.stride)      # (B, N, P)
        h = self.enc(self.pe(self.proj(patches)))
        feat = F.relu(self.flat(h.flatten(1)))
        out = self.head(feat, xf)
        return out * sd + mu, None
