"""
Proposed model — STLF V4.

CPTB: Causal-decomposition Patch Transformer - BiGRU with cross-attention.

The V3 pipeline applied CEEMDAN-SE-VMD to the FULL series before splitting,
which leaks future information into training (EMD sifting and VMD are global
transforms). V4 replaces the offline decomposition with a CAUSAL, in-model,
end-to-end learnable analogue that preserves the dual-stage identity:

  Stage 1 (CEEMDAN analogue) — adaptive multi-scale trend/seasonal split via
    causal moving averages at half-day / day / week scales. Components sum to
    the input exactly (perfect reconstruction) and use only past samples.

  Stage 2 (SE+VMD analogue) — the high-frequency residual is refined by a
    LEARNABLE causal band-pass FIR filter bank (VMD is, at its core, adaptive
    band-pass filtering; a learnable FIR bank is its causal counterpart).
    A remainder channel keeps reconstruction exact.

  Adaptive routing (SE+K-Means analogue) — a differentiable complexity gate
    scores each component from causal statistics (energy, roughness) and
    modulates its embedding, replacing the non-differentiable Sample-Entropy
    K-Means routing of V3.

Downstream, per-component patch embeddings feed a Transformer encoder over the
(component x patch) token grid; cross-attention fuses PAST covariates and
FUTURE known covariates (calendar, temperature proxy); a BiGRU decodes the
patch sequence; RevIN handles non-stationarity ([FIX-2]).

forward(x_load, x_cov_past, x_cov_fut) -> (y_hat, cross_attention_weights)
  x_load     : (B, L)      z-scored past load
  x_cov_past : (B, L, C)   past covariates
  x_cov_fut  : (B, H, Cf)  future known covariates
  y_hat      : (B, H)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_baselines import PositionalEncoding


# ═══════════════════ NORMALIZATION ═══════════════════

class RevIN(nn.Module):
    """Reversible per-window instance normalization (Kim et al., 2022)."""

    def __init__(self, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))

    def normalize(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sd = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        z = (x - mu) / sd
        if self.affine:
            z = z * self.gamma + self.beta
        return z, mu, sd

    def denormalize(self, y, mu, sd):
        if self.affine:
            y = (y - self.beta) / (self.gamma + 1e-8)
        return y * sd + mu


# ═══════════════════ STAGE 1: CAUSAL MULTI-SCALE SPLIT ═══════════════════

def causal_moving_average(x, k):
    """MA_k[t] = mean(x[t-k+1 .. t]) with replicate left padding. x: (B, L)."""
    xp = F.pad(x.unsqueeze(1), (k - 1, 0), mode="replicate")
    return F.avg_pool1d(xp, kernel_size=k, stride=1).squeeze(1)


class Stage1Decomposition(nn.Module):
    """
    x = (x - MA_k1) + (MA_k1 - MA_k2) + (MA_k2 - MA_k3) + MA_k3
        \\_ HF _/     \\_ intra-day _/   \\__ weekly __/    \\ trend /
    Perfect reconstruction; strictly causal.
    """

    def __init__(self, kernels):
        super().__init__()
        self.kernels = sorted(kernels)   # e.g. [12, 24, 168] (in steps)

    def forward(self, x):
        mas = [causal_moving_average(x, k) for k in self.kernels]
        comps = [x - mas[0]]
        for i in range(len(mas) - 1):
            comps.append(mas[i] - mas[i + 1])
        comps.append(mas[-1])
        return comps                      # list of (B, L); HF first


# ═══════════════════ STAGE 2: LEARNABLE FIR FILTER BANK ═══════════════════

class Stage2FilterBank(nn.Module):
    """
    K learnable causal FIR band-pass filters applied to the HF component,
    initialized as windowed-sinc band-pass filters on K equal sub-bands of
    (0, pi). A remainder channel keeps the sum equal to the input.
    """

    def __init__(self, n_bands=4, kernel_size=25):
        super().__init__()
        self.K = n_bands
        self.ks = kernel_size
        w = torch.zeros(n_bands, 1, kernel_size)
        # Center the sinc mid-kernel so its main lobe aligns with the Hamming
        # peak (round-3 audit: centering on the last tap multiplied the main
        # lobe by hamming[-1]≈0.08). This yields a linear-phase causal
        # band-pass with (ks-1)/2 group delay; the left-padded conv stays
        # causal regardless of the tap layout.
        n = torch.arange(kernel_size, dtype=torch.float64) - (kernel_size - 1) / 2
        hamming = torch.hamming_window(kernel_size, periodic=False,
                                       dtype=torch.float64)
        for b in range(n_bands):
            f1 = 0.5 * b / n_bands          # normalized (Nyquist = 0.5)
            f2 = 0.5 * (b + 1) / n_bands

            def sinc_lp(fc):
                h = 2 * fc * torch.sinc(2 * fc * n)
                return h

            h = (sinc_lp(f2) - sinc_lp(f1)) if b > 0 else sinc_lp(f2)
            h = h * hamming
            w[b, 0, :] = h.float()
        self.weight = nn.Parameter(w)

    def forward(self, x):
        """x: (B, L) -> list of K bands + remainder, each (B, L)."""
        xp = F.pad(x.unsqueeze(1), (self.ks - 1, 0))          # causal
        bands = F.conv1d(xp, self.weight)                     # (B, K, L)
        remainder = x - bands.sum(dim=1)
        return [bands[:, k] for k in range(self.K)] + [remainder]


# ═══════════════════ ADAPTIVE COMPLEXITY GATE ═══════════════════

class ComplexityGate(nn.Module):
    """
    Differentiable analogue of the SE + K-Means routing: per component,
    compute causal statistics (log-energy, roughness = E|dc|/(std+eps),
    energy share) and map them to a (0, 2) multiplicative gate.
    """

    def __init__(self, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, comps):
        """comps: (B, M, L) -> gates (B, M, 1)."""
        eps = 1e-8
        energy = comps.pow(2).mean(dim=2)                        # (B, M)
        # sqrt(var+eps) instead of .std(): std has an infinite gradient at
        # exactly-zero variance (flat gap-filled windows -> NaN parameters)
        sd = torch.sqrt(comps.var(dim=2, unbiased=False) + eps)
        rough = comps.diff(dim=2).abs().mean(dim=2) / sd
        share = energy / (energy.sum(dim=1, keepdim=True) + eps)
        feats = torch.stack([torch.log(energy + eps), rough, share], dim=2)
        return 2.0 * torch.sigmoid(self.mlp(feats))              # (B, M, 1)


# ═══════════════════ PROPOSED MODEL ═══════════════════

class CPTB(nn.Module):
    """Causal-decomposition Patch Transformer-BiGRU with cross-attention.

    Ablation flags disable exactly one mechanism each, keeping the rest:
      use_stage1, use_stage2  — causal decomposition stages
      use_gate                — adaptive complexity gating
      use_patch               — patch tokens (False -> point tokens)
      use_cross_att           — cross-attention fusion (False -> concat+MLP)
      use_bigru               — BiGRU decoder (False -> MLP on tokens)
      use_revin               — reversible instance normalization
      use_future_cov          — future known covariates
      use_linear_skip         — per-component direct linear map to the
                                horizon added to the deep branch (V4.1)

    V4.1 linear skip: each component c_m gets its own linear map
    W_m : R^L -> R^H; the base forecast is sum_m W_m c_m and the deep branch
    (whose final layer is ZERO-initialized when the skip is active) learns a
    residual correction. W_m is initialized as the seasonal-naive selector
    (W_m[h, h] = 1 for the one-week window, since L = 7s and S = 7s), so at
    initialization the whole model outputs exactly the weekly persistence
    forecast and training can only improve on it. This grafts the strength
    of direct linear forecasters (DLinear) onto the causal multi-band
    decomposition while keeping the covariate-driven nonlinear corrections
    in the deep branch.
    """

    def __init__(self, input_window, pred_horizon, patch_length,
                 n_cov_past, n_cov_fut, steps_per_day=24,
                 d_model=64, n_heads=4, n_layers=2, dim_ff=128, dropout=0.1,
                 bigru_hidden=64, mlp_hidden=128, dropout_mlp=0.2,
                 cross_att_heads=2, stage1_kernels=None, stage2_n_bands=4,
                 stage2_kernel_size=25, gate_hidden=16, n_future_patches=4,
                 use_stage1=True, use_stage2=True, use_gate=True,
                 use_patch=True, use_cross_att=True, use_bigru=True,
                 use_revin=True, use_future_cov=True, use_linear_skip=True,
                 use_cov_skip=True):
        super().__init__()
        self.L, self.H, self.P = input_window, pred_horizon, patch_length
        self.use_stage1, self.use_stage2 = use_stage1, use_stage2
        self.use_gate, self.use_patch = use_gate, use_patch
        self.use_cross_att, self.use_bigru = use_cross_att, use_bigru
        self.use_revin, self.use_future_cov = use_revin, use_future_cov
        self.use_linear_skip = use_linear_skip
        self.use_cov_skip = use_cov_skip and use_future_cov

        if stage1_kernels is None:
            stage1_kernels = [steps_per_day // 2, steps_per_day, 7 * steps_per_day]
        self.stage1 = Stage1Decomposition(stage1_kernels)
        self.stage2 = Stage2FilterBank(stage2_n_bands, stage2_kernel_size)
        self.revin = RevIN()

        # number of components M (stage2 expands the HF component into K+1)
        if use_stage1 and use_stage2:
            self.M = len(stage1_kernels) + stage2_n_bands + 1
        elif use_stage1:
            self.M = len(stage1_kernels) + 1
        elif use_stage2:
            self.M = stage2_n_bands + 1
        else:
            self.M = 1

        self.gate = ComplexityGate(gate_hidden)

        # Patch embedding (shared across components) + component embedding
        self.N = input_window // patch_length if use_patch else input_window
        token_in = patch_length if use_patch else 1
        self.token_proj = nn.Linear(token_in, d_model)
        self.comp_emb = nn.Parameter(torch.randn(self.M, d_model) * 0.02)
        self.pos = PositionalEncoding(d_model, max_len=max(self.N, input_window) + 8)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout,
                                           batch_first=True, activation="gelu",
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        # Covariate streams
        self.cov_past_proj = nn.Linear(n_cov_past, d_model)
        self.Pf = max(1, pred_horizon // n_future_patches)
        self.Nf = pred_horizon // self.Pf
        # The future-covariate reshape requires an integer number of equal
        # patches (Nf * Pf == H). The shipped config satisfies this on every
        # resolution; guard against a mis-set patch length (round-4 audit).
        assert self.Nf * self.Pf == pred_horizon, (
            f"future patch length {self.Pf} does not divide horizon "
            f"{pred_horizon}; adjust future_patch_length_hours")
        self.cov_fut_proj = nn.Linear(n_cov_fut * self.Pf, d_model)
        self.fut_type_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        if use_cross_att:
            self.cross_att = nn.MultiheadAttention(d_model, cross_att_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
            self.cross_norm = nn.LayerNorm(d_model)
        else:
            self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())

        if use_bigru:
            self.bigru = nn.GRU(d_model, bigru_hidden, batch_first=True,
                                bidirectional=True)
            self.bigru_drop = nn.Dropout(0.2)
            feat_dim = 4 * bigru_hidden          # last state + mean pool
        else:
            feat_dim = 2 * d_model               # last token + mean pool

        self.head = nn.Sequential(
            nn.Linear(feat_dim + d_model, mlp_hidden), nn.GELU(),
            nn.Dropout(dropout_mlp), nn.Linear(mlp_hidden, pred_horizon))

        # The deep head's final layer is ALWAYS zero-initialized so that the
        # 'w/o linear skip' ablation differs from the full model in exactly one
        # component (the skip pathway) and not also in the head init (round-3
        # audit). Zero-init of a final layer is benign — it still receives
        # gradients from the first step.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # V4.1 per-component linear skip: W (M, H, L). Initialized as the
        # seasonal-naive selector (window position h -> horizon step h, valid
        # because L = 7s and the weekly season S = 7s). Combined with the
        # zero-initialized deep head, the untrained model outputs EXACTLY the
        # weekly-persistence forecast.
        if use_linear_skip:
            W = torch.zeros(self.M, pred_horizon, input_window)
            for h in range(pred_horizon):
                W[:, h, h] = 1.0
            self.skip_W = nn.Parameter(W)
            self.skip_b = nn.Parameter(torch.zeros(pred_horizon))

        # [V4.2] Covariate skip: a direct additive map from the FULL-RESOLUTION
        # future known covariates to the horizon. The patch-averaging of
        # covariates in the cross-attention path throws away hourly temperature
        # resolution — the dominant signal on covariate-rich datasets — which
        # the round-4 GPU ablation showed the model was under-using. This head,
        # zero-initialized, gives temperature/calendar a direct route to the
        # forecast (the same additive zero-init injection that makes DLinear
        # competitive). Neutral at init; near-zero where covariates are weak.
        if self.use_cov_skip:
            self.cov_skip = nn.Sequential(
                nn.Linear(pred_horizon * n_cov_fut, mlp_hidden), nn.GELU(),
                nn.Linear(mlp_hidden, pred_horizon))
            nn.init.zeros_(self.cov_skip[-1].weight)
            nn.init.zeros_(self.cov_skip[-1].bias)

    # ── decomposition ──
    def decompose(self, z):
        """z: (B, L) -> components (B, M, L), summing to z exactly."""
        if self.use_stage1:
            comps = self.stage1(z)                    # [HF, intraday, weekly, trend]
            if self.use_stage2:
                comps = self.stage2(comps[0]) + comps[1:]
        elif self.use_stage2:
            comps = self.stage2(z)
        else:
            comps = [z]
        return torch.stack(comps, dim=1)              # (B, M, L)

    # ── tokenization ──
    def tokenize(self, comps):
        """(B, M, L) -> tokens (B, M, N, d)."""
        B, M, L = comps.shape
        if self.use_patch:
            x = comps.reshape(B * M, self.N, self.P)
        else:
            x = comps.reshape(B * M, L, 1)
        tok = self.token_proj(x)                      # (B*M, N, d)
        tok = self.pos(tok).reshape(B, M, -1, tok.size(-1))
        tok = tok + self.comp_emb[None, :, None, :]
        return self.drop(tok)

    def forward(self, x_load, x_cov_past, x_cov_fut):
        B = x_load.size(0)

        # RevIN
        if self.use_revin:
            z, mu, sd = self.revin.normalize(x_load)
        else:
            z = x_load

        # Causal adaptive dual-stage decomposition
        comps = self.decompose(z)                     # (B, M, L)
        gates = self.gate(comps) if self.use_gate else None

        # Tokens over the (component x patch) grid
        tok = self.tokenize(comps)                    # (B, M, N, d)
        if gates is not None:
            tok = tok * gates.unsqueeze(-1)
        N, d = tok.size(2), tok.size(3)
        h = self.encoder(tok.reshape(B, self.M * N, d))
        h = h.reshape(B, self.M, N, d).mean(dim=1)    # time tokens (B, N, d)

        # Covariate tokens: past per-patch means + future patches
        if self.use_patch:
            cp = x_cov_past.reshape(B, self.N, self.P, -1).mean(dim=2)
        else:
            cp = x_cov_past
        cov_tokens = self.pos(self.cov_past_proj(cp))            # (B, N, d)
        if self.use_future_cov:
            cf = x_cov_fut.reshape(B, self.Nf, self.Pf * x_cov_fut.size(-1))
            cf = self.cov_fut_proj(cf) + self.fut_type_emb       # (B, Nf, d)
            cov_tokens = torch.cat([cov_tokens, cf], dim=1)      # (B, N+Nf, d)

        # Fusion
        attn_w = None
        if self.use_cross_att:
            out, attn_w = self.cross_att(h, cov_tokens, cov_tokens,
                                         need_weights=True,
                                         average_attn_weights=True)
            h = self.cross_norm(h + out)
        else:
            pooled = cov_tokens.mean(dim=1, keepdim=True).expand(-1, h.size(1), -1)
            h = self.fuse(torch.cat([h, pooled], dim=-1))

        # Decoder
        if self.use_bigru:
            g, _ = self.bigru(h)
            g = self.bigru_drop(g)
            feat = torch.cat([g[:, -1], g.mean(dim=1)], dim=-1)
        else:
            feat = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)

        fut_summary = (cov_tokens[:, -self.Nf:].mean(dim=1)
                       if self.use_future_cov else
                       torch.zeros(B, self.comp_emb.size(1),
                                   device=x_load.device, dtype=feat.dtype))
        y = self.head(torch.cat([feat, fut_summary], dim=-1))

        # V4.1: per-component linear base + deep residual correction.
        # Uses the RAW components (pre-gating) so the base is a pure linear
        # map of the decomposition, independent of the gating pathway.
        if self.use_linear_skip:
            base = torch.einsum("bml,mhl->bh", comps, self.skip_W) + self.skip_b
            y = base + y

        # V4.2: full-resolution future-covariate skip (zero-init).
        if self.use_cov_skip:
            y = y + self.cov_skip(x_cov_fut.reshape(x_cov_fut.size(0), -1))

        if self.use_revin:
            y = self.revin.denormalize(y, mu, sd)
        return y, attn_w


# ═══════════════════ CAUSAL ERROR CORRECTION ═══════════════════

class CausalErrorCorrection(nn.Module):
    """
    [FIX-3] GRU that maps the last W OBSERVED day-ahead errors to a correction
    for the next horizon. Under the operational protocol (forecasts issued
    every H steps), the errors of all forecasts issued up to t-H are fully
    realized at issue time t, so the input never touches unobserved data.
    Trained on validation errors only.
    """

    def __init__(self, input_window, horizon, hidden=32):
        super().__init__()
        self.W = input_window
        self.gru = nn.GRU(1, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(),
                                nn.Linear(64, horizon))

    def forward(self, err_window):
        # Use the FINAL hidden state of both directions (h_n), i.e. each
        # direction's summary of the WHOLE window. This avoids the PyTorch
        # gotcha where output[:, -1] pairs the forward final state with the
        # backward direction's one-step state, and matches the MATLAB port
        # (round-3 parity audit).
        _, h_n = self.gru(err_window.unsqueeze(-1))     # h_n: (2, B, hidden)
        last = torch.cat([h_n[0], h_n[1]], dim=-1)      # (B, 2*hidden)
        return self.fc(last)


def create_proposed(cfg, mp, dp, n_cov_past, n_cov_fut, **ablation_flags):
    """Factory wired to config dicts (cfg=DATASETS[ds], mp=MODEL_PARAMS,
    dp=DECOMP_PARAMS)."""
    spd = cfg["steps_per_day"]
    scale = spd / 24.0
    kernels = [max(2, int(round(k * scale))) for k in dp["stage1_kernels_hours"]]
    # Derive the number of future-covariate patches from the config'd patch
    # length in hours (round-3 audit: the key was previously never read).
    fut_patch_steps = max(1, int(round(mp["future_patch_length_hours"] * scale)))
    n_future_patches = max(1, cfg["pred_horizon"] // fut_patch_steps)
    # [V5] Whether the full-resolution covariate path is active is decided by
    # an explicit selection rule evaluated on the VALIDATION split only
    # (config.COV_SKIP_BY_DATASET, see select_covskip_on_val.py). An earlier
    # version gated it on `has_temperature`, but that rule had been chosen
    # after inspecting TEST scores — a selection-leakage channel of exactly
    # the kind this paper argues against. The path is zero-initialised, so it
    # is neutral at initialisation whichever way the rule falls.
    from config import COV_SKIP_BY_DATASET
    kw = dict(use_cov_skip=COV_SKIP_BY_DATASET.get(cfg.get("name", ""), True))
    kw.update(ablation_flags)
    return CPTB(
        input_window=cfg["input_window"], pred_horizon=cfg["pred_horizon"],
        patch_length=cfg["patch_length"], n_cov_past=n_cov_past,
        n_cov_fut=n_cov_fut, steps_per_day=spd,
        d_model=mp["d_model"], n_heads=mp["n_heads"],
        n_layers=mp["n_encoder_layers"], dim_ff=mp["dim_feedforward"],
        dropout=mp["dropout_transformer"], bigru_hidden=mp["bigru_hidden"],
        mlp_hidden=mp["mlp_hidden"], dropout_mlp=mp["dropout_mlp"],
        cross_att_heads=mp["cross_att_heads"], n_future_patches=n_future_patches,
        stage1_kernels=kernels, stage2_n_bands=dp["stage2_n_bands"],
        stage2_kernel_size=dp["stage2_kernel_size"],
        gate_hidden=dp["gate_hidden"], **kw)
