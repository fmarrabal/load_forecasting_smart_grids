function model = cptb_init(cfg, cfgd, nCovPast, nCovFut, flags)
%CPTB_INIT  Initialize the CPTB model (parameters + constants + flags).
%
%   model = CPTB_INIT(cfg, cfgd, nCovPast, nCovFut, flags) builds:
%     model.p      struct of LEARNABLE dlarrays (flat field names, so that
%                  dlgradient / adamupdate can operate on the whole struct)
%     model.c      constants (positional encoding, stage-1 kernels, dims)
%     model.flags  ablation flags (use_stage1, use_stage2, use_gate,
%                  use_patch, use_cross_att, use_bigru, use_revin,
%                  use_future_cov, use_linear_skip, use_cov_skip) — each
%                  disables exactly one component.
%
%   Mirrors: SMART_GRIDS_CODE_V4/model_proposed.py (CPTB.__init__,
%   Stage2FilterBank sinc initialization, create_proposed factory).
%
%   Initialization: Glorot uniform for projection weights, zeros for
%   biases, Glorot for GRU input/recurrent weights (documented deviation
%   from PyTorch's uniform(-1/sqrt(h), 1/sqrt(h)); both are standard).

if nargin < 5, flags = struct(); end
% [V5] Whether the full-resolution covariate path is active is read from
% cfg.cov_skip_by_dataset, which holds the outcome of a VALIDATION-only
% selection rule (see config_v4.m). An earlier version gated it on the mere
% presence of temperature, but that rule had itself been settled on after
% looking at test-set scores. Mirrors create_proposed in model_proposed.py;
% overridable per call via flags, which is how the ablation drives it.
covdef = true;
if isfield(cfg, 'cov_skip_by_dataset') && isfield(cfgd, 'name') && ...
        isfield(cfg.cov_skip_by_dataset, cfgd.name)
    covdef = cfg.cov_skip_by_dataset.(cfgd.name);
end
def = struct('use_stage1', true, 'use_stage2', true, 'use_gate', true, ...
             'use_patch', true, 'use_cross_att', true, 'use_bigru', true, ...
             'use_revin', true, 'use_future_cov', true, ...
             'use_linear_skip', true, 'use_cov_skip', covdef);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(flags, fn{i}), flags.(fn{i}) = def.(fn{i}); end
end
flags.use_cov_skip = flags.use_cov_skip && flags.use_future_cov;

mp = cfg.model;  dp = cfg.decomp;
d  = mp.d_model;
L  = cfgd.input_window;
H  = cfgd.pred_horizon;
P  = cfgd.patch_length;
spd = cfgd.steps_per_day;
scale = spd / 24;
kernels = sort(max(2, round(dp.stage1_kernels_hours * scale)));  % ascending,
% mirroring Stage1Decomposition (forward assumes mas{1}=smallest kernel)
K  = dp.stage2_n_bands;
ks = dp.stage2_kernel_size;

% number of components M (stage 2 expands the HF component into K+1)
if flags.use_stage1 && flags.use_stage2
    M = numel(kernels) + K + 1;
elseif flags.use_stage1
    M = numel(kernels) + 1;
elseif flags.use_stage2
    M = K + 1;
else
    M = 1;
end

if flags.use_patch, N = L / P; tokIn = P; else, N = L; tokIn = 1; end
Pf = max(1, floor(H / mp.n_future_patches));
Nf = floor(H / Pf);

p = struct();

% ── RevIN affine ──
p.revin_g = dlarray(1);  p.revin_b = dlarray(0);

% ── Stage 2: FIR bank initialized as Hamming-windowed sinc band-pass ──
% Center the sinc mid-kernel so its main lobe aligns with the Hamming peak
% (round-3 audit: centering on the last tap scaled the main lobe by
% hamming[end]≈0.08). The left-padded conv stays causal regardless.
w = zeros(ks, 1, K);
nn = (0:ks-1)' - (ks - 1)/2;
ham = 0.54 - 0.46 * cos(2*pi*(0:ks-1)'/(ks-1));
sinc_lp = @(fc) 2*fc*sinc_local(2*fc*nn);
for b = 1:K
    f1 = 0.5 * (b-1) / K;  f2 = 0.5 * b / K;  % Nyquist = 0.5
    if b == 1, hb = sinc_lp(f2); else, hb = sinc_lp(f2) - sinc_lp(f1); end
    w(:, 1, b) = hb .* ham;
end
p.s2W = dlarray(single(w));

% ── Complexity gate: MLP(3 -> gate_hidden -> 1) ──
gh = dp.gate_hidden;
p.gateW1 = glorot(gh, 3);   p.gateb1 = dlarray(zeros(gh, 1, 'single'));
p.gateW2 = glorot(1, gh);   p.gateb2 = dlarray(zeros(1, 1, 'single'));

% ── Token projection + component embedding ──
p.tokW = glorot(d, tokIn);  p.tokb = dlarray(zeros(d, 1, 'single'));
p.compEmb = dlarray(0.02 * randn(d, M, 'single'));

% ── Transformer encoder (pre-norm), n layers ──
ff = mp.dim_feedforward;
for l = 1:mp.n_encoder_layers
    pre = sprintf('enc%d_', l);
    p.([pre 'n1g']) = dlarray(ones(d, 1, 'single'));
    p.([pre 'n1b']) = dlarray(zeros(d, 1, 'single'));
    p.([pre 'Wq']) = glorot(d, d);  p.([pre 'bq']) = dlarray(zeros(d, 1, 'single'));
    p.([pre 'Wk']) = glorot(d, d);  p.([pre 'bk']) = dlarray(zeros(d, 1, 'single'));
    p.([pre 'Wv']) = glorot(d, d);  p.([pre 'bv']) = dlarray(zeros(d, 1, 'single'));
    p.([pre 'Wo']) = glorot(d, d);  p.([pre 'bo']) = dlarray(zeros(d, 1, 'single'));
    p.([pre 'n2g']) = dlarray(ones(d, 1, 'single'));
    p.([pre 'n2b']) = dlarray(zeros(d, 1, 'single'));
    p.([pre 'W1']) = glorot(ff, d); p.([pre 'b1']) = dlarray(zeros(ff, 1, 'single'));
    p.([pre 'W2']) = glorot(d, ff); p.([pre 'b2']) = dlarray(zeros(d, 1, 'single'));
end

% ── Covariate streams ──
p.covW = glorot(d, nCovPast);  p.covb = dlarray(zeros(d, 1, 'single'));
p.futW = glorot(d, Pf * nCovFut); p.futb = dlarray(zeros(d, 1, 'single'));
p.futEmb = dlarray(0.02 * randn(d, 1, 'single'));

% ── Fusion ──
if flags.use_cross_att
    p.cWq = glorot(d, d); p.cbq = dlarray(zeros(d, 1, 'single'));
    p.cWk = glorot(d, d); p.cbk = dlarray(zeros(d, 1, 'single'));
    p.cWv = glorot(d, d); p.cbv = dlarray(zeros(d, 1, 'single'));
    p.cWo = glorot(d, d); p.cbo = dlarray(zeros(d, 1, 'single'));
    p.cNg = dlarray(ones(d, 1, 'single'));
    p.cNb = dlarray(zeros(d, 1, 'single'));
else
    p.fuseW = glorot(d, 2 * d); p.fuseb = dlarray(zeros(d, 1, 'single'));
end

% ── BiGRU decoder ──
hgru = mp.bigru_hidden;
if flags.use_bigru
    p.gWf = glorot(3*hgru, d);  p.gRf = glorot(3*hgru, hgru);
    p.gbf = dlarray(zeros(3*hgru, 1, 'single'));
    p.gWb = glorot(3*hgru, d);  p.gRb = glorot(3*hgru, hgru);
    p.gbb = dlarray(zeros(3*hgru, 1, 'single'));
    featDim = 4 * hgru;               % last state + mean pool, both dirs
else
    featDim = 2 * d;                  % last token + mean pool
end

% ── Head ── (final layer ALWAYS zero-initialized so the 'w/o linear skip'
% ablation differs from the full model in exactly one component, not also in
% the head init; round-3 audit)
mh = mp.mlp_hidden;
p.headW1 = glorot(mh, featDim + d);
p.headb1 = dlarray(zeros(mh, 1, 'single'));
p.headW2 = dlarray(zeros(H, mh, 'single'));
p.headb2 = dlarray(zeros(H, 1, 'single'));

% ── [V4.1] Per-component linear skip: W (H x L x M), seasonal-naive init
% (window position h -> horizon step h, valid because L = 7s = season).
% Combined with the zero-initialized head, the untrained model outputs
% EXACTLY the weekly-persistence forecast; training can only improve on it.
if flags.use_linear_skip
    Wsk = zeros(H, L, M, 'single');
    for hh_ = 1:H
        Wsk(hh_, hh_, :) = 1;
    end
    p.skipW = dlarray(Wsk);
    p.skipb = dlarray(zeros(H, 1, 'single'));
end

% ── [V4.2] Full-resolution future-covariate skip (zero-init last layer):
% a direct additive map from the hourly future covariates to the horizon,
% mirroring the injection that makes DLinear competitive. Enabled only where
% temperature is present (see flags above).
if flags.use_cov_skip
    nCf = nCovFut;
    p.covsW1 = glorot(mh, H * nCf);  p.covsb1 = dlarray(zeros(mh, 1, 'single'));
    p.covsW2 = dlarray(zeros(H, mh, 'single'));
    p.covsb2 = dlarray(zeros(H, 1, 'single'));
end

% ── constants ──
c = struct('d', d, 'L', L, 'H', H, 'P', P, 'N', N, 'M', M, 'K', K, ...
           'ks', ks, 'Pf', Pf, 'Nf', Nf, 'kernels', kernels, ...
           'nHeads', mp.n_heads, 'nHeadsCross', mp.cross_att_heads, ...
           'nLayers', mp.n_encoder_layers, ...
           'dropT', mp.dropout_transformer, 'dropG', mp.dropout_bigru, ...
           'dropM', mp.dropout_mlp, 'hgru', hgru);
c.posenc = single(positional_encoding(d, max(N, L) + 8));

model = struct('p', p, 'c', c, 'flags', flags);
end

% ────────────────────────────────────────────────────────────────────────
function W = glorot(nOut, nIn)
r = sqrt(6 / (nIn + nOut));
W = dlarray(single((rand(nOut, nIn) * 2 - 1) * r));
end

function y = sinc_local(x)
y = ones(size(x));
nz = x ~= 0;
y(nz) = sin(pi * x(nz)) ./ (pi * x(nz));
end

function pe = positional_encoding(d, maxLen)
pos = (0:maxLen-1)';
div = exp((0:2:d-1) * (-log(10000) / d));
pe = zeros(maxLen, d);
pe(:, 1:2:end) = sin(pos .* div);
pe(:, 2:2:end) = cos(pos .* div);
pe = pe';                               % d x maxLen
end
