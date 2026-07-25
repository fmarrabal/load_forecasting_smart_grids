function [Y, attnW] = cptb_forward(model, XL, XCP, XCF, doTraining)
%CPTB_FORWARD  Forward pass of the proposed CPTB model.
%
%   [Y, attnW] = CPTB_FORWARD(model, XL, XCP, XCF, doTraining)
%     XL  : L x B          past load (z-scored), dlarray or numeric
%     XCP : L x C x B      past covariates
%     XCF : H x Cf x B     future known covariates
%     Y   : H x B          forecast (z-scored scale)
%     attnW : cross-attention weights (Tkv x Tq x heads x B) or []
%
%   Pipeline (mirrors SMART_GRIDS_CODE_V4/model_proposed.py, CPTB.forward):
%     RevIN -> causal dual-stage decomposition -> complexity gate ->
%     patch tokens (+pos, +component embedding) -> Transformer encoder ->
%     component-mean time tokens -> cross-attention with past+future
%     covariate tokens -> BiGRU -> head -> RevIN denormalize.
%
%   Every operation is strictly causal within the input window; the
%   decomposition reconstructs the window exactly (see verify_cptb.m).

p = model.p;  c = model.c;  f = model.flags;
if nargin < 5, doTraining = false; end
if ~isa(XL, 'dlarray'), XL = dlarray(single(XL)); end
if ~isa(XCP, 'dlarray'), XCP = dlarray(single(XCP)); end
if ~isa(XCF, 'dlarray'), XCF = dlarray(single(XCF)); end
[L, B] = size(XL);
d = c.d;  M = c.M;  N = c.N;  eps_ = 1e-5;

% ── RevIN ───────────────────────────────────────────────────────────────
if f.use_revin
    mu = mean(XL, 1);
    sd = sqrt(mean((XL - mu).^2, 1) + eps_);
    z = (XL - mu) ./ sd;
    z = z * p.revin_g + p.revin_b;
else
    z = XL;
end

% ── Stage 1: causal multi-scale moving-average split ───────────────────
comps = {};                                    % each: L x 1 x B
z3 = reshape(z, L, 1, B);
if f.use_stage1
    mas = cell(1, numel(c.kernels));
    for i = 1:numel(c.kernels)
        mas{i} = causal_ma(z3, c.kernels(i));
    end
    hf = z3 - mas{1};
    for i = 1:numel(mas)-1
        comps{end+1} = mas{i} - mas{i+1};      %#ok<AGROW>
    end
    comps{end+1} = mas{end};
else
    hf = z3;
end

% ── Stage 2: learnable causal FIR band-pass bank on HF ─────────────────
if f.use_stage2
    hfp = cat(1, zeros(c.ks - 1, 1, B, 'like', hf), hf);   % causal zero pad
    bands = dlconv(hfp, p.s2W, 0, 'DataFormat', 'SCB');    % L x K x B
    rem_ = hf - sum(bands, 2);
    comps = [{bands}, {rem_}, comps];
elseif f.use_stage1
    comps = [{hf}, comps];
else
    comps = {hf};
end
Ccomp = cat(2, comps{:});                       % L x M x B

% ── Complexity gate ─────────────────────────────────────────────────────
if f.use_gate
    e8 = 1e-8;
    energy = mean(Ccomp.^2, 1);                                 % 1 x M x B
    % NOTE: sqrt(var+eps) instead of std() — std has an infinite gradient
    % at exactly-zero variance (flat gap-filled windows), which poisons
    % the RevIN parameters with NaN.
    sdC = sqrt(mean((Ccomp - mean(Ccomp, 1)).^2, 1) + e8);
    rough = mean(abs(Ccomp(2:end,:,:) - Ccomp(1:end-1,:,:)), 1) ./ sdC;
    share = energy ./ (sum(energy, 2) + e8);
    feats = cat(1, log(energy + e8), rough, share);             % 3 x M x B
    fx = reshape(feats, 3, M * B);
    g1 = max(fullyconnect(fx, p.gateW1, p.gateb1, 'DataFormat', 'CB'), 0);
    g2 = fullyconnect(g1, p.gateW2, p.gateb2, 'DataFormat', 'CB');
    gate = 2 ./ (1 + exp(-g2));                                 % 1 x (M*B)
    gate = reshape(gate, 1, 1, M, B);
else
    gate = [];
end

% ── Patch tokens ────────────────────────────────────────────────────────
if f.use_patch
    tok_in = reshape(Ccomp, c.P, N, M, B);          % P x N x M x B
    tok_in = reshape(tok_in, c.P, N * M * B);
else
    tok_in = reshape(Ccomp, 1, L * M * B);
end
tok = fullyconnect(tok_in, p.tokW, p.tokb, 'DataFormat', 'CB');  % d x (N*M*B)
tok = reshape(tok, d, N, M, B);
tok = tok + reshape(c.posenc(:, 1:N), d, N, 1, 1);   % positional enc
tok = tok + reshape(p.compEmb, d, 1, M, 1);          % component identity
if ~isempty(gate)
    tok = tok .* gate;
end
tok = dropout_(tok, c.dropT, doTraining);

% ── Transformer encoder over the M*N token grid ────────────────────────
X = reshape(tok, d, N * M, B);                       % d x T x B  ('CTB')
attDrop = c.dropT * double(doTraining);              % attention-prob dropout
for l = 1:c.nLayers
    pre = sprintf('enc%d_', l);
    % pre-norm self-attention
    Xn = layernorm(X, p.([pre 'n1b']), p.([pre 'n1g']), 'DataFormat', 'CTB');
    Q = fullyconnect(Xn, p.([pre 'Wq']), p.([pre 'bq']), 'DataFormat', 'CTB');
    K_ = fullyconnect(Xn, p.([pre 'Wk']), p.([pre 'bk']), 'DataFormat', 'CTB');
    V = fullyconnect(Xn, p.([pre 'Wv']), p.([pre 'bv']), 'DataFormat', 'CTB');
    A = attention(Q, K_, V, c.nHeads, 'DataFormat', 'CTB', ...
                  'DropoutProbability', attDrop);
    A = fullyconnect(A, p.([pre 'Wo']), p.([pre 'bo']), 'DataFormat', 'CTB');
    X = X + dropout_(A, c.dropT, doTraining);
    % pre-norm feed-forward (GELU); inner dropout mirrors PyTorch's
    % TransformerEncoderLayer linear2(dropout(gelu(linear1(x))))
    Xn = layernorm(X, p.([pre 'n2b']), p.([pre 'n2g']), 'DataFormat', 'CTB');
    Ff = fullyconnect(Xn, p.([pre 'W1']), p.([pre 'b1']), 'DataFormat', 'CTB');
    Ff = dropout_(gelu(Ff), c.dropT, doTraining);
    Ff = fullyconnect(Ff, p.([pre 'W2']), p.([pre 'b2']), 'DataFormat', 'CTB');
    X = X + dropout_(Ff, c.dropT, doTraining);
end

% ── Aggregate components -> N time tokens ──────────────────────────────
X = reshape(X, d, N, M, B);
Ht = squeeze_(mean(X, 3), [d, N, B]);                % d x N x B

% ── Covariate tokens: past per-patch means + future patches ────────────
if f.use_patch
    cp = reshape(XCP, c.P, N, [], B);                % P x N x C x B
    cp = squeeze_(mean(cp, 1), [N, size(XCP, 2), B]);% N x C x B
else
    cp = XCP;                                        % L(=N) x C x B
end
cp = permute(cp, [2 1 3]);                           % C x N x B
covTok = fullyconnect(cp, p.covW, p.covb, 'DataFormat', 'CTB');
covTok = covTok + reshape(c.posenc(:, 1:N), d, N, 1);

if f.use_future_cov
    Cf = size(XCF, 2);
    cf = reshape(XCF, c.Pf, c.Nf, Cf, B);            % Pf x Nf x Cf x B
    cf = permute(cf, [1 3 2 4]);                     % Pf x Cf x Nf x B
    cf = reshape(cf, c.Pf * Cf, c.Nf, B);
    futTok = fullyconnect(cf, p.futW, p.futb, 'DataFormat', 'CTB');
    futTok = futTok + p.futEmb;                      % future-type embedding
    covTok = cat(2, covTok, futTok);                 % d x (N+Nf) x B
end

% ── Fusion: cross-attention (queries = time tokens) ────────────────────
attnW = [];
if f.use_cross_att
    Q = fullyconnect(Ht, p.cWq, p.cbq, 'DataFormat', 'CTB');
    K_ = fullyconnect(covTok, p.cWk, p.cbk, 'DataFormat', 'CTB');
    V = fullyconnect(covTok, p.cWv, p.cbv, 'DataFormat', 'CTB');
    [A, attnW] = attention(Q, K_, V, c.nHeadsCross, 'DataFormat', 'CTB', ...
                           'DropoutProbability', attDrop);
    A = fullyconnect(A, p.cWo, p.cbo, 'DataFormat', 'CTB');
    Ht = layernorm(Ht + A, p.cNb, p.cNg, 'DataFormat', 'CTB');
else
    pooled = mean(covTok, 2);                        % d x 1 x B
    pooled = repmat(pooled, 1, size(Ht, 2), 1);
    Ht = fullyconnect(cat(1, Ht, pooled), p.fuseW, p.fuseb, ...
                      'DataFormat', 'CTB');
    Ht = gelu(Ht);
end

% ── BiGRU decoder ───────────────────────────────────────────────────────
if f.use_bigru
    H0 = dlarray(zeros(c.hgru, B, 'single'));
    Gf = gru(Ht, H0, p.gWf, p.gRf, p.gbf, 'DataFormat', 'CTB');
    Xr = Ht(:, end:-1:1, :);
    Gb = gru(Xr, H0, p.gWb, p.gRb, p.gbb, 'DataFormat', 'CTB');
    Gb = Gb(:, end:-1:1, :);
    G = cat(1, Gf, Gb);                              % 2h x N x B
    G = dropout_(G, c.dropG, doTraining);
    feat = cat(1, reshape(G(:, end, :), [], B), ...
                  reshape(mean(G, 2), [], B));       % 4h x B
else
    feat = cat(1, reshape(Ht(:, end, :), [], B), ...
                  reshape(mean(Ht, 2), [], B));      % 2d x B
end

if f.use_future_cov
    futSum = reshape(mean(covTok(:, end-c.Nf+1:end, :), 2), d, B);
else
    futSum = zeros(d, B, 'like', feat);
end

% ── Head + RevIN denormalization ────────────────────────────────────────
hh = fullyconnect(cat(1, feat, futSum), p.headW1, p.headb1, ...
                  'DataFormat', 'CB');
hh = dropout_(gelu(hh), c.dropM, doTraining);
Y = fullyconnect(hh, p.headW2, p.headb2, 'DataFormat', 'CB');  % H x B

% ── [V4.1] Per-component linear base + deep residual correction.
% Uses RAW components (pre-gating): the base is a pure linear map of the
% decomposition; mirrors model_proposed.py (einsum bml,mhl->bh).
if f.use_linear_skip
    base = repmat(p.skipb, 1, B);                     % H x B
    for m_ = 1:c.M
        base = base + p.skipW(:, :, m_) * reshape(Ccomp(:, m_, :), L, B);
    end
    Y = Y + base;
end

% ── [V4.2] Full-resolution future-covariate skip (zero-init). XCF is
% H x Cf x B; flatten as (Cf fastest, then H) to match model_proposed.py's
% x_cov_fut.reshape(B, H*Cf).
if f.use_cov_skip
    [Hh, Cf, ~] = size(XCF);
    xf = reshape(permute(XCF, [2 1 3]), Cf * Hh, B);  % (H*Cf) x B
    z1 = gelu(fullyconnect(xf, p.covsW1, p.covsb1, 'DataFormat', 'CB'));
    Y = Y + fullyconnect(z1, p.covsW2, p.covsb2, 'DataFormat', 'CB');
end

if f.use_revin
    Y = (Y - p.revin_b) ./ (p.revin_g + 1e-8);
    Y = Y .* sd + mu;
end
end

% ────────────────────────────────────────────────────────────────────────
function y = causal_ma(x3, k)
%CAUSAL_MA  MA_k[t] = mean(x[t-k+1..t]); replicate left padding. x3: L x 1 x B.
pad = repmat(x3(1, 1, :), k - 1, 1, 1);
xp = cat(1, pad, x3);
w = ones(k, 1, 1, 'single') / k;
y = dlconv(xp, w, 0, 'DataFormat', 'SCB');
end

function y = dropout_(x, rate, doTraining)
if doTraining && rate > 0
    mask = (rand(size(x), 'like', extractdata(x(1))) > rate) / (1 - rate);
    y = x .* mask;
else
    y = x;
end
end

function y = squeeze_(x, shape)
y = reshape(x, shape);
end
