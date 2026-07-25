function [yPredEc, ecInfo] = error_correction(model, data, valIdx, testIdx, ...
                                              yPredTest, yTrueTest, cfg, seed)
%ERROR_CORRECTION  Causal error correction ([FIX-3]).
%
%   Trains a small GRU on VALIDATION errors mapping the last W observed
%   errors to an H-step correction, then applies it on the test set using
%   only errors of forecasts already realized at issue time. Under the
%   operational protocol (stride = H), every previous forecast's target is
%   fully observed when the next forecast is issued, so no unobserved data
%   ever enters the input — unlike V3, which read test-set ground truth
%   through overlapping windows.
%
%   Mirrors: SMART_GRIDS_CODE_V4/train_pipeline.py
%   (fit_error_correction / apply_error_correction / build_error_series).

rng(seed);
ec = cfg.ec;
H = data.cfgd.pred_horizon;
W = ec.input_window_hours * data.cfgd.steps_per_day / 24;
W = round(W);

% ── validation errors (original scale) ──
[vp, vt] = predict_cptb(model, data, valIdx);
[errVal, ~] = build_error_series(vt, vp, valIdx, H);

% training pairs: last W errors -> next H errors, stepping by H
X = []; Y = [];
for s = W+1:H:numel(errVal)-H+1
    xw = errVal(s-W:s-1);  yw = errVal(s:s+H-1);
    if any(isnan(xw)) || any(isnan(yw)), continue; end
    X = [X, xw(:)];  Y = [Y, yw(:)];                 %#ok<AGROW>
end
if size(X, 2) < 32
    yPredEc = yPredTest;  ecInfo = struct('trained', false);
    return;
end

% ── tiny BIDIRECTIONAL GRU: 1 -> 2*hidden -> H (mirrors the Python
%    CausalErrorCorrection; bidirectionality over a fully-observed past
%    window is still causal) ──
hid = ec.hidden_size;
p.gWf = glorot(3*hid, 1);  p.gRf = glorot(3*hid, hid);
p.gbf = dlarray(zeros(3*hid, 1, 'single'));
p.gWb = glorot(3*hid, 1);  p.gRb = glorot(3*hid, hid);
p.gbb = dlarray(zeros(3*hid, 1, 'single'));
p.W1 = glorot(64, 2*hid);  p.b1 = dlarray(zeros(64, 1, 'single'));
p.W2 = glorot(H, 64);      p.b2 = dlarray(zeros(H, 1, 'single'));

nTr = size(X, 2);
avgG = []; avgSq = []; it = 0;
for epch = 1:ec.epochs
    perm = randperm(nTr);
    for s = 1:64:nTr
        sel = perm(s:min(s+63, nTr));
        % X(:,sel): W x B -> 1 x B x W  ('CBT': channel=1, batch, time)
        xb = dlarray(single(permute(reshape(X(:, sel), W, 1, []), [2 3 1])));
        yb = dlarray(single(Y(:, sel)));
        [~, g] = dlfeval(@ec_loss, p, xb, yb, hid);
        it = it + 1;
        [p, avgG, avgSq] = adamupdate(p, g, avgG, avgSq, it, ec.lr);
    end
end

% ── causal application on test ──
[errTest, t0] = build_error_series(yTrueTest, yPredTest, testIdx, H);
yPredEc = yPredTest;
for i = 1:numel(testIdx)
    rel = testIdx(i) - t0 + 1;
    if rel > W
        xw = errTest(rel-W:rel-1);
        if any(isnan(xw)), continue; end
        xin = dlarray(single(reshape(xw, 1, 1, W)));          % C x B x T
        corr = ec_forward(p, xin, hid);
        yPredEc(i, :) = yPredTest(i, :) + double(extractdata(corr))';
    end
end
ecInfo = struct('trained', true, 'n_train_pairs', nTr);
end

% ────────────────────────────────────────────────────────────────────────
function [err, t0] = build_error_series(yTrue, yPred, issueIdx, H)
%Flatten non-overlapping day-ahead forecasts into a time-indexed series.
t0 = issueIdx(1);
T = issueIdx(end) + H - t0;
err = nan(T, 1);
for i = 1:numel(issueIdx)
    a = issueIdx(i) - t0 + 1;
    err(a:a+H-1) = yTrue(i, :) - yPred(i, :);
end
end

function y = ec_forward(p, x, hid)
% x: 1 x B x T ('CBT'); bidirectional GRU, concat of both last states
B = size(x, 2);
h0 = dlarray(zeros(hid, B, 'single'));
gf = gru(x, h0, p.gWf, p.gRf, p.gbf, 'DataFormat', 'CBT');
gb = gru(x(:, :, end:-1:1), h0, p.gWb, p.gRb, p.gbb, 'DataFormat', 'CBT');
last = cat(1, reshape(gf(:, :, end), hid, B), ...
              reshape(gb(:, :, end), hid, B));        % 2*hid x B
z = max(fullyconnect(last, p.W1, p.b1, 'DataFormat', 'CB'), 0);
y = fullyconnect(z, p.W2, p.b2, 'DataFormat', 'CB');
end

function [loss, g] = ec_loss(p, x, y, hid)
yh = ec_forward(p, x, hid);
loss = mean((yh - y).^2, 'all');
g = dlgradient(loss, p);
end

function W = glorot(nOut, nIn)
r = sqrt(6 / (nIn + nOut));
W = dlarray(single((rand(nOut, nIn) * 2 - 1) * r));
end
