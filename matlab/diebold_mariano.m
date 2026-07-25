function out = diebold_mariano(yTrue, predA, predB, hacLags, lossType)
%DIEBOLD_MARIANO  DM test for day-ahead forecasts (round-3 statistics audit).
%
%   out = DIEBOLD_MARIANO(yTrue, predA, predB) tests equal accuracy of two
%   forecast sets (nForecasts x H). Forecasts are issued once per horizon
%   (stride = H), so the per-issue-day loss differential is NON-OVERLAPPING:
%   the HLN small-sample horizon is 1 (not H), the HAC (Bartlett) bandwidth
%   defaults to the Newey-West automatic rule floor(4*(n/100)^(2/9)) rather
%   than H, and autocovariances are scaled by 1/n. NEGATIVE mean(d) means
%   model A is more accurate. p-value uses t(n-1) via the incomplete beta.
%
%   Mirrors: SMART_GRIDS_CODE_V4/metrics_stats.py (diebold_mariano).

if nargin < 5, lossType = 'mse'; end

yt = double(yTrue); pa = double(predA); pb = double(predB);
switch lossType
    case 'mse'
        la = mean((yt - pa).^2, 2);
        lb = mean((yt - pb).^2, 2);
    otherwise
        la = mean(abs(yt - pa), 2);
        lb = mean(abs(yt - pb), 2);
end

d = la - lb;
n = numel(d);
dbar = mean(d);
dc = d - dbar;

if nargin < 4 || isempty(hacLags)
    hacLags = floor(4 * (n / 100)^(2/9));
end
hacLags = max(0, min(hacLags, n - 1));

% HAC variance of the mean (Bartlett weights, autocovariances scaled by 1/n)
gamma0 = sum(dc.^2) / n;
v = gamma0;
for k = 1:hacLags
    w = 1 - k / (hacLags + 1);
    gk = sum(dc(k+1:end) .* dc(1:end-k)) / n;
    v = v + 2 * w * gk;
end
out.hac_fallback = false;
if v <= 0, v = gamma0; out.hac_fallback = true; end   % degenerate HAC
v = v / n;

if v > 0, dm = dbar / sqrt(v); else, dm = 0; end   % guard equal-loss case
corr = sqrt((n - 1) / n);             % HLN correction with horizon 1
dmHln = dm * corr;
out.n = n;
out.hac_lags = hacLags;

% two-sided p-value from t(n-1) via the regularized incomplete beta
df = n - 1;
x = df / (df + dmHln^2);
out.p_value = betainc(x, df/2, 0.5);          % == 2 * tcdf(-|dm|, df)
out.dm_stat = dmHln;
out.mean_loss_diff = dbar;
end
