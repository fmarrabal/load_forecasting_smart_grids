function [yPred, yTrue] = seasonal_naive(data, issueIdx)
%SEASONAL_NAIVE  Weekly persistence baseline: y[t:t+H) = x[t-7s : t-7s+H).
%
%   The honest floor every proposed model must beat. Original scale.
%
%   Mirrors: SMART_GRIDS_CODE_V4/models_baselines.py (SeasonalNaive).

H = data.cfgd.pred_horizon;
season = 7 * data.cfgd.steps_per_day;
nF = numel(issueIdx);
yPred = zeros(nF, H);
yTrue = zeros(nF, H);
for i = 1:nF
    t = issueIdx(i);
    yPred(i, :) = data.load_raw(t-season : t-season+H-1)';
    yTrue(i, :) = data.load_raw(t : t+H-1)';
end
end
