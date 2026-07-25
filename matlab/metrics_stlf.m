function m = metrics_stlf(yTrue, yPred)
%METRICS_STLF  MAE / RMSE / MAPE / R2 / sMAPE on the original scale.
%
%   Metrics are computed against UNMODIFIED targets ([FIX-4]); MAPE guards
%   only against literal zeros (documented, not a silent mask as in V3).
%
%   Mirrors: SMART_GRIDS_CODE_V4/metrics_stats.py (compute_metrics).

yt = double(yTrue(:));
yp = double(yPred(:));
err = yt - yp;

m.MAE  = mean(abs(err));
m.RMSE = sqrt(mean(err.^2));
den = abs(yt);  den(den < 1e-8) = NaN;
m.MAPE = mean(abs(err) ./ den, 'omitnan') * 100;
m.sMAPE = mean(2 * abs(err) ./ (abs(yt) + abs(yp) + 1e-8)) * 100;
ssRes = sum(err.^2);
ssTot = sum((yt - mean(yt)).^2);
if ssTot > 0, m.R2 = 1 - ssRes / ssTot; else, m.R2 = NaN; end
end
