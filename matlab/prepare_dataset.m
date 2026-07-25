function data = prepare_dataset(datasetName, cfg, verbose)
%PREPARE_DATASET  Load and preprocess one dataset, LEAK-FREE.
%
%   data = PREPARE_DATASET(name, cfg) mirrors SMART_GRIDS_CODE_V4/data_utils.py:
%     1. Read CSV, average duplicate timestamps, reindex to a regular grid.
%     2. Interpolate short gaps (<= 6 steps), then forward/backward fill.
%     3. Chronological 70/15/15 split indices computed FIRST.
%     4. [FIX-4] IQR outlier bounds (3x IQR) fitted on TRAIN only; rolling
%        daily-median replacement applied to the TRAIN segment only.
%     5. Calendar features (deterministic, leak-free).
%     6. [FIX-2] Z-score scaler fitted on the TRAIN segment only.
%
%   Returns a struct with fields:
%     load_z (n x 1), load_raw (n x 1), cov_past (n x C), cov_fut (n x Cf),
%     train_end, val_end, scaler (mu, sigma), n_cov_past, n_cov_fut, cfgd.
%
%   Requires base MATLAB only (no Statistics Toolbox: quantiles and rolling
%   medians are implemented locally).

if nargin < 3, verbose = true; end
cfgd = cfg.datasets.(datasetName);
fp = fullfile(cfg.data_dir, cfgd.csv_file);
if ~isfile(fp)
    error("Dataset file not found: %s", fp);
end

% ── 1. Read + regular grid ─────────────────────────────────────────────
opts = detectImportOptions(fp);
T = readtable(fp, opts);
switch datasetName
    case 'GEFCom2014'
        dt = T.datetime; load_v = T.load;
        temp_v = T.temperature;
        step = hours(1);
    case 'PJM'
        dt = T.Datetime; load_v = T.PJME_MW;
        temp_v = [];
        step = hours(1);
    case 'AEMO'
        dt = T.datetime; load_v = T.load;
        temp_v = [];
        step = minutes(30);
    otherwise
        error("Unknown dataset '%s'", datasetName);
end
if ~isdatetime(dt), dt = datetime(dt); end

% average duplicated timestamps (PJM has 4 at DST transitions);
% omitnan mirrors pandas groupby().mean() when one duplicate is NaN
[dtu, ~, ic] = unique(dt);
load_u = accumarray(ic, load_v, [], @(v) mean(v, 'omitnan'));
if ~isempty(temp_v)
    temp_u = accumarray(ic, temp_v, [], @(v) mean(v, 'omitnan'));
end

% strict regular grid (drops sub-grid records, e.g. AEMO 5-min data)
grid = (dtu(1):step:dtu(end))';
[tf, pos] = ismember(grid, dtu);
loadg = nan(numel(grid), 1);
loadg(tf) = load_u(pos(tf));
% [CRITICAL-FIX, round-3 audit] Record which timestamps carry a genuine
% observation BEFORE any gap filling, so fabricated targets can be excluded
% from every evaluated window (mirrors data_utils.observed).
observed = ~isnan(loadg);
if ~isempty(temp_v)
    tempg = nan(numel(grid), 1);
    tempg(tf) = temp_u(pos(tf));
end

% ── 2. Gap filling (pandas-parity: interpolate(limit=6) fills the FIRST
%      6 NaNs of every gap using the two-sided linear ramp, then ffill
%      holds the last value; no extrapolation at the series edges) ─────
loadg = pandas_interp_limit(loadg, 6);
loadg = fillmissing(loadg, 'previous');
loadg = fillmissing(loadg, 'next');
if ~isempty(temp_v)
    tempg = fillmissing(tempg, 'linear', 'EndValues', 'none');
    tempg = fillmissing(tempg, 'previous');
    tempg = fillmissing(tempg, 'next');
end

n = numel(loadg);
train_end = floor(n * cfgd.train_ratio);
val_end   = floor(n * (cfgd.train_ratio + cfgd.val_ratio));

% ── 3. [FIX-4] Train-only outlier handling ─────────────────────────────
tr = loadg(1:train_end);
q1 = local_quantile(tr, 0.25);
q3 = local_quantile(tr, 0.75);
iqr_ = q3 - q1;
lo = q1 - 3.0 * iqr_;  hi = q3 + 3.0 * iqr_;
mask = (tr < lo) | (tr > hi);
if any(mask)
    % Trailing median over the TRAIN slice only (round-3 audit: a centered
    % window over the full series would peek into validation at the boundary).
    med = movmedian(tr, [cfgd.steps_per_day - 1, 0], 'omitnan');
    tr(mask) = med(mask);
    loadg(1:train_end) = tr;
    if verbose
        fprintf("  [%s] corrected %d train outliers (val/test untouched)\n", ...
                datasetName, nnz(mask));
    end
end

% ── 4. Calendar features (mirror data_utils.add_calendar_features) ─────
hh  = hour(grid);
dow = mod(weekday(grid) + 5, 7);          % Monday = 0 ... Sunday = 6
mo  = month(grid);
dd  = day(grid);
cov = [sin(2*pi*hh/24),  cos(2*pi*hh/24), ...
       sin(2*pi*dow/7),  cos(2*pi*dow/7), ...
       sin(2*pi*(mo-1)/12), cos(2*pi*(mo-1)/12), ...
       double(dow >= 5), ...
       double((mo == 1 & dd == 1) | (mo == 12 & dd >= 25))];

% ── 5. [FIX-2] Train-only z-score scaling (population std, ddof=0,
%      mirroring numpy/TrainOnlyScaler) ─────────────────────────────────
mu = mean(loadg(1:train_end));
sigma = std(loadg(1:train_end), 1);
if sigma < 1e-12, sigma = 1; end
load_z = (loadg - mu) / sigma;

cov_past = cov;
cov_fut  = cov;
if ~isempty(temp_v)
    mut = mean(tempg(1:train_end));
    st  = std(tempg(1:train_end), 1); if st < 1e-12, st = 1; end
    tz = (tempg - mut) / st;
    cov_past = [cov_past, tz];
    cov_fut  = [cov_fut, tz];    % perfect-forecast proxy, stated in paper

    % [V4.1] Degree-day features around the TRAIN-median temperature
    % (causal; z-scored with train statistics; mirrors data_utils.py).
    % Gated by cfg.use_degree_days for parity with USE_DEGREE_DAYS (round-3).
    use_dd = ~isfield(cfg, 'use_degree_days') || cfg.use_degree_days;
    if use_dd
        tref = median(tempg(1:train_end));
        hdd = max(tref - tempg, 0);
        cdd = max(tempg - tref, 0);
        mh = mean(hdd(1:train_end)); sh = std(hdd(1:train_end), 1);
        mc = mean(cdd(1:train_end)); sc = std(cdd(1:train_end), 1);
        if sh < 1e-12, sh = 1; end
        if sc < 1e-12, sc = 1; end
        cov_past = [cov_past, (hdd - mh)/sh, (cdd - mc)/sc];
        cov_fut  = [cov_fut,  (hdd - mh)/sh, (cdd - mc)/sc];
    end
end

if verbose
    fprintf("  [%s] n=%d  train=[1:%d]  val=(%d:%d]  test=(%d:%d]\n", ...
            datasetName, n, train_end, train_end, val_end, val_end, n);
    fprintf("  [%s] observed=%d/%d (%.1f%%); fabricated target windows " + ...
            "excluded from evaluation\n", datasetName, nnz(observed), n, ...
            100*mean(observed));
end

data = struct('name', datasetName, 'cfgd', cfgd, ...
              'load_z', single(load_z), 'load_raw', loadg, ...
              'cov_past', single(cov_past), 'cov_fut', single(cov_fut), ...
              'observed', observed, ...
              'train_end', train_end, 'val_end', val_end, ...
              'scaler_mu', mu, 'scaler_sigma', sigma, ...
              'n_cov_past', size(cov_past, 2), ...
              'n_cov_fut', size(cov_fut, 2));
end

% ────────────────────────────────────────────────────────────────────────
function q = local_quantile(x, p)
%LOCAL_QUANTILE Linear-interpolation quantile (avoids Statistics Toolbox).
x = sort(x(~isnan(x)));
m = numel(x);
h = (m - 1) * p + 1;
lo = floor(h); hi = ceil(h);
q = x(lo) + (h - lo) * (x(hi) - x(lo));
end

function y = pandas_interp_limit(y, lim)
%PANDAS_INTERP_LIMIT Replicate pandas interpolate(method='linear', limit=n):
%   compute the two-sided linear interpolation for every interior gap, then
%   keep only the FIRST `lim` filled values of each gap; edges untouched.
full = fillmissing(y, 'linear', 'EndValues', 'none');
isn = isnan(y);
d = diff([0; isn; 0]);
gapStart = find(d == 1);
gapEnd = find(d == -1) - 1;
for g = 1:numel(gapStart)
    s = gapStart(g); e = gapEnd(g);
    keepTo = min(s + lim - 1, e);
    y(s:keepTo) = full(s:keepTo);
end
end
