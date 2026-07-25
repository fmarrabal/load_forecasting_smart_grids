function results = main_v4(datasetName, quickMode)
%MAIN_V4  STLF V4 experiment pipeline (MATLAB implementation).
%
%   results = MAIN_V4()                    % GEFCom2014, full protocol
%   results = MAIN_V4('PJM')               % one dataset
%   results = MAIN_V4('GEFCom2014', true)  % quick validation (2 epochs,
%                                          %  1 seed) — minutes, not hours
%
%   Protocol ([FIX-6], mirrors SMART_GRIDS_CODE_V4/main.py):
%     * SeasonalNaive baseline + proposed CPTB across seeds.
%     * Operational day-ahead evaluation (stride = horizon).
%     * mean ± std metrics over seeds; Diebold-Mariano vs the baseline.
%     * Optional single-flag ablation loop.
%     * Causal error correction evaluated as an ablation row.
%   Results are saved to results/results_v4_<dataset>.mat and .csv.
%
%   Note: the Python release additionally provides the full baseline suite
%   (ARIMA, GBMs, LSTM..., DLinear, PatchTST) and the leakage-quantification
%   experiment; this MATLAB implementation focuses on the proposed model,
%   its ablations and the honest evaluation protocol.

if nargin < 1 || isempty(datasetName), datasetName = 'GEFCom2014'; end
if nargin < 2, quickMode = false; end

cfg = config_v4();
if quickMode
    cfg.train.epochs = 2;  cfg.train.patience = 2;
    cfg.seeds = cfg.seeds(1);
    fprintf("QUICK MODE: 2 epochs, 1 seed (installation check only)\n");
end

data = prepare_dataset(datasetName, cfg);
H = data.cfgd.pred_horizon;
n = numel(data.load_z);

trainIdx = make_windows(data, 1, data.train_end, 1);
valIdx   = make_windows(data, data.train_end + 1, data.val_end, H);
testIdx  = make_windows(data, data.val_end + 1, n, H);
fprintf("windows: %d train / %d val / %d test (stride %d on val/test)\n", ...
        numel(trainIdx), numel(valIdx), numel(testIdx), H);

% ── SeasonalNaive floor ─────────────────────────────────────────────────
[npred, ntrue] = seasonal_naive(data, testIdx);
mNaive = metrics_stlf(ntrue, npred);
fprintf("\nSeasonalNaive: MAPE=%.2f%%  RMSE=%.2f\n", mNaive.MAPE, mNaive.RMSE);

% ── Proposed CPTB, multi-seed ───────────────────────────────────────────
perSeed = struct('metrics', {}, 'metrics_ec', {}, 'trainTime', {});
yPredPrimary = [];
ypCell = cell(1, numel(cfg.seeds));    % for the seed ensemble ([V4.1])
for s = 1:numel(cfg.seeds)
    seed = cfg.seeds(s);
    fprintf("\n── CPTB seed %d ──\n", seed);
    rng(seed);
    model = cptb_init(cfg, data.cfgd, data.n_cov_past, data.n_cov_fut);
    [model, tt] = train_cptb(model, data, trainIdx, valIdx, cfg);
    [yp, yt, attn] = predict_cptb(model, data, testIdx);
    m = metrics_stlf(yt, yp);
    fprintf("  MAPE=%.2f%%  MAE=%.2f  RMSE=%.2f  R2=%.4f\n", ...
            m.MAPE, m.MAE, m.RMSE, m.R2);

    mec = struct();
    if cfg.ec.enabled
        [ypEc, ~] = error_correction(model, data, valIdx, testIdx, ...
                                     yp, yt, cfg, seed);
        mec = metrics_stlf(yt, ypEc);
        fprintf("  with causal EC: MAPE=%.2f%%\n", mec.MAPE);
    end
    perSeed(s) = struct('metrics', m, 'metrics_ec', mec, 'trainTime', tt);
    ypCell{s} = yp;
    % Use the primary seed if present, else fall back to the first seed
    % (round-3 audit: DM/ensemble read these unconditionally).
    if seed == cfg.primary_seed || isempty(yPredPrimary)
        yPredPrimary = yp;  yTruePrimary = yt;  attnPrimary = attn;
    end
end

% ── aggregate + DM test ─────────────────────────────────────────────────
agg = aggregate_metrics([perSeed.metrics]);
fprintf("\nCPTB over %d seeds: MAPE=%.2f±%.2f%%  RMSE=%.2f±%.2f\n", ...
        numel(cfg.seeds), agg.mean.MAPE, agg.std.MAPE, ...
        agg.mean.RMSE, agg.std.RMSE);

nMin = min(size(yPredPrimary, 1), size(npred, 1));
dm = diebold_mariano(yTruePrimary(1:nMin, :), yPredPrimary(1:nMin, :), ...
                     npred(1:nMin, :));
fprintf("DM vs SeasonalNaive: %.3f (p=%.2e)\n", dm.dm_stat, dm.p_value);

results = struct('dataset', datasetName, 'naive', mNaive, ...
                 'per_seed', perSeed, 'agg', agg, 'dm_vs_naive', dm, ...
                 'attn_primary', attnPrimary);

% causal-EC aggregate ([FIX from review]: was computed per seed but never
% aggregated nor written to the CSV)
aggEc = [];
if cfg.ec.enabled && ~isempty(fieldnames(perSeed(1).metrics_ec))
    aggEc = aggregate_metrics([perSeed.metrics_ec]);
    results.agg_ec = aggEc;
    fprintf("CPTB + causal EC: MAPE=%.2f±%.2f%%\n", ...
            aggEc.mean.MAPE, aggEc.std.MAPE);
end

% [V4.1] seed ensemble: mean of per-seed predictions (same test windows)
mEns = [];
if numel(cfg.seeds) > 1
    ypEns = mean(cat(3, ypCell{:}), 3);
    mEns = metrics_stlf(yTruePrimary, ypEns);
    results.ensemble = mEns;
    fprintf("CPTB-Ens (%d seeds): MAPE=%.2f%%  RMSE=%.2f\n", ...
            numel(cfg.seeds), mEns.MAPE, mEns.RMSE);
end

% ── ablation (skip in quick mode) ───────────────────────────────────────
if ~quickMode
    vars = {"use_stage1", "use_stage2", "use_gate", "use_patch", ...
            "use_cross_att", "use_bigru", "use_revin", "use_future_cov", ...
            "use_linear_skip", "use_cov_skip"};
    abl = struct();
    for i = 1:numel(vars)
        fprintf("\n── Ablation w/o %s ──\n", vars{i});
        ms = struct('MAE', {}, 'RMSE', {}, 'MAPE', {}, 'R2', {}, 'sMAPE', {});
        for s = 1:numel(cfg.seeds)
            rng(cfg.seeds(s));
            fl = struct(vars{i}, false);
            mdl = cptb_init(cfg, data.cfgd, data.n_cov_past, ...
                            data.n_cov_fut, fl);
            mdl = train_cptb(mdl, data, trainIdx, valIdx, cfg, false);
            [yp, yt] = predict_cptb(mdl, data, testIdx);
            ms(s) = metrics_stlf(yt, yp);
        end
        a = aggregate_metrics(ms);
        abl.(vars{i}) = a;
        fprintf("  -> MAPE=%.2f±%.2f%%\n", a.mean.MAPE, a.std.MAPE);
    end
    results.ablation = abl;
end

% ── save ────────────────────────────────────────────────────────────────
out = fullfile(cfg.results_dir, sprintf("results_v4_%s.mat", datasetName));
save(out, "results");
write_csv(cfg, datasetName, mNaive, agg, dm, aggEc, mEns);
fprintf("\nSaved: %s\n", out);
end

% ────────────────────────────────────────────────────────────────────────
function agg = aggregate_metrics(ms)
fn = fieldnames(ms(1));
for i = 1:numel(fn)
    v = arrayfun(@(x) x.(fn{i}), ms);
    agg.mean.(fn{i}) = mean(v);
    % Sample std (default, n-1) matching Python metrics_mean_std ddof=1, the
    % correct estimator for '± std over 5 seeds' (round-3 audit).
    if numel(v) > 1, agg.std.(fn{i}) = std(v); else, agg.std.(fn{i}) = 0; end
end
end

function write_csv(cfg, dsName, mNaive, agg, dm, aggEc, mEns)
fp = fullfile(cfg.results_dir, sprintf("results_v4_%s.csv", dsName));
fid = fopen(fp, "w");
fprintf(fid, "Model,MAE,RMSE,MAPE,R2,sMAPE,DM_vs_naive,p_value\n");
fprintf(fid, "SeasonalNaive,%.3f,%.3f,%.3f,%.4f,%.3f,,\n", ...
        mNaive.MAE, mNaive.RMSE, mNaive.MAPE, mNaive.R2, mNaive.sMAPE);
fprintf(fid, "CPTB (mean),%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%.2e\n", ...
        agg.mean.MAE, agg.mean.RMSE, agg.mean.MAPE, agg.mean.R2, ...
        agg.mean.sMAPE, dm.dm_stat, dm.p_value);
fprintf(fid, "CPTB (std),%.3f,%.3f,%.3f,%.4f,%.3f,,\n", ...
        agg.std.MAE, agg.std.RMSE, agg.std.MAPE, agg.std.R2, agg.std.sMAPE);
if ~isempty(aggEc)
    fprintf(fid, "CPTB+EC (mean),%.3f,%.3f,%.3f,%.4f,%.3f,,\n", ...
            aggEc.mean.MAE, aggEc.mean.RMSE, aggEc.mean.MAPE, ...
            aggEc.mean.R2, aggEc.mean.sMAPE);
    fprintf(fid, "CPTB+EC (std),%.3f,%.3f,%.3f,%.4f,%.3f,,\n", ...
            aggEc.std.MAE, aggEc.std.RMSE, aggEc.std.MAPE, ...
            aggEc.std.R2, aggEc.std.sMAPE);
end
if ~isempty(mEns)
    fprintf(fid, "CPTB-Ens,%.3f,%.3f,%.3f,%.4f,%.3f,,\n", ...
            mEns.MAE, mEns.RMSE, mEns.MAPE, mEns.R2, mEns.sMAPE);
end
fclose(fid);
end
