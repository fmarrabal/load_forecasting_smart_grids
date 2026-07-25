function [XL, XCP, XCF, Y] = get_batch(data, issueIdx)
%GET_BATCH  Extract a minibatch of windows for the given issue indices.
%
%   [XL, XCP, XCF, Y] = GET_BATCH(data, issueIdx) returns
%     XL  : L  x B        past load (z-scored, train statistics)
%     XCP : L  x C  x B   past covariates
%     XCF : H  x Cf x B   future known covariates
%     Y   : H  x B        target load (z-scored)
%
%   Mirrors: SMART_GRIDS_CODE_V4/data_utils.py (STLFDataset.__getitem__)

L = data.cfgd.input_window;
H = data.cfgd.pred_horizon;
B = numel(issueIdx);
C  = data.n_cov_past;
Cf = data.n_cov_fut;

XL  = zeros(L, B, 'single');
XCP = zeros(L, C, B, 'single');
XCF = zeros(H, Cf, B, 'single');
Y   = zeros(H, B, 'single');
for b = 1:B
    t = issueIdx(b);
    XL(:, b)     = data.load_z(t-L:t-1);
    XCP(:, :, b) = data.cov_past(t-L:t-1, :);
    XCF(:, :, b) = data.cov_fut(t:t+H-1, :);
    Y(:, b)      = data.load_z(t:t+H-1);
end
end
