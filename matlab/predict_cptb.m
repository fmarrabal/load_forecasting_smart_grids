function [yPred, yTrue, attn] = predict_cptb(model, data, issueIdx, batchSize)
%PREDICT_CPTB  Batched inference on the ORIGINAL scale.
%
%   [yPred, yTrue, attn] = PREDICT_CPTB(model, data, issueIdx) returns
%     yPred, yTrue : nForecasts x H, in original units (MW), obtained by
%                    inverting the train-only z-score scaler.
%     attn         : cross-attention weights averaged over heads,
%                    nForecasts x Tq x Tkv (or [] if disabled).
%
%   Mirrors: SMART_GRIDS_CODE_V4/train_pipeline.py (predict).

if nargin < 4, batchSize = 256; end
nF = numel(issueIdx);
H = data.cfgd.pred_horizon;
yPred = zeros(nF, H);
yTrue = zeros(nF, H);
attn = [];

for s = 1:batchSize:nF
    sel = s:min(s + batchSize - 1, nF);
    [XL, XCP, XCF, Y] = get_batch(data, issueIdx(sel));
    [Yh, aw] = cptb_forward(model, dlarray(XL), dlarray(XCP), ...
                            dlarray(XCF), false);
    yPred(sel, :) = double(extractdata(Yh))';
    yTrue(sel, :) = double(Y)';
    if ~isempty(aw)
        % attention() returns weights as (numKeys x numQueries x heads x B)
        a = double(extractdata(aw));
        a = squeeze(mean(a, 3));                    % keys x queries x B
        a = permute(a, [3 2 1]);                    % B x queries x keys
        if isempty(attn)
            attn = zeros(nF, size(a, 2), size(a, 3));
        end
        attn(sel, :, :) = a;
    end
end

% invert train-only z-score scaling -> original units
yPred = yPred * data.scaler_sigma + data.scaler_mu;
yTrue = yTrue * data.scaler_sigma + data.scaler_mu;
end
