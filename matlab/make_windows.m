function issueIdx = make_windows(data, segStart, segEnd, stride)
%MAKE_WINDOWS  Issue-time indices for a segment (mirrors STLFDataset).
%
%   issueIdx = MAKE_WINDOWS(data, segStart, segEnd, stride) returns the
%   1-based issue indices t such that the input window [t-L, t-1] is fully
%   available and the target [t, t+H-1] lies inside (segStart, segEnd].
%   Train uses stride 1; val/test use stride = H (operational day-ahead,
%   non-overlapping forecasts).
%
%   Mirrors: SMART_GRIDS_CODE_V4/data_utils.py (STLFDataset)

L = data.cfgd.input_window;
H = data.cfgd.pred_horizon;
firstT = max(segStart, L + 1);       % need L past samples
lastT  = segEnd - H + 1;             % target must fit inside the segment
issueIdx = (firstT:stride:lastT)';

% [CRITICAL-FIX, round-3 audit] Drop issue times whose TARGET window
% [t, t+H-1] contains a fabricated step; [round-4] also require the last
% IMPUTE_LIMIT input steps observed (two-sided interpolation could pull a
% target value into the input tail). Mirrors STLFDataset.
if isfield(data, 'observed')
    IMPUTE_LIMIT = 6;
    tail = min(IMPUTE_LIMIT, L);
    missPrefix = [0; cumsum(~data.observed(:))];
    missInTarget = missPrefix(issueIdx + H) - missPrefix(issueIdx);
    missInTail = missPrefix(issueIdx) - missPrefix(issueIdx - tail);
    issueIdx = issueIdx(missInTarget == 0 & missInTail == 0);
end
end
