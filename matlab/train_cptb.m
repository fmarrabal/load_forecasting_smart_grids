function [model, trainTime, history] = train_cptb(model, data, trainIdx, valIdx, cfg, verbose)
%TRAIN_CPTB  Custom training loop for the CPTB model.
%
%   Adam + PyTorch-style coupled weight decay, cosine LR annealing,
%   global-norm gradient clipping, early stopping on validation Huber loss.
%
%   Mirrors: SMART_GRIDS_CODE_V4/train_pipeline.py (train_model).

if nargin < 6, verbose = true; end
tp = cfg.train;
t0 = tic;

avgG = [];  avgSqG = [];  iter = 0;
bestVl = inf;  patience = 0;  bestP = [];   % restore only if val improved
history = struct('trainLoss', [], 'valLoss', []);

nTr = numel(trainIdx);
nBatches = floor(nTr / tp.batch_size);

for ep = 1:tp.epochs
    % cosine LR annealing
    lr = tp.min_lr + 0.5 * (tp.lr - tp.min_lr) * ...
         (1 + cos(pi * (ep - 1) / tp.epochs));

    perm = trainIdx(randperm(nTr));
    epLoss = 0;
    for b = 1:nBatches
        idx = perm((b-1)*tp.batch_size + 1 : b*tp.batch_size);
        [XL, XCP, XCF, Y] = get_batch(data, idx);
        [loss, grads] = dlfeval(@cptb_loss, model.p, model, ...
                                dlarray(XL), dlarray(XCP), dlarray(XCF), ...
                                dlarray(Y), tp.huber_delta);
        % PyTorch order: clip the RAW gradients first, then Adam adds the
        % coupled wd*p term (unclipped) inside the step.
        [grads, gn] = clip_global_norm(grads, tp.grad_clip);
        if ~isfinite(double(loss)) || ~isfinite(gn)
            continue;   % skip pathological batch, never poison parameters
        end
        grads = add_weight_decay(grads, model.p, tp.weight_decay);
        iter = iter + 1;
        [model.p, avgG, avgSqG] = adamupdate(model.p, grads, ...
                                             avgG, avgSqG, iter, lr);
        epLoss = epLoss + double(loss);
    end
    epLoss = epLoss / max(nBatches, 1);

    % validation
    vl = eval_loss(model, data, valIdx, tp);
    history.trainLoss(end+1) = epLoss;
    history.valLoss(end+1) = vl;

    if vl < bestVl - 1e-6
        bestVl = vl;  patience = 0;  bestP = model.p;
    else
        patience = patience + 1;
        if patience >= tp.patience
            if verbose, fprintf("    early stop at epoch %d\n", ep); end
            break;
        end
    end
    if verbose && mod(ep, 10) == 0
        fprintf("    epoch %3d  train=%.5f  val=%.5f\n", ep, epLoss, vl);
    end
end

if ~isempty(bestP)
    model.p = bestP;   % as in Python: keep last weights if val never improved
end
trainTime = toc(t0);
if verbose
    fprintf("    done in %.0f s, best val=%.5f\n", trainTime, bestVl);
end
end

% ────────────────────────────────────────────────────────────────────────
function vl = eval_loss(model, data, valIdx, tp)
vl = 0;  nb = 0;
for s = 1:tp.batch_size:numel(valIdx)
    idx = valIdx(s:min(s+tp.batch_size-1, numel(valIdx)));
    [XL, XCP, XCF, Y] = get_batch(data, idx);
    Yhat = cptb_forward(model, dlarray(XL), dlarray(XCP), dlarray(XCF), false);
    vl = vl + double(huber(Yhat, dlarray(Y), 'TransitionPoint', tp.huber_delta, ...
        'NormalizationFactor', 'all-elements', 'DataFormat', 'CB'));
    nb = nb + 1;
end
vl = vl / max(nb, 1);
end

function grads = add_weight_decay(grads, p, wd)
if wd <= 0, return; end
fn = fieldnames(grads);
for i = 1:numel(fn)
    grads.(fn{i}) = grads.(fn{i}) + wd * p.(fn{i});
end
end

function [grads, gn] = clip_global_norm(grads, maxNorm)
fn = fieldnames(grads);
total = 0;
for i = 1:numel(fn)
    g = extractdata(grads.(fn{i}));
    total = total + sum(double(g(:)).^2);
end
gn = sqrt(total);
if gn > maxNorm
    sc = maxNorm / (gn + 1e-6);
    for i = 1:numel(fn)
        grads.(fn{i}) = grads.(fn{i}) * sc;
    end
end
end
