function [loss, grads] = cptb_loss(p, model, XL, XCP, XCF, Y, delta)
%CPTB_LOSS  Loss + gradients for dlfeval (Huber, PyTorch 'mean' semantics).
%
%   [loss, grads] = dlfeval(@cptb_loss, model.p, model, XL, XCP, XCF, Y, 1.0)
%
%   The learnable struct p is passed separately (first argument) so that
%   dlgradient can trace it; `model.p` is overwritten with it inside.
%
%   Mirrors: SMART_GRIDS_CODE_V4/train_pipeline.py (train_model loss).

model.p = p;
Yhat = cptb_forward(model, XL, XCP, XCF, true);
loss = huber(Yhat, Y, 'TransitionPoint', delta, ...
             'NormalizationFactor', 'all-elements', 'DataFormat', 'CB');
grads = dlgradient(loss, p);
end
