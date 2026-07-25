function verify_cptb()
%VERIFY_CPTB  Machine-checkable correctness certificate for the model.
%
%   Run this after installation. It checks, on random data:
%     1. Forward shapes for hourly (168->24) and half-hourly (336->48).
%     2. Exact reconstruction: decomposition components sum to the input.
%     3. STRICT CAUSALITY: perturbing the most recent input sample changes
%        no decomposition output at earlier positions (the certificate
%        cited in Section 3.8 of the manuscript).
%     4. All nine single-flag ablation variants run (incl. use_linear_skip).
%     5. Gradients flow (one dlfeval training step decreases nothing NaN).
%
%   Mirrors: the smoke test of the Python release.

cfg = config_v4();
rng(42);

for dsName = ["GEFCom2014", "AEMO"]
    cfgd = cfg.datasets.(dsName);
    L = cfgd.input_window;  H = cfgd.pred_horizon;
    nC = 9; nCf = 9; B = 4;
    model = cptb_init(cfg, cfgd, nC, nCf);
    XL = dlarray(randn(L, B, 'single'));
    XCP = dlarray(randn(L, nC, B, 'single'));
    XCF = dlarray(randn(H, nCf, B, 'single'));

    % 1. shapes
    [Y, aw] = cptb_forward(model, XL, XCP, XCF, false);
    assert(isequal(size(Y), [H, B]), "bad output shape");
    fprintf("[%s] forward OK: y %dx%d, attn %s\n", dsName, size(Y, 1), ...
            size(Y, 2), mat2str(size(aw)));

    % 1b. [V4.1] naive-init property: with the seasonal-naive selector init
    % and the zero-initialized deep head, the untrained model must output
    % EXACTLY the weekly-persistence forecast (window positions 1..H).
    naiveErr = max(abs(extractdata(Y) - extractdata(XL(1:H, :))), [], 'all');
    assert(naiveErr < 1e-3, "naive-init property violated: %g", naiveErr);
    fprintf("[%s] naive-init OK (max dev %.2e)\n", dsName, naiveErr);

    % 2. exact reconstruction (on the raw window, gates do not affect it)
    comps = decompose_only(model, XL);
    rec = squeeze(sum(comps, 2));
    err = max(abs(extractdata(rec) - extractdata(XL)), [], 'all');
    assert(err < 1e-4, "reconstruction error %g", err);
    fprintf("[%s] reconstruction OK (max err %.2e)\n", dsName, err);

    % 3. strict causality
    XL2 = XL;  XL2(end, :) = XL2(end, :) + 100;
    c1 = extractdata(decompose_only(model, XL));
    c2 = extractdata(decompose_only(model, XL2));
    dPast = max(abs(c1(1:end-1, :, :) - c2(1:end-1, :, :)), [], 'all');
    assert(dPast < 1e-4, "NOT causal: past positions changed by %g", dPast);
    fprintf("[%s] causality OK (past change %.2e)\n", dsName, dPast);
end

% 4. ablation variants
cfgd = cfg.datasets.GEFCom2014;
vars = {"use_stage1", "use_stage2", "use_gate", "use_patch", ...
        "use_cross_att", "use_bigru", "use_revin", "use_future_cov", ...
        "use_linear_skip", "use_cov_skip"};
for i = 1:numel(vars)
    fl = struct(vars{i}, false);
    m = cptb_init(cfg, cfgd, 9, 9, fl);
    y = cptb_forward(m, dlarray(randn(168, 2, 'single')), ...
        dlarray(randn(168, 9, 2, 'single')), ...
        dlarray(randn(24, 9, 2, 'single')), false);
    assert(isequal(size(y), [24, 2]));
    fprintf("ablation w/o %s OK\n", vars{i});
end

% 5. gradient step
model = cptb_init(cfg, cfgd, 9, 9);
XL = dlarray(randn(168, 8, 'single'));
XCP = dlarray(randn(168, 9, 8, 'single'));
XCF = dlarray(randn(24, 9, 8, 'single'));
Yt = dlarray(randn(24, 8, 'single'));
[loss, grads] = dlfeval(@cptb_loss, model.p, model, XL, XCP, XCF, Yt, 1.0);
fn = fieldnames(grads);
for i = 1:numel(fn)
    assert(~any(isnan(extractdata(grads.(fn{i}))), 'all'), ...
           "NaN gradient in %s", fn{i});
end
fprintf("gradient step OK (loss %.4f, %d learnable tensors)\n", ...
        double(loss), numel(fn));
fprintf("\nALL CHECKS PASSED\n");
end

function C = decompose_only(model, XL)
%Replicates the decomposition path of cptb_forward (no RevIN, raw input).
c = model.c;  p = model.p;  f = model.flags;
[L, B] = size(XL);
z3 = reshape(XL, L, 1, B);
comps = {};
if f.use_stage1
    mas = cell(1, numel(c.kernels));
    for i = 1:numel(c.kernels)
        k = c.kernels(i);
        pad = repmat(z3(1, 1, :), k - 1, 1, 1);
        xp = cat(1, pad, z3);
        w = ones(k, 1, 1, 'single') / k;
        mas{i} = dlconv(xp, w, 0, 'DataFormat', 'SCB');
    end
    hf = z3 - mas{1};
    for i = 1:numel(mas)-1, comps{end+1} = mas{i} - mas{i+1}; end %#ok<AGROW>
    comps{end+1} = mas{end};
else
    hf = z3;
end
if f.use_stage2
    hfp = cat(1, zeros(c.ks - 1, 1, B, 'like', hf), hf);
    bands = dlconv(hfp, p.s2W, 0, 'DataFormat', 'SCB');
    rem_ = hf - sum(bands, 2);
    comps = [{bands}, {rem_}, comps];
elseif f.use_stage1
    comps = [{hf}, comps];
else
    comps = {hf};
end
C = cat(2, comps{:});
end
