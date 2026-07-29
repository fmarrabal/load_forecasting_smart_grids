#!/usr/bin/env python3
"""
Machine-checkable correctness certificate for the proposed CPTB model
(Python mirror of MATLAB_CODE_V4/verify_cptb.m). Run:  python verify_cptb.py

Checks, on random data and on both temporal resolutions:
  1. Forward shapes for hourly (168->24) and half-hourly (336->48).
  2. Exact reconstruction: decomposition components sum to the input.
  3. STRICT CAUSALITY: perturbing the most recent input sample changes no
     decomposition output at earlier positions (the certificate cited in
     Section 3.9 of the manuscript).
  4. NAIVE-INIT: an untrained full model outputs exactly the weekly-persistence
     forecast (Section 3.6).
  5. All single-flag ablation variants instantiate and run.
  6. Gradients are finite through one loss/backward step.
"""
import numpy as np
import sys
import torch

from config import DATASETS, MODEL_PARAMS, DECOMP_PARAMS
from model_proposed import CPTB, create_proposed

ABLATION_FLAGS = ["use_stage1", "use_stage2", "use_gate", "use_patch",
                  "use_cross_att", "use_bigru", "use_revin", "use_future_cov",
                  "use_linear_skip", "use_cov_skip"]


def _decompose(model, xl):
    # Test the DECOMPOSITION operator (causal filters) on the raw window,
    # matching MATLAB verify_cptb.m/decompose_only. RevIN is a window-global
    # normalization (mean/std over the whole past window) — legitimately
    # causal w.r.t. the forecast but not a position-local filter, so it is
    # excluded from the position-wise causality certificate.
    return model.decompose(xl), xl


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    n_cov = 11

    for ds in ("GEFCom2014", "AEMO"):
        cfg = DATASETS[ds]
        L, H, B = cfg["input_window"], cfg["pred_horizon"], 4
        model = create_proposed(cfg, MODEL_PARAMS, DECOMP_PARAMS, n_cov, n_cov)
        model.eval()
        xl = torch.randn(B, L)
        xc = torch.randn(B, L, n_cov)
        xf = torch.randn(B, H, n_cov)

        with torch.no_grad():
            y, attn = model(xl, xc, xf)
            assert y.shape == (B, H), f"bad output shape {y.shape}"
            print(f"[{ds}] forward OK: y {tuple(y.shape)}, "
                  f"attn {tuple(attn.shape)}")

            # naive-init: untrained model == weekly persistence (window[:H])
            dev = (y - xl[:, :H]).abs().max().item()
            assert dev < 1e-3, f"naive-init property violated: {dev}"
            print(f"[{ds}] naive-init OK (max dev {dev:.2e})")

            # exact reconstruction
            comps, z = _decompose(model, xl)
            rec = (comps.sum(dim=1) - z).abs().max().item()
            assert rec < 1e-4, f"reconstruction error {rec}"
            print(f"[{ds}] reconstruction OK (max err {rec:.2e})")

            # strict causality
            xl2 = xl.clone(); xl2[:, -1] += 100.0
            c1, _ = _decompose(model, xl)
            c2, _ = _decompose(model, xl2)
            dpast = (c1 - c2).abs()[:, :, :-1].max().item()
            assert dpast < 1e-4, f"NOT causal: past changed by {dpast}"
            print(f"[{ds}] causality OK (past change {dpast:.2e})")

    # ablation variants
    cfg = DATASETS["GEFCom2014"]
    L, H = cfg["input_window"], cfg["pred_horizon"]
    xl, xc, xf = torch.randn(2, L), torch.randn(2, L, n_cov), torch.randn(2, H, n_cov)
    for flag in ABLATION_FLAGS:
        m = create_proposed(cfg, MODEL_PARAMS, DECOMP_PARAMS, n_cov, n_cov,
                            **{flag: False})
        m.eval()
        with torch.no_grad():
            yv, _ = m(xl, xc, xf)
        assert yv.shape == (2, H)
        print(f"ablation w/o {flag} OK")

    # gradient step
    model = create_proposed(cfg, MODEL_PARAMS, DECOMP_PARAMS, n_cov, n_cov)
    xl, xc, xf = torch.randn(8, L), torch.randn(8, L, n_cov), torch.randn(8, H, n_cov)
    yt = torch.randn(8, H)
    yp, _ = model(xl, xc, xf)
    loss = torch.nn.functional.huber_loss(yp, yt)
    loss.backward()
    n_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite gradient"
            n_grad += 1
    print(f"gradient step OK (loss {loss.item():.4f}, {n_grad} grad tensors)")
    print("\nALL CHECKS PASSED")


def check_covariate_sensitivity(steps=40):
    """Every model that is GIVEN future covariates must actually use them.

    [V6] Two silent failures in the TiDE baseline motivated this check. Its
    temporal decoder ended in a LayerNorm over a SIZE-1 output, which is
    identically zero, so the whole encoder-decoder branch emitted zeros and the
    model collapsed to its linear residual from the lookback. Its covariate
    projection also ended in a LayerNorm, which made the projection's scale
    unobservable to the loss so weight decay drove it to 1.5e-39. The model
    trained, converged, and reported plausible numbers while ignoring the
    covariates completely — it would have entered the paper as a
    covariate-native baseline that lost.

    Sensitivity is measured AFTER a few optimisation steps, not at
    initialization: the proposed model and DLinear zero-initialize their
    covariate branches on purpose so that training starts at the
    seasonal-naive solution, and an init-time test flags that legitimate design
    as dead. What must not survive training is a path that cannot come alive.
    """
    import torch
    from config import DATASETS
    from train_pipeline import _make_dl_model, set_seed

    ds = "GEFCom2014"
    cfg = DATASETS[ds]
    L, H, ncp, ncf = cfg["input_window"], cfg["pred_horizon"], 11, 11
    fake = {"cfg": cfg, "n_cov_past": ncp, "n_cov_fut": ncf, "name": ds}
    names = ["Proposed", "TiDE", "DLinear", "PatchTST", "LSTM", "GRU", "TCN",
             "Transformer", "CNN_LSTM", "BiLSTM", "GRU_TCN_Attention"]
    g = torch.Generator().manual_seed(0)
    xl = torch.randn(32, L, generator=g)
    xc = torch.randn(32, L, ncp, generator=g)
    xf = torch.randn(32, H, ncf, generator=g)
    # a target that DEPENDS on the future covariates, so a live path has a
    # reason to learn to use them
    y = xl[:, -H:] + 2.0 * xf[:, :, 0]

    bad = []
    for name in names:
        set_seed(0)
        m = _make_dl_model(name, fake, 0).cpu().train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.mse_loss(m(xl, xc, xf)[0], y).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            a = m(xl, xc, xf)[0]
            b = m(xl, xc, torch.zeros_like(xf))[0]
        rel = (a - b).abs().mean().item() / (a.std().item() + 1e-12)
        ok = rel > 1e-3
        print(f"  {name:20s} sensitivity after {steps} steps = {rel:8.4f}"
              f"  {'OK' if ok else 'DEAD'}")
        if not ok:
            bad.append(name)
    if bad:
        raise SystemExit(f"FAIL: these models cannot use their future "
                         f"covariates: {bad}")
    print("  every model responds to its future covariates")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
    print("\n-- covariate sensitivity --")
    check_covariate_sensitivity()
