# STLF V4 — MATLAB implementation

MATLAB port of the leak-free STLF V4 framework (see `../SMART_GRIDS_CODE_V4`
for the reference PyTorch implementation and `../AppliedEnergy_Manuscript_v4.docx`).

## Requirements

* **MATLAB R2022b or newer** (needs the `attention` and `gelu` deep learning
  functions; R2023b+ recommended).
* **Deep Learning Toolbox** (dlarray, dlfeval, dlgradient, adamupdate, gru,
  attention, layernorm, fullyconnect, dlconv, huber).
* No other toolboxes: quantiles, rolling medians and the DM test p-value
  (via `betainc`) are implemented with base MATLAB.
* GPU optional (`cfg.train.use_gpu`); CPU works.

## Quick start

```matlab
cd MATLAB_CODE_V4

% 1. Correctness certificate (shapes, exact reconstruction, causality,
%    ablation variants, gradient flow) — seconds:
verify_cptb

% 2. Installation check (2 epochs, 1 seed) — minutes:
results = main_v4('GEFCom2014', true);

% 3. Full protocol (5 seeds + ablation + DM test):
results = main_v4('GEFCom2014');
```

Datasets are read from `../SMART_GRIDS_CODE/data/` (same CSVs as the Python
version); adjust `cfg.data_dir` in `config_v4.m` if needed.

## File map (→ Python reference)

| MATLAB | Python | Purpose |
|---|---|---|
| `config_v4.m` | `config.py` | All hyperparameters, seeds, dataset configs |
| `prepare_dataset.m` | `data_utils.py` | Leak-free loading/preprocessing |
| `make_windows.m`, `get_batch.m` | `data_utils.py` (STLFDataset) | Windowing |
| `cptb_init.m` | `model_proposed.py` | Parameter initialization (+ablation flags) |
| `cptb_forward.m` | `model_proposed.py` (CPTB.forward) | Full forward pass |
| `cptb_loss.m` | `train_pipeline.py` | Huber loss + gradients (dlfeval) |
| `train_cptb.m` | `train_pipeline.py` (train_model) | Adam/cosine/clip/early-stop |
| `predict_cptb.m` | `train_pipeline.py` (predict) | Batched inference, MW scale |
| `metrics_stlf.m` | `metrics_stats.py` | MAE/RMSE/MAPE/R²/sMAPE |
| `diebold_mariano.m` | `metrics_stats.py` | DM test (HLN + HAC) |
| `error_correction.m` | `train_pipeline.py` (EC functions) | Causal EC |
| `seasonal_naive.m` | `models_baselines.py` | Weekly-persistence floor |
| `main_v4.m` | `main.py` | Orchestration, multi-seed, ablation |
| `verify_cptb.m` | smoke test | Machine-checkable causality certificate |

## Scope notes

* This port covers the **proposed model, its eight single-flag ablations,
  the SeasonalNaive floor, the causal error correction and the DM test**.
  The extended baseline suite (ARIMA, XGBoost/LightGBM, LSTM family,
  DLinear, PatchTST) and the leakage-quantification experiment live in the
  Python release, which is the reference for the paper's Tables 3–5 and
  Section 5.4.
* Known intentional deviations from PyTorch (documented in code):
  GRU weights use Glorot instead of PyTorch's uniform init; the future-
  covariate patch flattening orders (time, channel) instead of (channel,
  time) — both are equivalent up to a fixed permutation of a learned
  linear layer's inputs.
* `attention()` returns weights as (keys × queries × heads × batch);
  `predict_cptb.m` averages heads and permutes to (batch × queries × keys)
  to match the Python convention.
