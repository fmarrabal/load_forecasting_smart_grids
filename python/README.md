# Python implementation

Reference PyTorch implementation of the leak-free causal STLF framework. See the
[root README](../README.md) for the scientific context and headline results.

## Install

```bash
pip install -r requirements.txt
```

`torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `xgboost`,
`lightgbm`, `statsmodels` (and `EMD-signal` for the leakage experiment).

## Run

```bash
python verify_cptb.py                 # machine-checked causality certificate (no data needed)
python main.py --smoke                # tiny end-to-end validation
python main.py                        # full: 3 datasets x 14 models x 5 seeds + ablation + DM
python main.py --dataset PJM          # one dataset
python main.py --figures-only         # regenerate tables/figures from saved results
python leakage_demo.py --dataset GEFCom2014   # decompose-then-split leakage quantification
```

Datasets are read from `../data/` (see [`../data/README.md`](../data/README.md));
override with the `STLF_DATA_DIR` environment variable. Outputs are written to
`../results/`, `../figures/` and `../models_saved/`. `main.py` runs a preflight
dependency check and aborts early if a requested baseline's package is missing.

## Files

| File | Purpose |
|---|---|
| `config.py` | Hyperparameters, dataset configs, seeds, paths |
| `data_utils.py` | Leak-free loading/preprocessing, observed-data masking, windowing |
| `model_proposed.py` | The proposed **CPTB** model: causal dual-stage decomposition, patch Transformer, cross-attention, BiGRU, linear base, full-resolution covariate path, RevIN |
| `models_baselines.py` | 13 baselines: seasonal-naïve, DLinear, PatchTST, LSTM/BiLSTM/GRU, TCN, Transformer, CNN-LSTM, GRU-TCN-Attention |
| `train_pipeline.py` | Training loop, causal error correction, XGBoost/LightGBM/ARIMA |
| `metrics_stats.py` | MAE/RMSE/MAPE/R²/sMAPE + Diebold–Mariano (Newey-West HAC, Harvey–Leybourne–Newbold, Holm–Bonferroni) |
| `figures_tables.py` | All tables and figures — generated only from result files |
| `leakage_demo.py` | Controlled decompose-then-split leakage experiment |
| `verify_cptb.py` | Correctness certificate: shapes, exact reconstruction, strict causality, naive-init, all ablations, gradient flow |
| `main.py` | Orchestration |

## Key design choices

- **In-model causal decomposition** replaces the offline CEEMDAN–SE–VMD chain, so no
  global transform sees future data. Reconstruction is exact and causality is
  machine-checked.
- **Full-resolution covariate path**: on temperature datasets, the hourly future
  covariates are mapped directly to the horizon by a zero-initialized additive head,
  enabled only when a strong exogenous covariate is present.
- **Operational evaluation**: one forecast per horizon (non-overlapping), train-only
  statistics, unmodified test targets, five seeds, Holm-corrected DM tests.
