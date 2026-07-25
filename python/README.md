# STLF V4 — Leak-free pipeline (Applied Energy manuscript V4)

**Paper:** Adaptive dual-stage signal decomposition and Patch Transformer–BiGRU
with cross-attention for short-term load forecasting in smart grids (V4).

**Author:** Francisco M. Arrabal-Campos — CIAIMBITAL, Universidad de Almería.

## Why V4 exists

The V3 pipeline (in `../SMART_GRIDS_CODE`) applied CEEMDAN–SE–VMD to the
**full series before the train/test split**. EMD-family and VMD transforms are
global: every training sub-signal sample contained information from the test
period, which inflated the reported improvements (45–60 % over baselines) far
beyond what is achievable operationally. This is the documented
"decompose-then-split" pitfall (Quilty & Adamowski 2018; Yang et al. 2024;
VMDNet 2025). V4 fixes it — and every other issue found in the audit — while
keeping the model identity: adaptive dual-stage decomposition + Patch
Transformer + cross-attention + BiGRU.

| # | V3 problem | V4 fix |
|---|---|---|
| 1 | CEEMDAN/VMD on full series (leak) | Causal, in-model, learnable dual-stage decomposition per input window |
| 2 | Scalers fit on full data | Train-only z-score + RevIN per window |
| 3 | Error correction read test ground truth | Causal EC: only errors of already-realized forecasts |
| 4 | EC trained on shuffled/misaligned train predictions | EC trained on chronological validation errors |
| 5 | "w/o VMD" ablation ran the full model (fake row) | Every ablation variant differs in exactly one component |
| 6 | Component ablations dropped decomposition too (confounded) | Single-flag ablations on the same architecture |
| 7 | Test ground truth modified by IQR replacement | Outlier fix on train only; test never touched |
| 8 | No future covariates | Future calendar (+ temperature proxy) fed to ALL models |
| 9 | ARIMA strawman (no seasonality, 168-pt history) | Seasonal ARIMA, 4-week history, refit per day; + SeasonalNaive floor |
| 10 | Single seed, no significance tests | 5 seeds, mean±std, Diebold–Mariano (HLN-corrected) |
| 11 | Fig 7/10 rendered from hardcoded/random data | All figures generated from actual result files only |
| 12 | Overlapping stride-1 test windows | Operational protocol: one forecast per day (stride = horizon) |
| 13 | 30 models per dataset (~4.7 M params total) | ONE model per dataset (~0.22 M params), multi-seed affordable |

## V4.1 upgrades

* **Per-component linear base** (`use_linear_skip`): each causal component
  gets its own direct linear map window→horizon, initialized as the
  seasonal-naive selector; the deep branch (zero-initialized final layer)
  learns residual corrections. The untrained model therefore outputs exactly
  the weekly-persistence forecast — a machine-checked property (smoke test /
  `verify_cptb.m`). This grafts DLinear's strength onto the 8-band causal
  decomposition.
* **Degree-day covariates** (`USE_DEGREE_DAYS`): HDD/CDD around the
  train-median temperature (causal), given to ALL models equally.
* **Seed ensemble**: the mean of the 5 seeds' predictions is reported as an
  additional `Proposed-Ens` row (honest variance reduction; the cost is
  already paid by the multi-seed protocol).

## Structure

```
SMART_GRIDS_CODE_V4/
├── config.py            # hyperparameters, seeds, dataset configs
├── data_utils.py        # leak-free loading/preprocessing/windowing
├── model_proposed.py    # CPTB: causal dual-stage decomp + PatchTransformer
│                        #       + cross-attention + BiGRU (+ causal EC)
├── models_baselines.py  # SeasonalNaive, LSTM/BiLSTM/GRU/TCN/Transformer,
│                        # CNN-LSTM, GRU-TCN-Att, DLinear, PatchTST
├── train_pipeline.py    # training, prediction, causal EC, GBM/ARIMA runners
├── metrics_stats.py     # metrics + Diebold-Mariano test
├── main.py              # orchestration (multi-seed, DM, ablation)
├── leakage_demo.py      # quantifies the V3 leakage (Section 5.4)
├── figures_tables.py    # all tables/figures from ACTUAL results
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# validate the code end-to-end in minutes (tiny epochs):
python main.py --smoke

# one dataset, one seed, reduced baselines:
python main.py --quick --dataset GEFCom2014

# full protocol: 3 datasets x all models x 5 seeds + ablation + DM tests
python main.py

# leakage quantification experiment (Section 5.4 / Fig. 10):
python leakage_demo.py --dataset GEFCom2014

# regenerate tables/figures from saved results:
python main.py --figures-only
```

Datasets are read from `../SMART_GRIDS_CODE/data` by default
(`gefcom2014_load.csv`, `PJME_hourly.csv`, `aemo_nsw.csv`); override with the
environment variable `STLF_DATA_DIR`.

## Evaluation protocol (what reviewers will check)

* Chronological 70/15/15 split; **split indices computed before any fitting**.
* All statistics (scalers, outlier bounds) fit on train only.
* Test evaluated in the **operational day-ahead setting**: one forecast per
  day, issued at 00:00, using only information available at issue time.
* Metrics on the original MW scale against **unmodified** targets.
* 5 seeds (42–46), mean ± std; Diebold–Mariano vs every baseline with
  Harvey–Leybourne–Newbold correction.
* Baselines include SeasonalNaive (weekly), DLinear and PatchTST.
