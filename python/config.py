"""
Configuration and hyperparameters — STLF V4 (leak-free pipeline).

Paper: Adaptive dual-stage signal decomposition and Patch Transformer-BiGRU
       with cross-attention for short-term load forecasting in smart grids (V4)
Target: Applied Energy (Elsevier)

V4 changes vs V3 (all methodological fixes are marked [FIX]):
  [FIX-1] Decomposition is CAUSAL and computed inside the model per input
          window (no CEEMDAN/VMD over the full series -> no look-ahead leakage).
  [FIX-2] All scalers are fit on the TRAINING segment only; the proposed model
          additionally uses RevIN (per-window reversible instance normalization).
  [FIX-3] Error correction uses only errors observable at forecast issue time.
  [FIX-4] Outlier treatment is fit on train and NEVER modifies val/test targets.
  [FIX-5] Future known covariates (calendar; temperature proxy where available)
          are provided to ALL models for the forecast horizon.
  [FIX-6] Multi-seed protocol + Diebold-Mariano significance testing.
"""
import os
import torch

# ═══════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
# Datasets live in <repo>/data/ (see data/README.md for download links).
# Override with the STLF_DATA_DIR environment variable if stored elsewhere.
DATA_DIR = os.environ.get("STLF_DATA_DIR", os.path.join(REPO_DIR, "data"))
RESULTS_DIR = os.path.join(REPO_DIR, "results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")
MODELS_DIR = os.path.join(REPO_DIR, "models_saved")

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════
# DEVICE & REPRODUCIBILITY
# ═══════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44, 45, 46]      # [FIX-6] multi-seed protocol
PRIMARY_SEED = 42

# ═══════════════════════════════════════════
# DATASET CONFIGURATIONS
# ═══════════════════════════════════════════
DATASETS = {
    "GEFCom2014": {
        "resolution": "hourly",
        "steps_per_day": 24,
        "input_window": 168,      # 1 week
        "pred_horizon": 24,       # day-ahead
        "patch_length": 24,       # 1 day per patch
        "has_temperature": True,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
    "PJM": {
        "resolution": "hourly",
        "steps_per_day": 24,
        "input_window": 168,
        "pred_horizon": 24,
        "patch_length": 24,
        "has_temperature": False,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
    "AEMO": {
        "resolution": "half-hourly",
        "steps_per_day": 48,
        "input_window": 336,      # 1 week of half-hours
        "pred_horizon": 48,       # day-ahead
        "patch_length": 48,
        "has_temperature": False,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
}

# ═══════════════════════════════════════════
# V4.1 FEATURES
# ═══════════════════════════════════════════
# Degree-day covariates (HDD/CDD around the train-median temperature) on
# datasets with temperature; provided to ALL models equally.
USE_DEGREE_DAYS = True

# ═══════════════════════════════════════════
# CAUSAL ADAPTIVE DUAL-STAGE DECOMPOSITION  [FIX-1]
# Stage 1: adaptive multi-kernel causal moving-average trend/seasonal split
# Stage 2: learnable causal band-pass FIR filter bank on the HF residual,
#          with entropy-guided adaptive gating (causal analogue of SE+VMD)
# ═══════════════════════════════════════════
DECOMP_PARAMS = {
    # Stage 1 kernel sizes are expressed in DAYS and scaled by steps_per_day,
    # e.g. for hourly data: [12, 24, 168] -> half-day, day, week averages.
    "stage1_kernels_hours": [12, 24, 168],
    # Stage 2: number of learnable band-pass filters applied to the residual
    "stage2_n_bands": 4,
    "stage2_kernel_size": 25,     # FIR length (causal, left-padded)
    # Entropy-guided gating (adaptive weighting of components)
    "gate_hidden": 16,
}

# ═══════════════════════════════════════════
# MODEL HYPERPARAMETERS — Proposed CPTB
# (Causal-decomposition Patch Transformer BiGRU with cross-attention)
# ═══════════════════════════════════════════
MODEL_PARAMS = {
    "d_model": 64,
    "n_heads": 4,
    "n_encoder_layers": 2,
    "dim_feedforward": 128,
    "dropout_transformer": 0.1,
    "bigru_hidden": 64,
    "bigru_layers": 1,
    "dropout_bigru": 0.2,
    "mlp_hidden": 128,
    "dropout_mlp": 0.2,
    "cross_att_heads": 2,
    # future-covariate patching: horizon is split into patches of this length
    "future_patch_length_hours": 6,
}

# ═══════════════════════════════════════════
# CAUSAL ERROR CORRECTION  [FIX-3]
# Trained on VALIDATION errors; at test time it only sees errors of past
# forecasts whose targets are fully realized at issue time (lag >= horizon).
# ═══════════════════════════════════════════
ERROR_CORRECTION = {
    "enabled": True,          # evaluated as an ablation; honest reporting
    "hidden_size": 32,
    "input_window_hours": 168,  # past observed errors fed to the EC model
    "epochs": 50,
    "lr": 1e-3,
}

# ═══════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════
TRAIN_PARAMS = {
    "epochs": 60,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 10,
    "huber_delta": 1.0,
    "min_lr": 1e-6,
    "grad_clip": 1.0,
}

# ═══════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════
BASELINES = [
    "SeasonalNaive",        # weekly persistence — the honest floor
    "ARIMA",
    "XGBoost",
    "LightGBM",
    "LSTM",
    "BiLSTM",
    "GRU",
    "TCN",
    "Transformer",
    "CNN_LSTM",
    "GRU_TCN_Attention",
    "DLinear",              # strong linear SOTA baseline
    "PatchTST",             # strong transformer SOTA baseline
]

METRICS = ["MAE", "RMSE", "MAPE", "R2", "sMAPE"]

# ═══════════════════════════════════════════
# FIGURE SETTINGS (Applied Energy style)
# ═══════════════════════════════════════════
FIG_PARAMS = {
    "dpi": 300,
    "font_family": "serif",
    "font_size": 10,
    "title_size": 12,
    "label_size": 11,
    "tick_size": 9,
    "legend_size": 9,
    "line_width": 1.5,
    "colors": {
        "proposed": "#D62728",
        "actual": "#1F77B4",
        "baseline1": "#2CA02C",
        "baseline2": "#FF7F0E",
        "baseline3": "#9467BD",
        "hf_group": "#FF6B6B",
        "lf_group": "#4ECDC4",
    },
}
