"""
Data loading, LEAK-FREE preprocessing, and windowing — STLF V4.

Key differences vs V3 (each tagged [FIX-n], see config.py):
  [FIX-2] Scalers (load z-score, temperature) are fit on the TRAIN segment only.
  [FIX-4] IQR outlier bounds are fit on train; replacement is applied to the
          train segment only. Validation/test targets are NEVER modified.
  [FIX-5] Every sample carries FUTURE known covariates (calendar features for
          the forecast horizon; temperature as a perfect-forecast proxy on
          GEFCom2014, stated explicitly in the manuscript).

Sample layout (day-ahead, issue time t):
  x_load     : (L,)          past load, z-scored with train statistics
  x_cov_past : (L, C)        past covariates
  x_cov_fut  : (H, Cf)       future KNOWN covariates for the horizon
  y          : (H,)          target load, z-scored with train statistics
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR, DATASETS

PAST_COV_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                 "month_sin", "month_cos", "is_weekend", "is_holiday_proxy"]
# future covariates: calendar always known; temperature appended when available
FUT_COV_COLS = list(PAST_COV_COLS)

# Max consecutive steps filled by linear interpolation. Because that fill is
# two-sided, a gap ending at the input/target boundary could otherwise pull a
# target value into the input tail; kept windows therefore also require their
# last IMPUTE_LIMIT input steps to be observed (round-4 audit).
IMPUTE_LIMIT = 6


# ═══════════════════════════════════════════
# RAW LOADERS
# ═══════════════════════════════════════════

def _reindex_regular(df, freq):
    """Reindex to a strictly regular grid and record which timestamps carried
    a genuine observation. The boolean `observed` column ([CRITICAL-FIX,
    round-3 audit]) is later used to exclude fabricated (gap-filled) targets
    from every evaluated/trained-on window, so no synthetic value is ever
    reported as ground truth."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_idx)
    df.index.name = "datetime"
    df["observed"] = df["load"].notna().values     # before any gap filling
    return df


def load_gefcom2014():
    fp = os.path.join(DATA_DIR, "gefcom2014_load.csv")
    df = pd.read_csv(fp, parse_dates=["datetime"])
    df = df.groupby("datetime", as_index=True).mean().sort_index()
    # The raw file has ~3.8k missing hours, incl. multi-week blocks in 2010.
    return _reindex_regular(df, "h")


def load_pjm():
    fp = os.path.join(DATA_DIR, "PJME_hourly.csv")
    df = pd.read_csv(fp, parse_dates=["Datetime"])
    df = df.rename(columns={"Datetime": "datetime", "PJME_MW": "load"})
    # PJM has duplicated timestamps at DST transitions -> average them.
    df = df.groupby("datetime", as_index=True).mean().sort_index()
    return _reindex_regular(df, "h")


def load_aemo():
    fp = os.path.join(DATA_DIR, "aemo_nsw.csv")
    df = pd.read_csv(fp, parse_dates=["datetime"])
    df = df.groupby("datetime", as_index=True).mean().sort_index()
    return _reindex_regular(df, "30min")


RAW_LOADERS = {"GEFCom2014": load_gefcom2014, "PJM": load_pjm, "AEMO": load_aemo}


# ═══════════════════════════════════════════
# LEAK-FREE PREPROCESSING
# ═══════════════════════════════════════════

class TrainOnlyScaler:
    """Z-score scaler whose statistics come from the train segment only."""

    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mu = np.nanmean(x)
        self.sigma = np.nanstd(x)
        if self.sigma < 1e-12:
            self.sigma = 1.0
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mu) / self.sigma

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float64) * self.sigma + self.mu


def add_calendar_features(df):
    idx = df.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    # Cheap holiday proxy that needs no external calendar: Jan 1 / Dec 25/26.
    df["is_holiday_proxy"] = (((idx.month == 1) & (idx.day == 1)) |
                              ((idx.month == 12) & (idx.day >= 25))).astype(float)
    return df


def prepare_dataset(dataset_name, verbose=True):
    """
    Load -> impute -> split indices -> train-only outlier fix -> train-only
    scaling -> feature engineering. Returns a dict with arrays and metadata.
    """
    cfg = DATASETS[dataset_name]
    df = RAW_LOADERS[dataset_name]().copy()

    # 1) Missing values: time interpolation (short gaps), then ffill/bfill.
    df["load"] = df["load"].interpolate(method="linear", limit=IMPUTE_LIMIT)
    df["load"] = df["load"].ffill().bfill()
    if "temperature" in df.columns:
        df["temperature"] = df["temperature"].interpolate(method="linear").ffill().bfill()

    n = len(df)
    train_end = int(n * cfg["train_ratio"])
    val_end = int(n * (cfg["train_ratio"] + cfg["val_ratio"]))

    # 2) [FIX-4] Outlier bounds from TRAIN only; replace on train segment only.
    #    The rolling median is computed on the TRAIN slice alone with a
    #    trailing window (round-3 audit: a center-aligned window at the
    #    train/val boundary would peek into validation).
    train_load = df["load"].iloc[:train_end]
    q1, q3 = train_load.quantile(0.25), train_load.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    mask_train = (train_load < lo) | (train_load > hi)
    if mask_train.sum() > 0:
        med = train_load.rolling(cfg["steps_per_day"], center=False,
                                 min_periods=1).median()
        df.iloc[:train_end, df.columns.get_loc("load")] = \
            train_load.where(~mask_train, med)
        if verbose:
            print(f"  [{dataset_name}] corrected {int(mask_train.sum())} train outliers "
                  f"(test/val untouched)")

    # 3) Calendar features (deterministic, no leakage possible).
    df = add_calendar_features(df)

    # 4) [FIX-2] Train-only scalers.
    scaler_load = TrainOnlyScaler().fit(df["load"].iloc[:train_end].values)
    df["load_z"] = scaler_load.transform(df["load"].values)

    scaler_temp = None
    if "temperature" in df.columns:
        scaler_temp = TrainOnlyScaler().fit(df["temperature"].iloc[:train_end].values)
        df["temp_z"] = scaler_temp.transform(df["temperature"].values)

    past_cols = list(PAST_COV_COLS)
    fut_cols = list(FUT_COV_COLS)
    if "temp_z" in df.columns:
        past_cols.append("temp_z")
        fut_cols.append("temp_z")   # perfect-forecast proxy, stated in paper

        # [V4.1] Degree-day features: the load-temperature relation is
        # V-shaped (heating below / cooling above a comfort point). The
        # reference temperature is the TRAIN median (train-only statistic,
        # hence causal); both features are z-scored with train statistics
        # and provided to ALL models equally.
        from config import USE_DEGREE_DAYS
        if USE_DEGREE_DAYS:
            t_ref = float(df["temperature"].iloc[:train_end].median())
            hdd = np.maximum(t_ref - df["temperature"].values, 0.0)
            cdd = np.maximum(df["temperature"].values - t_ref, 0.0)
            sh = TrainOnlyScaler().fit(hdd[:train_end])
            sc = TrainOnlyScaler().fit(cdd[:train_end])
            df["hdd_z"] = sh.transform(hdd)
            df["cdd_z"] = sc.transform(cdd)
            past_cols += ["hdd_z", "cdd_z"]
            fut_cols += ["hdd_z", "cdd_z"]

    observed = df["observed"].values.astype(bool)
    if verbose:
        n_test_missing = int((~observed[val_end:]).sum())
        print(f"  [{dataset_name}] n={n}  train=[0:{train_end})  "
              f"val=[{train_end}:{val_end})  test=[{val_end}:{n})")
        print(f"  [{dataset_name}] observed={int(observed.sum())}/{n} "
              f"({100*observed.mean():.1f}%); {n_test_missing} fabricated steps "
              f"in test window will be excluded from all evaluated targets")

    return {
        "df": df,
        "load_z": df["load_z"].values.astype(np.float32),
        "load_raw": df["load"].values.astype(np.float64),
        "cov_past": df[past_cols].values.astype(np.float32),
        "cov_fut": df[fut_cols].values.astype(np.float32),
        "observed": observed,
        "index": df.index,
        "train_end": train_end,
        "val_end": val_end,
        "scaler_load": scaler_load,
        "scaler_temp": scaler_temp,
        "n_cov_past": len(past_cols),
        "n_cov_fut": len(fut_cols),
        "cfg": cfg,
        "name": dataset_name,
    }


# ═══════════════════════════════════════════
# WINDOWED DATASET
# ═══════════════════════════════════════════

class STLFDataset(Dataset):
    """
    Sliding-window dataset over a HALF-OPEN segment [seg_start, seg_end).

    A sample with issue index t uses inputs from [t-L, t) and targets [t, t+H).
    Only samples whose TARGET lies fully inside the segment belong to it; the
    input window may reach back into the previous segment (standard practice,
    it uses strictly past data so it cannot leak).
    """

    def __init__(self, data, seg_start, seg_end, stride=1,
                 require_observed_target=True):
        cfg = data["cfg"]
        self.L, self.H = cfg["input_window"], cfg["pred_horizon"]
        self.load = data["load_z"]
        self.cov_past = data["cov_past"]
        self.cov_fut = data["cov_fut"]
        first_t = max(seg_start, self.L)
        last_t = seg_end - self.H          # last valid issue index
        idx = np.arange(first_t, last_t + 1, stride, dtype=np.int64)
        # [CRITICAL-FIX, round-3 audit] Drop any issue time whose TARGET window
        # contains a fabricated (gap-filled) step, so no synthetic value is ever
        # trained on, used for early stopping/EC, or reported as ground truth.
        # [round-4 audit] Also require the last IMPUTE_LIMIT INPUT steps to be
        # observed: the two-sided interpolation could otherwise pull the first
        # target value into the input tail (a residual look-ahead leak).
        # Vectorized via a prefix count of unobserved steps: a span [a, b) is
        # fully observed iff cumulative[b] - cumulative[a] == 0.
        observed = data.get("observed")
        if require_observed_target and observed is not None and len(idx):
            miss_prefix = np.concatenate([[0], np.cumsum(~observed)])
            miss_target = miss_prefix[idx + self.H] - miss_prefix[idx]
            tail = min(IMPUTE_LIMIT, self.L)
            miss_tail = miss_prefix[idx] - miss_prefix[idx - tail]
            idx = idx[(miss_target == 0) & (miss_tail == 0)]
        self.issue_idx = idx

    def __len__(self):
        return len(self.issue_idx)

    def __getitem__(self, i):
        t = self.issue_idx[i]
        L, H = self.L, self.H
        return (
            torch.from_numpy(self.load[t - L:t].copy()),
            torch.from_numpy(self.cov_past[t - L:t].copy()),
            torch.from_numpy(self.cov_fut[t:t + H].copy()),
            torch.from_numpy(self.load[t:t + H].copy()),
            int(t),
        )


def make_loaders(data, batch_size=64, eval_stride=1):
    """Train/val/test DataLoaders with leak-free segment boundaries."""
    tr = STLFDataset(data, 0, data["train_end"])
    va = STLFDataset(data, data["train_end"], data["val_end"], stride=eval_stride)
    te = STLFDataset(data, data["val_end"], len(data["load_z"]), stride=eval_stride)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   drop_last=sh, num_workers=0)
    return mk(tr, True), mk(va, False), mk(te, False)
