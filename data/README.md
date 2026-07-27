# Datasets

All three benchmarks are public, but **none of the three sources distributes a
file in the shape the pipeline reads** — the raw downloads must be converted
first. [`python/prepare_data.py`](../python/prepare_data.py) does exactly that
and is the supported path:

```bash
# put the raw downloads (see below) in one folder, then:
cd python
python prepare_data.py --raw /path/to/raw_downloads      # writes ../data/*.csv
```

It prints the row count and date range of each file it writes and **fails
loudly if a count differs from the published one**. That check matters: every
split boundary is a *fraction of the series length*, so a file assembled over a
different date range silently produces different train/validation/test sets and
will not reproduce any number in the paper.

## What to download

| Dataset | Download | Raw shape |
|---|---|---|
| **GEFCom2014** | Competition archive → `GEFCom2014-L_V2.zip` → `Load/Task 1/L1-train.csv` | `ZONEID, TIMESTAMP, LOAD, w1…w25` — `TIMESTAMP` is the competition's `MDDYYYY H:MM` format (`112001 1:00` = 2001-01-01 01:00); the 25 `w` columns are weather stations |
| **PJM (East)** | Kaggle *Hourly Energy Consumption* → `PJME_hourly.csv` | `Datetime, PJME_MW` — used as-is |
| **AEMO (NSW)** | AEMO aggregated-data portal → monthly `PRICE_AND_DEMAND_YYYYMM_NSW1.csv` | `REGION, SETTLEMENTDATE, TOTALDEMAND, RRP` — the monthly files are concatenated |

Links:

- GEFCom2014 — see Hong et al., *Int. J. Forecasting* 32(3), 2016, for the archive
- PJM — https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- AEMO — https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data

## What the converter produces

These are the exact files every published number was computed from:

| File | Columns | Rows | Period |
|---|---|---|---|
| `gefcom2014_load.csv` | `datetime, load, temperature` | 50,376 | 2005-01-01 01:00 → 2010-12-09 23:00 |
| `PJME_hourly.csv` | `Datetime, PJME_MW` | 145,366 | 2002-01-01 01:00 → 2018-08-03 00:00 |
| `aemo_nsw.csv` | `datetime, load` | 87,696 | 2020-01-01 00:30 → 2025-01-01 00:00 |

`temperature` is the mean of the 25 GEFCom2014 weather stations, the standard
regional aggregate.

**A note on the AEMO file.** From October 2021 the AEMO portal also publishes a
**5-minute dispatch** series alongside the half-hourly trading-interval demand,
and a download can easily contain both — the archived file behind the published
numbers has 372,816 rows for exactly that reason. The two are *different
quantities*: resampled to half-hours they disagree by about 104 MW on average.
Both the converter and `data_utils.load_aemo` keep only the rows that fall on
the half-hour grid, and report how many they excluded, so a 372,816-row file
and an 87,696-row file produce the **identical** analysed series (verified
value-for-value). The row count in the table above is the converter's output.

## What the loaders then do

The loaders reindex each series to a strictly regular grid, mark which
timestamps carried a genuine observation, and **exclude from training and from
evaluation any window whose forecast horizon (or whose last six input steps)
overlaps gap-filled data** — so no synthetic value is ever trained on or
reported as ground truth. On GEFCom2014, which has multi-week gaps in 2010,
this removes 66 of 325 candidate day-ahead windows; on PJM it removes 3 and on
AEMO none. The resulting test-forecast counts are 259 / 902 / 274.
