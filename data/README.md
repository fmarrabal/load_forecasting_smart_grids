# Datasets

All three benchmarks are publicly available. Download them into this folder
(`data/`) with the file names below, or point the code at another location with
the `STLF_DATA_DIR` environment variable. The loaders reindex each series to a
strictly regular grid, mark genuinely observed timestamps, and exclude any
window overlapping fabricated (gap-filled) data — so no synthetic value is ever
trained on or reported as ground truth.

| Dataset | File name | Resolution | Covariate | Source |
|---|---|---|---|---|
| GEFCom2014 | `gefcom2014_load.csv` | hourly | temperature | Global Energy Forecasting Competition 2014 archive |
| PJM (East) | `PJME_hourly.csv` | hourly | — | Kaggle: *Hourly Energy Consumption* (robikscube) |
| AEMO (NSW) | `aemo_nsw.csv` | half-hourly | — | AEMO aggregated price & demand data portal |

Expected columns:

- `gefcom2014_load.csv`: `datetime, load, temperature`
- `PJME_hourly.csv`: `Datetime, PJME_MW`
- `aemo_nsw.csv`: `datetime, load`

Links:

- GEFCom2014: https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip (competition archive; see Hong et al., *Int. J. Forecasting*, 2016)
- PJM: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- AEMO: https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data
