#!/usr/bin/env python3
"""
Build the three canonical dataset CSVs from the raw public downloads.

None of the three sources distributes a file in the shape the pipeline reads,
so this script is the missing link between "download the data" and "run the
experiments". It writes, into the data directory:

    gefcom2014_load.csv   datetime, load, temperature
    PJME_hourly.csv       Datetime, PJME_MW      (copied through unchanged)
    aemo_nsw.csv          datetime, load

and prints the row count and date range of each, which must match the values
recorded in data/README.md — because every split boundary is a FRACTION of
the series length, a file assembled over a different date range silently
produces different train/validation/test sets and therefore different numbers.

Usage
-----
    python prepare_data.py --raw <folder with the raw downloads> [--out <data dir>]

Expected raw inputs
-------------------
GEFCom2014 : L1-train.csv from the competition archive
             (GEFCom2014 Data/GEFCom2014-L_V2.zip → Load/Task 1/L1-train.csv),
             columns ZONEID, TIMESTAMP, LOAD, w1..w25.
             TIMESTAMP is the competition's own MDDYYYY H:MM format, e.g.
             "112001 1:00" = 2001-01-01 01:00 and "10192001 17:00" =
             2001-10-19 17:00. `temperature` is the mean of the 25 weather
             stations w1..w25, the standard regional aggregate.
PJM        : PJME_hourly.csv from the Kaggle "Hourly Energy Consumption" set.
AEMO       : the monthly PRICE_AND_DEMAND_YYYYMM_NSW1.csv files from the AEMO
             aggregated-data portal (any number of them; they are concatenated,
             de-duplicated and sorted). Columns REGION, SETTLEMENTDATE,
             TOTALDEMAND, RRP.
"""
import argparse
import glob
import os
import shutil
from datetime import datetime

import pandas as pd

# Row counts of the canonical files used for every number in the paper.
EXPECTED = {
    "gefcom2014_load.csv": 50376,
    "PJME_hourly.csv": 145366,
    "aemo_nsw.csv": 87696,
}


def parse_gefcom_timestamp(ts):
    """'112001 1:00' -> datetime(2001, 1, 1, 1, 0). The month/day part has no
    separator and a variable width, so the year is taken from the last four
    characters and a two-digit month is preferred when it is a valid one."""
    ts = str(ts).strip()
    parts = ts.split(" ")
    date_str, time_str = parts[0], (parts[1] if len(parts) > 1 else "0:00")
    year = int(date_str[-4:])
    md = date_str[:-4]
    h, m = map(int, time_str.split(":"))
    if len(md) >= 3:
        m2, d2 = int(md[:2]), int(md[2:])
        if 10 <= m2 <= 12 and 1 <= d2 <= 31:
            return datetime(year, m2, d2, h, m)
    return datetime(year, int(md[0]), int(md[1:]), h, m)


def build_gefcom(raw, out):
    src = os.path.join(raw, "L1-train.csv")
    if not os.path.isfile(src):
        print(f"  [skip] GEFCom2014: {src} not found")
        return None
    df = pd.read_csv(src)
    w = [c for c in df.columns if c.startswith("w")]
    res = pd.DataFrame({
        "datetime": df["TIMESTAMP"].apply(parse_gefcom_timestamp),
        "load": df["LOAD"],
        "temperature": df[w].mean(axis=1),
    }).dropna(subset=["load"]).sort_values("datetime").reset_index(drop=True)
    dst = os.path.join(out, "gefcom2014_load.csv")
    res.to_csv(dst, index=False)
    return dst, res


def build_pjm(raw, out):
    src = os.path.join(raw, "PJME_hourly.csv")
    if not os.path.isfile(src):
        print(f"  [skip] PJM: {src} not found")
        return None
    dst = os.path.join(out, "PJME_hourly.csv")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    return dst, pd.read_csv(dst)


def build_aemo(raw, out):
    files = sorted(glob.glob(os.path.join(raw, "PRICE_AND_DEMAND_*_NSW1.csv")))
    if not files:
        print(f"  [skip] AEMO: no PRICE_AND_DEMAND_*_NSW1.csv in {raw}")
        return None
    parts = [pd.read_csv(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    res = (pd.DataFrame({
        "datetime": pd.to_datetime(df["SETTLEMENTDATE"]),
        "load": df["TOTALDEMAND"],
    })
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .reset_index(drop=True))
    dst = os.path.join(out, "aemo_nsw.csv")
    res.to_csv(dst, index=False)
    print(f"       (concatenated {len(files)} monthly files)")
    return dst, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with the raw downloads")
    ap.add_argument("--out", default=None, help="data directory (default: ../data)")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")
    os.makedirs(out, exist_ok=True)

    print(f"raw: {args.raw}\nout: {out}\n")
    ok = True
    for label, fn in (("GEFCom2014", build_gefcom), ("PJM", build_pjm),
                      ("AEMO", build_aemo)):
        r = fn(args.raw, out)
        if r is None:
            ok = False
            continue
        dst, df = r
        base = os.path.basename(dst)
        col = "datetime" if "datetime" in df.columns else "Datetime"
        span = f"{pd.to_datetime(df[col]).min()} .. {pd.to_datetime(df[col]).max()}"
        exp = EXPECTED.get(base)
        flag = "OK " if exp is None or len(df) == exp else "!! "
        if exp is not None and len(df) != exp:
            ok = False
        print(f"  [{flag}] {base:22s} rows={len(df):>7,}"
              + (f" (expected {exp:,})" if exp else "") + f"  {span}")

    print("\n" + ("All files match the published row counts." if ok else
                  "WARNING: a file is missing or its row count differs from the "
                  "published one. Because the splits are fractions of the series "
                  "length, the reported numbers will NOT reproduce."))


if __name__ == "__main__":
    main()
