#!/usr/bin/env python3
"""Run the V6 experiments one after another, resuming what is already done.

Sequential on purpose: the model is small and CPU-bound on kernel launches, so
running these concurrently on a machine that already shares its cores makes
every one of them slower without finishing any sooner.

Everything it writes — logs included — lives on the project drive. An earlier
attempt kept logs and the interpreter under %TEMP%, and a machine restart
cleared it mid-run, which lost both the evidence and the ability to tell how
far each job had got.

    python scripts/run_v6_experiments.py [--force]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from config import RESULTS_DIR                                    # noqa: E402

LOGS = os.path.join(CODE, "logs")
os.makedirs(LOGS, exist_ok=True)
PY = sys.executable


def ablation_done(ds):
    p = os.path.join(RESULTS_DIR, "summary_v4.json")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return ds in (json.load(f).get("ablations") or {})


def models_done(ds, names):
    p = os.path.join(RESULTS_DIR, "summary_v4.json")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        block = (json.load(f).get("results") or {}).get(ds, {})
    return all(n in block for n in names)


def file_done(name):
    return os.path.exists(os.path.join(RESULTS_DIR, name))


JOBS = [
    ("Ablation on AEMO (univariate)",
     ["main.py", "--dataset", "AEMO", "--ablation-only"],
     "abl_aemo.log", lambda: ablation_done("AEMO")),
    ("TiDE + Proposed on GEFCom2014",
     ["main.py", "--dataset", "GEFCom2014", "--models", "Proposed,TiDE",
      "--no-ablation"], "tide_gefcom.log",
     lambda: models_done("GEFCom2014", ["TiDE"])),
    ("TiDE + Proposed on AEMO",
     ["main.py", "--dataset", "AEMO", "--models", "Proposed,TiDE",
      "--no-ablation"], "tide_aemo.log", lambda: models_done("AEMO", ["TiDE"])),
    ("TiDE + Proposed on PJM",
     ["main.py", "--dataset", "PJM", "--models", "Proposed,TiDE",
      "--no-ablation"], "tide_pjm.log", lambda: models_done("PJM", ["TiDE"])),
    ("Rolling-origin on PJM",
     ["scripts/rolling_origin.py", "--dataset", "PJM", "--origins", "3",
      "--seeds", "42", "43", "--models", "Proposed,PatchTST,DLinear,TiDE"],
     "rolling_pjm.log", lambda: file_done("rolling_origin_PJM.json")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-run jobs whose output already exists")
    args = ap.parse_args()

    print(f"interpreter: {PY}\nlogs: {LOGS}\n", flush=True)
    for label, cmd, log, done in JOBS:
        if done() and not args.force:
            print(f"--- {label}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"\n>>> {label}", flush=True)
        with open(os.path.join(LOGS, log), "w") as f:
            rc = subprocess.run([PY, "-u"] + cmd, cwd=CODE, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        print(f"<<< {label}: exit {rc} after {(time.time() - t0) / 60:.0f} min",
              flush=True)
        if rc != 0:
            print(f"    FAILED — see logs/{log}; continuing", flush=True)

    print("\nQUEUE DONE", flush=True)


if __name__ == "__main__":
    main()
