#!/usr/bin/env python3
"""Obtain the five-seed Diebold-Mariano test of the proposed model against the
corrected TiDE.

The DM test needs both models' per-seed loss profiles in the same run, and the
released predictions only kept the primary seed, so this retrains both. From
this run onward save_all also stores every seed's forecast, so a future baseline
can be tested against the proposed model without retraining it.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
LOGS = os.path.join(CODE, "logs")
PY = sys.executable

for ds in ("GEFCom2014", "AEMO", "PJM"):
    t0 = time.time()
    print(f"\n>>> Proposed + TiDE on {ds} (for the DM test)", flush=True)
    with open(os.path.join(LOGS, f"dm_{ds}.log"), "w") as f:
        rc = subprocess.run([PY, "-u", "main.py", "--dataset", ds,
                             "--models", "Proposed,TiDE", "--no-ablation"],
                            cwd=CODE, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    print(f"<<< {ds}: exit {rc} after {(time.time()-t0)/60:.0f} min", flush=True)
print("\nDM RUN DONE", flush=True)
