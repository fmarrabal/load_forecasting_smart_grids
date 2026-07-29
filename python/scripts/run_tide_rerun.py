#!/usr/bin/env python3
"""Re-run TiDE on all three benchmarks after the LayerNorm fixes.

The first TiDE run is withdrawn: a LayerNorm over its size-1 temporal-decoder
output made that branch emit exact zeros, so the model collapsed to a linear map
of the lookback and ignored the covariates entirely. Both offending norms are
removed and verify_cptb.py now fails if any model cannot use its future
covariates. Waits for the rolling-origin job so the two do not contend.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
LOGS = os.path.join(CODE, "logs")
PY = sys.executable


def busy():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\")"
         ".CommandLine -join \"`n\""],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:200])
    return "rolling_origin.py" in out.stdout


print("waiting for the rolling-origin job...", flush=True)
while busy():
    time.sleep(60)

for ds in ("GEFCom2014", "AEMO", "PJM"):
    t0 = time.time()
    print(f"\n>>> TiDE (fixed) on {ds}", flush=True)
    with open(os.path.join(LOGS, f"tide2_{ds}.log"), "w") as f:
        rc = subprocess.run([PY, "-u", "main.py", "--dataset", ds,
                             "--models", "TiDE", "--no-ablation"],
                            cwd=CODE, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    print(f"<<< {ds}: exit {rc} after {(time.time()-t0)/60:.0f} min", flush=True)

print("\nTIDE RERUN DONE", flush=True)
