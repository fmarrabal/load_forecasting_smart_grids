# -*- coding: utf-8 -*-
"""Cross-check EVERY numeric cell of the manuscript's result, ablation,
ensemble and efficiency tables against the result files. Prints a defect for
any cell that does not round-trip."""
import io, json, re, sys, pickle
import numpy as np
from docx import Document

ROOT = r"e:\ARTICULOS-CIENTIFICOS\20260318_load_forecasting_smart_grids"
import os as _o
DOC = _o.environ.get("STLF_DOCX_OUT", ROOT + r"\AppliedEnergy_Manuscript_v6.docx")
RES = ROOT + r"\SMART_GRIDS_CODE_V4\results"

saved = json.load(open(RES + r"\summary_v4.json"))
res, abl = saved["results"], saved["ablation"]

ALIAS = {"Proposed (CPTB)": "Proposed", "Proposed-Ens (CPTB)": "Proposed@ENS",
         "Proposed": "Proposed", "Proposed-Ens": "Proposed@ENS",
         "ARIMA (seasonal)": "ARIMA", "Seasonal-naïve": "SeasonalNaive",
         "Seasonal-naive": "SeasonalNaive", "GRU-TCN-Attention": "GRU_TCN_Attention",
         "GRU-TCN-Att.": "GRU_TCN_Attention", "CNN-LSTM": "CNN_LSTM"}

def key(name):
    # the tables carry DM significance marks in the model name; they are
    # presentational and must come off before the results lookup
    n = name.strip().rstrip("*").strip().replace("-Ens", "@ENS")
    base = n.replace("@ENS", "")
    base = ALIAS.get(base, ALIAS.get(name.strip(), base.replace("-", "_")))
    return base + ("@ENS" if "@ENS" in n else "")

def num(cell):
    """First number in a cell like '6.51 ± 0.27' / '+2.18' / '<0.001'."""
    t = cell.replace("−", "-").replace("–", "-").replace(",", "").strip()
    if t.startswith("<"):
        return None
    m = re.match(r"[+-]?\d+(?:\.\d+)?", t)
    return float(m.group()) if m else None

def std(cell):
    t = cell.replace("±", "|").replace(",", "")
    return float(t.split("|")[1]) if "|" in t else None

bad, tot = [], 0
def chk(label, got, want, tol):
    global tot
    if got is None or want is None:
        return
    tot += 1
    if abs(got - want) > tol:
        bad.append(f"{label}: doc {got} vs results {want:.4f}")

doc = Document(DOC)

def _hdr(t):
    return [c.text.strip() for c in t.rows[0].cells]

# Locate tables by their HEADER, not by position: inserting a table anywhere
# earlier in the document silently reindexed everything when this was
# positional, and the checker then compared the wrong table to the wrong data.
_all = doc.tables
_res = [t for t in _all if _hdr(t)[:2] == ["Model", "MAE"]]
_abl = [t for t in _all if _hdr(t)[0] == "Variant"]
_dm = [t for t in _all if _hdr(t)[0] == "Baseline"]
_ens = [t for t in _all if _hdr(t)[0] == "Rank"]
_eff = [t for t in _all if _hdr(t)[:2] == ["Model", "Parameters"]]
for nm, lst, k in (("result", _res, 3), ("ablation", _abl, len(_abl)),
                   ("DM", _dm, 1), ("ensemble", _ens, 1), ("efficiency", _eff, 1)):
    assert len(lst) == k, f"expected {k} {nm} table(s), found {len(lst)}"
T = {"res": _res, "abl": _abl[0], "abls": _abl, "dm": _dm[0], "ens": _ens[0], "eff": _eff[0]}
DS = ["GEFCom2014", "PJM", "AEMO"]
MET = {"MAE": "MAE", "RMSE": "RMSE", "MAPE (%)": "MAPE", "R²": "R2", "sMAPE (%)": "sMAPE"}

# --- Tables 3,4,5 : result tables (docx indices 3,4,5) ---
for ti, (tb, ds) in enumerate(zip(T["res"], DS)):
    hdr = _hdr(tb)
    for row in tb.rows[1:]:
        cells = [c.text.strip() for c in row.cells]
        k = key(cells[0])
        ens = k.endswith("@ENS"); k = k.replace("@ENS", "")
        if k not in res[ds]:
            bad.append(f"T{ti} {ds}: unknown model '{cells[0]}'"); continue
        blk = res[ds][k]
        src = blk.get("ensemble") if ens else blk["mean"]
        if src is None:
            bad.append(f"T{ti} {ds}: '{cells[0]}' has no ensemble in results"); continue
        for h, cell in zip(hdr[1:], cells[1:]):
            m = MET.get(h)
            if not m or m not in src: continue
            w = src[m]
            # cells are rounded to the printed precision; tolerance = half a unit
            dec = len((cell.split("±")[0].strip().split(".") + [""])[1])
            tol = 0.5 * 10 ** (-dec) + 1e-9
            chk(f"T{ti} {ds} {cells[0]} {h}", num(cell), w, max(tol, abs(w) * 5e-4))
            if not ens and blk.get("std", {}).get(m) is not None:
                chk(f"T{ti} {ds} {cells[0]} {h} std", std(cell), blk["std"][m],
                    max(tol, abs(blk['std'][m]) * 5e-4))

# --- Table 6 : ablation (docx index 6) ---
# The manuscript renames two variants to name the MECHANISM rather than the
# code's flag ("per-component linear base" for linear skip, "full-resolution
# covariate path" for covariate skip). Map both spellings.
LBL = {"full (proposed)": "Full (Proposed)",
       "w/o stage 1 (multi-scale split)": "w/o Stage 1 (multi-scale split)",
       "w/o stage 2 (filter bank)": "w/o Stage 2 (filter bank)",
       "w/o adaptive gating": "w/o adaptive gating",
       "w/o patch embedding": "w/o patch embedding",
       "w/o cross-attention": "w/o cross-attention",
       "w/o bigru decoder": "w/o BiGRU decoder",
       "w/o revin": "w/o RevIN",
       "w/o future covariates": "w/o future covariates",
       "w/o linear skip": "w/o linear skip",
       "w/o per-component linear base": "w/o linear skip",
       "w/o covariate skip": "w/o covariate skip",
       "w/o full-resolution covariate path": "w/o covariate skip",
       "w/ causal error correction": "w/ causal error correction"}
ab = {k: v for k, v in abl.items() if not k.startswith("_")}
def find_ab(label):
    return ab.get(LBL.get(label.strip().lower(), "\0"))
_ABL_DS = ["GEFCom2014", "AEMO", "PJM"]       # document order
for _ai, _tb in enumerate(T["abls"]):
  ab = {k: v for k, v in (saved.get("ablations") or {}).get(
        _ABL_DS[_ai], abl).items() if not k.startswith("_")}
  hdr6 = _hdr(_tb)
  for row in _tb.rows[1:]:
    cells = [c.text.strip() for c in row.cells]
    e = find_ab(cells[0])
    if e is None:
        bad.append(f"T6[{_ABL_DS[_ai]}]: variant '{cells[0]}' not in results")
        continue
    mean = e.get("mean", e)
    for h, cell in zip(hdr6[1:], cells[1:]):
        m = MET.get(h)
        if not m or m not in mean: continue
        dec = len((cell.split("±")[0].strip().split(".") + [""])[1])
        chk(f"T6[{_ABL_DS[_ai]}] {cells[0]} {h}", num(cell), mean[m],
            0.5 * 10 ** (-dec) + 1e-9)

# --- Table 7 : DM (docx index 7) ---
for row in T["dm"].rows[1:]:
    cells = [c.text.strip() for c in row.cells]
    k = key(cells[0])
    for j, ds in enumerate(DS):
        dm = res[ds].get("_dm_multiseed", {}) or res[ds].get("_dm", {})
        e = dm.get(k)
        if e is None: continue
        got = num(cells[1 + 2 * j])
        want = e.get("DM", e.get("dm", e.get("stat")))
        chk(f"T7 {ds} {cells[0]} DM", got, want, 0.05)

# --- Table 8 : ensembles (docx index 8) ---
for row in T["ens"].rows[1:]:
    cells = [c.text.strip() for c in row.cells]
    for j, ds in enumerate(DS):
        name, mape = cells[1 + 2 * j], cells[2 + 2 * j]
        k = key(name)
        if k not in res[ds]:
            bad.append(f"T8 {ds}: unknown model '{name}'"); continue
        ens = res[ds][k].get("ensemble")
        if ens is None:
            bad.append(f"T8 {ds}: '{name}' has no ensemble"); continue
        chk(f"T8 {ds} {name} MAPE", num(mape), ens["MAPE"], 0.005)

# --- Table 9 : efficiency (docx index 9) ---
for row in T["eff"].rows[1:]:
    cells = [c.text.strip() for c in row.cells]
    k = key(cells[0])
    if k not in res["GEFCom2014"]: continue
    b = res["GEFCom2014"][k]
    chk(f"T9 {cells[0]} params", num(cells[1]), b.get("n_params"), 0.5)
    chk(f"T9 {cells[0]} train_s", num(cells[2]), b.get("train_time_s"), 0.5)
    chk(f"T9 {cells[0]} infer_ms", num(cells[3]), b.get("inference_ms"), 0.005)

out = [f"{tot} numeric cells checked across tables 3-9", ""]
out += ([f"  [X] {b}" for b in bad] if bad else ["  no mismatches"])
txt = "\n".join(out)
io.open(SP := r"C:\Users\fmarr\AppData\Local\Temp\claude\e--ARTICULOS-CIENTIFICOS-20260318-load-forecasting-smart-grids\4b26e038-ddeb-4121-8af3-881127a52999\scratchpad\table_check.txt", "w", encoding="utf-8").write(txt)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(txt)
sys.exit(1 if bad else 0)
