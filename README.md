# Adaptive dual-stage signal decomposition and Patch Transformer–BiGRU with cross-attention for short-term load forecasting in smart grids

A **leakage-free**, fully causal deep-learning framework for day-ahead short-term
load forecasting (STLF), released in **PyTorch and MATLAB** with a machine-checked
causality certificate and a five-seed, significance-tested evaluation protocol.

<p align="center">
  <img src="figures/Fig11_leakage_effect.png" width="88%" alt="The same learner and splits under a leaky and a causal decomposition protocol"/>
</p>
<p align="center">
  <img src="figures/Fig4_accuracy_overview.png" width="88%" alt="Day-ahead accuracy across the three benchmarks"/>
</p>

---

## Why this repository exists

Most decomposition-based STLF hybrids in the literature apply CEEMDAN/EMD/VMD to the
**whole series before the train–test split**. Because these are *global* transforms,
this injects future information into every training subsequence and inflates reported
accuracy far beyond what is attainable in operation. This project:

1. **Quantifies that bias.** A controlled experiment with an identical learner and
   identical splits shows the decompose-then-split protocol reports **45.9% lower
   error** than a causal protocol — an illusion created purely by the evaluation
   (`python/leakage_demo.py`, `figures/Fig11_leakage_effect.png`).
2. **Fixes it.** The offline CEEMDAN–sample-entropy–VMD chain is replaced by an
   **in-model, end-to-end learnable, strictly causal** analogue (multi-scale causal
   moving-average split → learnable causal FIR band-pass bank → differentiable
   complexity gate), tokenized into day-length patches, encoded by a Transformer,
   fused with future known covariates via cross-attention, decoded by a BiGRU with
   reversible instance normalization, and anchored by a per-component linear base
   plus a full-resolution covariate path.
3. **Evaluates honestly.** Operational day-ahead protocol (non-overlapping windows),
   train-only preprocessing statistics, unmodified test targets, five seeds, and
   Holm-corrected Diebold–Mariano tests.

## Headline results (leak-free protocol, 5 seeds, MAPE %)

Single models (mean over five seeds), ranked against the other thirteen:

| Dataset | Type | **Proposed** | Rank | Best baseline | Verdict (Holm-DM) |
|---|---|---|---|---|---|
| **PJM** | univariate | **4.89** | **1 of 14** | PatchTST 4.98 | Beats most; none beat it |
| **AEMO** | univariate | **5.54** | **1 of 14** | DLinear 5.60 | Beats most; none beat it |
| **GEFCom2014** | + temperature | 4.73 | 11 of 14 | TCN 4.13 | **No baseline beats it significantly** |

Seed ensembles are compared **like-for-like** (every model's five-seed ensemble,
never an ensemble against single models): the proposed ensemble is 1st of 12 on
PJM (4.75) and AEMO (5.22), and 9th of 12 on GEFCom2014 (4.27 vs 3.70 for the TCN
ensemble), with no baseline ensemble significantly better on any dataset.

**No baseline outperforms the proposed model with statistical significance on any
dataset** — read as statistical parity at n = 259/902/274 with Holm correction, not
as demonstrated equality. It significantly beats seasonal persistence everywhere and
seasonal ARIMA on GEFCom2014 and PJM. Full tables: [`results/`](results/).

The ablation is the more interesting result: on the temperature-rich benchmark the
**covariate pathways carry the accuracy while the decomposition machinery is
neutral**, the mirror image of the univariate case where the same architecture
leads. Decomposition-based inductive biases help univariate load but yield to
direct covariate modelling when a strong exogenous driver is present.

## Repository layout

```
.
├── python/        Reference implementation (PyTorch)
│   ├── main.py                 full pipeline: train, evaluate, DM tests, tables, figures
│   ├── model_proposed.py       the proposed CPTB model + causal decomposition
│   ├── models_baselines.py     13 baselines incl. DLinear, PatchTST, seasonal-naïve
│   ├── data_utils.py           leak-free loading, windowing, observed-data masking
│   ├── prepare_data.py         raw public downloads → the canonical CSVs, row-count checked
│   ├── train_pipeline.py       training loop, causal error correction, GBM/ARIMA
│   ├── metrics_stats.py        metrics + Diebold–Mariano (Newey-West, HLN, Holm)
│   ├── figures_tables.py       every table + the single driver for all 11 figures
│   ├── figstyle.py             shared publication style; palette validated for CVD
│   ├── figures_diagrams.py     Figs 1–2 (protocol contrast, architecture)
│   ├── figures_results.py      Figs 3–11, all computed from saved result files
│   ├── leakage_demo.py         the controlled decompose-then-split leakage experiment
│   └── verify_cptb.py          machine-checked causality / reconstruction certificate
├── matlab/        Independent MATLAB port (Deep Learning Toolbox), verified in parity
├── results/       Result tables (CSV + LaTeX) and JSON summaries
├── figures/       Publication figures (PDF + 300-dpi PNG)
└── data/          Place datasets here (see data/README.md)
```

Regenerating every table and figure from the saved results takes seconds and needs
no GPU:

```bash
cd python && python -c "import figures_tables as f; f.regenerate_from_saved()"
```

## Quick start (Python)

```bash
cd python
pip install -r requirements.txt          # torch, numpy, pandas, scikit-learn, xgboost, lightgbm, statsmodels
python verify_cptb.py                     # machine-checked causality certificate (seconds)
python prepare_data.py --raw /path/to/raw_downloads   # build the canonical CSVs
python main.py                            # full 3-dataset, 5-seed protocol + ablation + figures
python main.py --smoke                    # tiny end-to-end validation run
python leakage_demo.py --dataset GEFCom2014   # reproduce the leakage quantification
```

## Quick start (MATLAB)

```matlab
cd matlab
verify_cptb                 % correctness certificate: shapes, exact reconstruction,
                            % strict causality, naive-init, all ablations, gradient flow
results = main_v4('GEFCom2014');   % full protocol on one dataset
```

Requires MATLAB R2022b+ and the Deep Learning Toolbox only. See [`matlab/README.md`](matlab/README.md).

## Figures

All eleven are written by one driver from the saved result files, in a shared
style whose categorical palette is machine-validated for colour-vision
deficiency (OKLab ΔE on every adjacent pair, plus a contrast check).

| | | |
|---|---|---|
| **1** protocol contrast | **2** architecture | **3** causal decomposition of one window |
| **4** accuracy across benchmarks | **5–7** day-ahead forecasts, one per dataset | **8** ablation |
| **9** cross-attention | **10** lead time vs time of day | **11** the leakage experiment |

Two of them are worth singling out because they report against the paper's own
thesis. Figure 8 shows that on the temperature-rich benchmark it is the
covariate pathways, not the decomposition, that carry the accuracy. Figure 10
shows that the error is shaped as much by time of day as by lead time, and that
the once-a-day issue schedule confounds the two.

## Reproducibility & rigor

- **Causality is machine-checked**, not asserted: `verify_cptb` (both languages) perturbs
  the most recent input and confirms no earlier decomposition output changes, and checks
  exact reconstruction and the seasonal-naïve initialization.
- **No fabricated ground truth**: multi-week gaps in GEFCom2014 are detected and every
  window overlapping fabricated data is excluded from training and evaluation.
- **Every figure is generated from result files**, and a missing input raises rather than
  falling back to a default. This is enforced, not merely intended: an audit of this
  repository found Fig. 11 being drawn from literals typed into the figure code, because
  `leakage_demo.py` printed its measurement without saving it. It now writes
  `results/leakage_GEFCom2014.json`, and the figure reads it back.
- **Two independent implementations** (PyTorch and MATLAB) with verified numerical parity.

## Datasets

All three benchmarks are public; download instructions are in [`data/README.md`](data/README.md).
GEFCom2014 (hourly, with temperature), PJM Interconnection East (hourly), and AEMO New
South Wales (half-hourly).

## Citation

```bibtex
@article{ArrabalCampos_STLF,
  author  = {Arrabal-Campos, Francisco M.},
  title   = {Adaptive dual-stage signal decomposition and Patch Transformer--BiGRU
             with cross-attention for short-term load forecasting in smart grids},
  journal = {Applied Energy},
  note    = {Under review},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
