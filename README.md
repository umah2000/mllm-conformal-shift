# Shift-Robust Conformal Uncertainty Quantification for Streaming Multimodal Foundation Models

This repository contains the full data-generation and analysis pipeline behind the
paper *"Shift-Robust Conformal Uncertainty Quantification for Streaming Multimodal
Foundation Models."* It combines (i) a training-free, hybrid nonconformity score
for multimodal LLM outputs and (ii) an online, drift-aware conformal calibration
mechanism, and evaluates both on synthetic data and on a real, five-benchmark
multimodal domain-shift stream (MathVista &rarr; AI2D &rarr; ChartQA &rarr; MMMU &rarr; TextVQA,
n=1675).

Every number and figure in the paper can be reproduced from the scripts in this
repository, either directly (the `analysis/` scripts, which are CPU-only and run
in seconds to minutes) or by first regenerating the underlying model outputs
(the `data_generation/` scripts, which need a GPU).

---

## Repository layout

```
.
├── requirements.txt              # CPU-only deps for analysis/ and calibration/
├── calibration/
│   └── calibration_methods.py    # The 4 calibration methods, shared by every
│                                  # analysis script (no duplicated code)
├── data_generation/               # GPU-required: query a VLM, produce score CSVs
│   ├── local_cpu_moondream2.py           # moondream2 on CPU (slow; exploratory
│   │                                      #   pilot only, see paper Sec. 5.2)
│   ├── colab_qwen2vl_v1_score.py         # Qwen2-VL-2B, v1 (semantic-volume-only)
│   │                                      #   score, for Colab/Kaggle GPU
│   ├── colab_qwen2vl_hybrid_score.py     # Qwen2-VL-2B, hybrid score (paper's
│   │                                      #   main score, Sec. 3.2)
│   ├── colab_scale_up_5domain.ipynb      # Hybrid score on AI2D / ChartQA / TextVQA
│   ├── colab_lacaps_baseline.py          # LAC/APS token-probability baseline
│   │                                      #   (Azad et al., paper ref [6])
│   └── fix_ai2d_labels.py                # One-off fix for a ground-truth-label
│                                          #   bug in the first AI2D run (see below)
│   ├── colab_lacaps_baseline.py
│   └── colab_concurrent_baselines_comparison.ipynb  # Best-effort re-impl. of
│                                                      #   PromptShift-CRC [8] and
│                                                      #   Domain-Shift-Aware CP [9]
│                                                      #   -> Figure 8, Sec. 5.7
├── analysis/                      # CPU-only: turn score CSVs into paper results
│   ├── 01_synthetic_validation.py        # -> Figure 1  (Sec. 4)
│   ├── 02_ablation_hyperparameters.py    # -> Figure 2  (Sec. 4.1)
│   ├── 03_hybrid_score_comparison.py     # -> Figure 3  (Sec. 5.2)
│   ├── 04_five_domain_calibration.py     # -> Figure 4  (Sec. 5.3) -- main result
│   ├── 05_bootstrap_validation.py        # -> Figures 5, 6 (Sec. 5.4)
│   └── 06_lacaps_comparison.py           # -> Figure 7  (Sec. 5.6)
├── figures/                       # Pre-rendered reference copies of every
│                                  # figure in the paper, so you can see the
│                                  # results without running anything. Running
│                                  # any analysis/ script regenerates these
│                                  # fresh under outputs/figures/ at the repo
│                                  # root (kept separate so re-runs never
│                                  # silently overwrite the checked-in copies).

```

Running any script under `analysis/` creates an `outputs/` directory (at the repo
root) containing `outputs/figures/*.png` and any intermediate CSVs (bootstrap
samples, per-domain result tables, etc.).

---

## Quick start (reproduce the figures from already-generated data)

If you just want to regenerate the paper's figures and numbers from the score
CSVs (i.e. skip the GPU step), you need the CSV files listed in each script's
docstring, then:

```bash
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# Each script is independent; --data-dir points at wherever your CSVs live.
python analysis/01_synthetic_validation.py                       # no CSVs needed
python analysis/02_ablation_hyperparameters.py                   # no CSVs needed
python analysis/03_hybrid_score_comparison.py --data-dir ./data
python analysis/04_five_domain_calibration.py --data-dir ./data
python analysis/05_bootstrap_validation.py    --data-dir ./data --n-boot 1000
python analysis/06_lacaps_comparison.py       --data-dir ./data
```

Scripts `01` and `02` are fully self-contained (they generate their own synthetic
data) and need no external files. Scripts `03`, `04`, `05`, and `06` expect the
CSVs produced by `data_generation/` (see next section) to already exist.

---

## Full pipeline (regenerate everything from scratch, including the model outputs)

### 1. Data generation (needs a GPU)

All `data_generation/` scripts were developed and run on **free-tier GPU
notebooks** (Google Colab T4, Kaggle T4&times;2) -- no paid compute was used
anywhere in this project. `local_cpu_moondream2.py` is the one exception and
runs on CPU only, but is slow and was used solely for the small exploratory
pilot that first surfaced the score-collapse issue (paper Sec. 5.2); it is not
needed to reproduce the paper's main results.

Recommended order:

1. **`colab_qwen2vl_hybrid_score.py`** -- run for `DATASET = "mathvista"` and then
   `DATASET = "mmmu"` (edit the constant at the top of the file). Produces
   `scores_mathvista_hybrid.csv` and `scores_mmmu_hybrid.csv`.
2. **`colab_scale_up_5domain.ipynb`** -- open in Colab or Kaggle (Kaggle: enable
   GPU T4&times;2 and Internet access in the notebook's Settings panel first),
   run all cells. Produces `scores_ai2d_hybrid.csv`, `scores_chartqa_hybrid.csv`,
   `scores_textvqa_hybrid.csv`.
3. **`fix_ai2d_labels.py`** -- **required** after step 2. The first AI2D run
   stored ground-truth answers as choice *text* while the model was prompted to
   answer with a *letter*, causing a near-total (and spurious) accuracy
   collapse. This script reloads only the AI2D dataset metadata (no GPU, no
   model, a few seconds) and rewrites the ground-truth column correctly.
   Produces `scores_ai2d_hybrid_fixed.csv` (this is the file `analysis/`
   scripts expect, *not* the unfixed `scores_ai2d_hybrid.csv`). The bug is
   already patched in `colab_scale_up_5domain.ipynb` for future runs; this
   script is only needed to repair output already collected before the fix.
4. **`colab_lacaps_baseline.py`** -- run for `DATASET = "mathvista"` and then
   `"mmmu"`. Produces `scores_mathvista_lacaps.csv` and `scores_mmmu_lacaps.csv`,
   used only by `analysis/06_lacaps_comparison.py`.
5. *(Optional, exploratory only)* **`local_cpu_moondream2.py`** -- the original
   CPU pipeline used for the small pilot that first revealed the semantic-volume
   score-collapse issue. Not required for the main results.
6. **`colab_qwen2vl_v1_score.py`** -- regenerates the *pre-fix* (v1,
   semantic-volume-only) score, used only as the "before" condition in
   `analysis/03_hybrid_score_comparison.py`'s before/after comparison. If you
   only care about the paper's main (hybrid-score) results, you can skip this.

All `data_generation/` scripts are checkpointed: interrupt at any time and
re-run to resume from the last completed item.

### 2. Analysis (CPU only)

Once the CSVs above exist in one directory, run the `analysis/` scripts in
order as shown in "Quick start." Each script prints its console output (which
should match the paper's tables) and saves its figure(s) to `outputs/figures/`.

### 3. Paper (optional)

`paper/build_paper_docx.js` regenerates the Word-format manuscript draft,
embedding the figures produced above. Requires Node.js and the `docx` npm
package (`npm install docx`).

```bash
cd paper
npm install docx
node build_paper_docx.js
```

---

## Notes on honesty and reproducibility

- Every number quoted in the paper's Results sections was produced by the
  scripts in this repository, run against the CSVs described above -- nothing
  is hand-typed or estimated.
- `calibration/calibration_methods.py` is the single source of truth for all
  four calibration methods; every analysis script imports from it rather than
  redefining the methods locally, so there is exactly one implementation to
  audit.
- The `PID-ACI` baseline (Angelopoulos, Cand&egrave;s & Tibshirani, 2023) uses
  heuristic, not exhaustively-searched, hyperparameters (`C_sat`, `K_I`) -- see
  the docstring in `calibration_methods.py` and Sec. 5.5 of the paper for the
  full discussion of this limitation.
- The AI2D label bug (see step 3 above) is documented rather than silently
  fixed, because it is a useful worked example of a subtle but easy-to-make
  data-pipeline mistake (prompting for one answer format while storing ground
  truth in another) that is worth leaving visible for anyone extending this
  pipeline to a new multiple-choice dataset.

---

## Citation

If you use this code, please cite the paper (full citation to be added once
the manuscript is finalized / assigned a DOI). See `paper/build_paper_docx.js`
for the full reference list (with DOIs) of prior work this project builds on
and compares against.

## License

[TO DECIDE] -- add a LICENSE file (e.g. MIT or Apache-2.0) before making this
repository public, if you intend to allow reuse.
