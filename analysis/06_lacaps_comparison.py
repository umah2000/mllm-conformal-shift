"""
06_lacaps_comparison.py
--------------------------
Tests whether the classical, bounded token-probability conformal scores of
Azad et al. [6] (LAC, APS) are shift-robust under fixed-threshold
calibration on the SAME real domain shift used elsewhere in this paper
(Sec. 5.6). Complements 04_five_domain_calibration.py, which uses our own
sample-based hybrid score.

Expects two CSVs (produced by data_generation/colab_lacaps_baseline.py):
  scores_mathvista_lacaps.csv
  scores_mmmu_lacaps.csv
Each row has both a `lac_score` and an `aps_score` column plus `correct`.

Produces: figures/fig7_lacaps.png

Usage:
    python 06_lacaps_comparison.py --data-dir /path/to/csvs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
from calibration_methods import METHODS, coverage  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.30
C_FIXED, C_ACI, C_OURS, C_PID = "#9e9e9e", "#3b6fb6", "#e0742a", "#7b5ea6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=".")
    args = ap.parse_args()
    d = Path(args.data_dir)

    mv = pd.read_csv(d / "scores_mathvista_lacaps.csv").sort_values("order").reset_index(drop=True)
    mm = pd.read_csv(d / "scores_mmmu_lacaps.csv").sort_values("order").reset_index(drop=True)
    switch = len(mv)

    results = {}
    for score_col, label in [("lac_score", "LAC"), ("aps_score", "APS")]:
        scores = np.concatenate([mv[score_col].values, mm[score_col].values]).astype(float)
        T = len(scores)
        print(f"\n=== {label} score, n={T}, switch at {switch} ===")
        row = {}
        for name, fn in METHODS.items():
            err = fn(scores, alpha=ALPHA)[0]
            row[name] = {
                "overall": coverage(err), "mathvista": coverage(err, 0, switch),
                "mmmu_first20": coverage(err, switch, switch + 20),
                "mmmu_rest": coverage(err, switch + 20, T),
            }
            print(f"  {name:<26} overall={row[name]['overall']:.3f}  "
                  f"mathvista={row[name]['mathvista']:.3f}  "
                  f"mmmu_first20={row[name]['mmmu_first20']:.3f}  "
                  f"mmmu_rest={row[name]['mmmu_rest']:.3f}")
        results[label] = row

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    method_order = ["Fixed / Split CP", "Standard ACI", "PID-ACI", "Drift-Aware ACI (ours)"]
    colors = [C_FIXED, C_ACI, C_PID, C_OURS]
    phases = ["overall", "mathvista", "mmmu_first20", "mmmu_rest"]
    phase_labels = ["Overall", "MathVista", "First 20\nMMMU", "Rest\nMMMU"]

    for ax, label in zip(axes, ["LAC", "APS"]):
        row = results[label]
        x = np.arange(len(phases))
        w = 0.2
        for i, (m, c) in enumerate(zip(method_order, colors)):
            vals = [row[m][p] for p in phases]
            bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=c)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.2)
        ax.axhline(1 - ALPHA, color="#111111", ls=":", lw=1.3)
        ax.set_xticks(x); ax.set_xticklabels(phase_labels, fontsize=9)
        ax.set_title(f"{label} score (Azad et al. [6] style)", fontsize=11.5, fontweight="bold")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Testing [6]'s bounded token-probability scores on our real domain shift",
                 fontsize=12.5, fontweight="bold", y=1.05)
    plt.tight_layout()
    out_path = OUT_DIR / "fig7_lacaps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
