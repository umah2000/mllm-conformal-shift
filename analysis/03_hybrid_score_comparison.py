"""
03_hybrid_score_comparison.py
-------------------------------
Compares the v1 nonconformity score (semantic volume only, Sec. 3.2 of the
paper) against the hybrid score (semantic volume + first-token entropy
fallback for short answers) on real model outputs, using two metrics:

  - zero-score fraction: how often the score degenerates to exactly 0
    (all k samples for an item came back identical)
  - AUROC: how well the score predicts whether the model's answer was
    wrong (score vs. incorrectness)

Expects four CSVs (produced by the data_generation/ scripts) in the
working directory or a path you supply:
  scores_mathvista_gpu_v2.csv   (v1 score, MathVista)
  scores_mmmu_gpu_v2.csv        (v1 score, MMMU)
  scores_mathvista_hybrid.csv   (hybrid score, MathVista)
  scores_mmmu_hybrid.csv        (hybrid score, MMMU)

Produces: figures/fig3_hybrid_comparison.png

Usage:
    python 03_hybrid_score_comparison.py --data-dir /path/to/csvs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_V1, C_HYBRID = "#d1495b", "#e0742a"


def summarize(path):
    df = pd.read_csv(path)
    zero_frac = (df["nonconformity_score"] == 0).mean()
    try:
        auc = roc_auc_score(1 - df["correct"], df["nonconformity_score"])
    except ValueError:
        auc = float("nan")  # only one class present
    return zero_frac, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=".",
                     help="Directory containing the four score CSVs")
    args = ap.parse_args()
    d = Path(args.data_dir)

    files = {
        "MathVista": {"v1": d / "scores_mathvista_gpu_v2.csv", "hybrid": d / "scores_mathvista_hybrid.csv"},
        "MMMU":      {"v1": d / "scores_mmmu_gpu_v2.csv",      "hybrid": d / "scores_mmmu_hybrid.csv"},
    }

    results = {}
    for dataset, paths in files.items():
        for version, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {path} -- run the data_generation scripts first.")
            zero_frac, auc = summarize(path)
            results[(dataset, version)] = (zero_frac, auc)
            print(f"{dataset:<10} {version:<8} zero-score-frac={zero_frac:.3f}  AUROC={auc:.3f}")

    labels = list(files.keys())
    zero_v1 = [results[(d_, "v1")][0] for d_ in labels]
    zero_hy = [results[(d_, "hybrid")][0] for d_ in labels]
    auc_v1 = [results[(d_, "v1")][1] for d_ in labels]
    auc_hy = [results[(d_, "hybrid")][1] for d_ in labels]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    x = np.arange(len(labels))
    w = 0.32

    ax = axes[0]
    b1 = ax.bar(x - w / 2, zero_v1, w, label="v1 (semantic volume only)", color=C_V1, alpha=0.85)
    b2 = ax.bar(x + w / 2, zero_hy, w, label="Hybrid (ours)", color=C_HYBRID)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Zero-score fraction")
    ax.set_ylim(0, 1.15)
    ax.set_title("Degenerate (zero) scores\n(lower is better)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.18))

    ax = axes[1]
    b1 = ax.bar(x - w / 2, auc_v1, w, label="v1 (semantic volume only)", color=C_V1, alpha=0.85)
    b2 = ax.bar(x + w / 2, auc_hy, w, label="Hybrid (ours)", color=C_HYBRID)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.5, color="#111111", ls=":", lw=1.3, label="Chance level (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC (score vs. incorrectness)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Uncertainty-score informativeness\n(higher is better)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.18))

    fig.suptitle("Effect of the hybrid nonconformity score (Sec. 3.2)", fontsize=12.5, fontweight="bold", y=1.04)
    plt.tight_layout()
    out_path = OUT_DIR / "fig3_hybrid_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
