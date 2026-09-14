"""
04_five_domain_calibration.py
--------------------------------
The paper's main real-data validation (Sec. 5.3): builds a real, non-
synthetic multi-shift stream by concatenating five multimodal benchmarks
in a fixed order, and compares all four calibration methods on it.

Order: MathVista -> AI2D -> ChartQA -> MMMU -> TextVQA
(math diagrams -> science diagrams -> charts -> exam questions -> natural
photos -- four genuine domain-shift points)

Expects five hybrid-score CSVs (produced by the data_generation/ scripts)
in the working directory or a path you supply:
  scores_mathvista_hybrid.csv
  scores_ai2d_hybrid_fixed.csv   (NOTE: the "_fixed" version -- see
                                   data_generation/fix_ai2d_labels.py)
  scores_chartqa_hybrid.csv
  scores_mmmu_hybrid.csv
  scores_textvqa_hybrid.csv

Produces: figures/fig4_five_domain_stream.png, outputs/five_domain_results.csv

Usage:
    python 04_five_domain_calibration.py --data-dir /path/to/csvs
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

OUT_FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
OUT_DATA_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.30
C_FIXED, C_ACI, C_OURS, C_PID = "#9e9e9e", "#3b6fb6", "#e0742a", "#7b5ea6"

DATASET_ORDER = [
    ("MathVista", "scores_mathvista_hybrid.csv"),
    ("AI2D",      "scores_ai2d_hybrid_fixed.csv"),
    ("ChartQA",   "scores_chartqa_hybrid.csv"),
    ("MMMU",      "scores_mmmu_hybrid.csv"),
    ("TextVQA",   "scores_textvqa_hybrid.csv"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=".")
    args = ap.parse_args()
    d = Path(args.data_dir)

    dfs = []
    for name, fname in DATASET_ORDER:
        path = d / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} -- run the data_generation scripts first.")
        df = pd.read_csv(path).sort_values("order").reset_index(drop=True)
        df["domain"] = name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    scores = combined["nonconformity_score"].values.astype(float)
    correct = combined["correct"].values.astype(int)
    T = len(scores)
    boundaries = np.cumsum([len(d_) for d_ in dfs])
    switch_points = boundaries[:-1]
    domain_names = [n for n, _ in DATASET_ORDER]
    domain_bounds = list(zip([0] + list(switch_points), list(switch_points) + [T]))

    print(f"Combined 5-domain stream: n={T}")
    for name, df in zip(domain_names, dfs):
        print(f"  {name}: n={len(df)}, accuracy={df['correct'].mean():.3f}")

    # ---- run all four methods, per-domain coverage ----
    print(f"\ntarget coverage = {1 - ALPHA:.2f}")
    results_table = {}
    err_traces, q_traces = {}, {}
    for name, fn in METHODS.items():
        out = fn(scores, alpha=ALPHA)
        err, q = out[0], out[1]
        err_traces[name] = err
        q_traces[name] = q
        row = [coverage(err)] + [coverage(err, a, b) for a, b in domain_bounds]
        results_table[name] = row
        print(f"{name:<26}" + "".join(f"{v:<10.3f}" for v in row))

    rdf = pd.DataFrame(results_table, index=["Overall"] + domain_names).T
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(OUT_DATA_DIR / "five_domain_results.csv")

    # ---- figure ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                              gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.5})
    bstarts, bends = [0] + list(switch_points), list(switch_points) + [T]
    dom_colors = ["#eef3fa", "#fdf1e8", "#eafaf1", "#fdf0f5", "#f5f0fa"]

    ax = axes[0]
    for (a, b), c in zip(zip(bstarts, bends), dom_colors):
        ax.axvspan(a, b, color=c, zorder=0)
    pt_colors = ["#d1495b" if c == 0 else "#2e9e5b" for c in correct]
    ax.scatter(np.arange(T), scores, c=pt_colors, s=8, alpha=0.6, linewidths=0)
    ax.plot(q_traces["Fixed / Split CP"], color=C_FIXED, ls="--", lw=1.3, label="Fixed / Split CP")
    ax.plot(q_traces["Standard ACI"], color=C_ACI, lw=1.3, label="Standard ACI")
    ax.plot(q_traces["Drift-Aware ACI (ours)"], color=C_OURS, lw=1.8, label="Drift-Aware ACI (ours)")
    for sp in switch_points:
        ax.axvline(sp, color="#111111", ls=":", lw=1.2)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.18)
    for (a, b), name in zip(zip(bstarts, bends), domain_names):
        ax.text((a + b) / 2, ymax * 1.10, name, ha="center", fontsize=10, fontweight="bold", color="#333333")
    ax.set_ylabel("Nonconformity score / threshold")
    ax.set_title(f"Real 5-domain multi-shift stream (n={T}): "
                 + "\u2192".join(domain_names), fontsize=12.5, fontweight="bold", pad=10)

    def rc(err, w=30):
        return np.convolve(1 - err, np.ones(w) / w, mode="same")

    ax = axes[1]
    for (a, b), c in zip(zip(bstarts, bends), dom_colors):
        ax.axvspan(a, b, color=c, zorder=0)
    ax.plot(rc(err_traces["Fixed / Split CP"]), color=C_FIXED, ls="--", lw=1.6, label="Fixed / Split CP")
    ax.plot(rc(err_traces["Standard ACI"]), color=C_ACI, lw=1.6, label="Standard ACI")
    ax.plot(rc(err_traces["PID-ACI"]), color=C_PID, lw=1.6, label="PID-ACI")
    ax.plot(rc(err_traces["Drift-Aware ACI (ours)"]), color=C_OURS, lw=2.0, label="Drift-Aware ACI (ours)")
    ax.axhline(1 - ALPHA, color="#111111", ls=":", lw=1.3, label=f"Target ({1-ALPHA:.2f})")
    for sp in switch_points:
        ax.axvline(sp, color="#111111", ls=":", lw=1.2)
    ax.set_ylabel("Rolling coverage (window=30)")
    ax.set_xlabel("Stream position $t$", labelpad=12)
    ax.legend(loc="lower center", fontsize=8.5, ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.32))
    plt.tight_layout()
    out_path = OUT_FIG_DIR / "fig4_five_domain_stream.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
