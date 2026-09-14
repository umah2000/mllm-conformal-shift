"""
05_bootstrap_validation.py
-----------------------------
Quantifies sampling uncertainty on the 5-domain real stream (Sec. 5.4 of
the paper) via a block bootstrap: resample with replacement independently
WITHIN each of the five domain blocks (preserving the fixed
MathVista->AI2D->ChartQA->MMMU->TextVQA block order, which is the source
of the four genuine shift events), 1000 times, and recompute coverage for
all four methods on each resample.

Produces:
  figures/fig5_forest_5domain.png     (95% CI forest plot, per domain)
  figures/fig6_ridgeline_5domain.png  (full bootstrap distributions)
  outputs/bootstrap_5domain.csv       (raw per-resample results)
  outputs/bootstrap_5domain_ci.csv    (summary CIs)

Usage:
    python 05_bootstrap_validation.py --data-dir /path/to/csvs --n-boot 1000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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
TARGET = 1 - ALPHA
C_FIXED, C_ACI, C_OURS, C_PID = "#9e9e9e", "#3b6fb6", "#e0742a", "#7b5ea6"
METHOD_ORDER = ["Fixed / Split CP", "PID-ACI", "Standard ACI", "Drift-Aware ACI (ours)"]
COLORS = {"Fixed / Split CP": C_FIXED, "PID-ACI": C_PID, "Standard ACI": C_ACI, "Drift-Aware ACI (ours)": C_OURS}

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
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = Path(args.data_dir)

    dfs = {}
    for name, fname in DATASET_ORDER:
        path = d / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} -- run the data_generation scripts first.")
        dfs[name] = pd.read_csv(path).sort_values("order").reset_index(drop=True)
    domain_names = [n for n, _ in DATASET_ORDER]

    rng = np.random.default_rng(args.seed)
    records = []
    for b in range(args.n_boot):
        resampled = [dfs[name].sample(n=len(dfs[name]), replace=True,
                                       random_state=rng.integers(1e9)).reset_index(drop=True)
                     for name in domain_names]
        scores = np.concatenate([r["nonconformity_score"].values for r in resampled]).astype(float)
        bounds = np.cumsum([0] + [len(r) for r in resampled])
        for name, fn in METHODS.items():
            err = fn(scores, alpha=ALPHA)[0]
            rec = {"method": name, "boot": b, "overall": coverage(err)}
            for i, dn in enumerate(domain_names):
                rec[dn] = coverage(err, bounds[i], bounds[i + 1])
            records.append(rec)

    bdf = pd.DataFrame(records)
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    bdf.to_csv(OUT_DATA_DIR / "bootstrap_5domain.csv", index=False)

    print(f"=== Bootstrap 95% CIs (N={args.n_boot}) ===")
    ci_rows = []
    for name in METHODS:
        for metric in ["overall"] + domain_names:
            sub = bdf[bdf.method == name][metric]
            lo, hi = np.percentile(sub, [2.5, 97.5])
            ci_rows.append((name, metric, sub.mean(), lo, hi))
            print(f"{name:<26}{metric:<12}mean={sub.mean():.3f}  CI=[{lo:.3f}, {hi:.3f}]")
    cidf = pd.DataFrame(ci_rows, columns=["method", "metric", "mean", "ci_lo", "ci_hi"])
    cidf.to_csv(OUT_DATA_DIR / "bootstrap_5domain_ci.csv", index=False)

    ours = bdf[bdf.method == "Drift-Aware ACI (ours)"].sort_values("boot")["overall"].values
    std = bdf[bdf.method == "Standard ACI"].sort_values("boot")["overall"].values
    diff = ours - std
    print(f"\nOurs - Standard ACI (overall coverage): mean={diff.mean():.4f}, "
          f"95% CI=[{np.percentile(diff, 2.5):.4f}, {np.percentile(diff, 97.5):.4f}]")
    print(f"Fraction of resamples where ours > standard ACI: {(diff > 0).mean():.3f}")

    # ---- Figure 5: forest plot ----
    metrics = ["overall"] + domain_names
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 4.4), sharey=True)
    for ax, metric in zip(axes, metrics):
        sub = cidf[cidf.metric == metric].set_index("method").loc[METHOD_ORDER]
        y = np.arange(len(METHOD_ORDER))[::-1]
        for yi, m in zip(y, METHOD_ORDER):
            row = sub.loc[m]
            ax.plot([row.ci_lo, row.ci_hi], [yi, yi], color=COLORS[m], lw=3, solid_capstyle="round", alpha=0.85, label=m)
            ax.scatter([row["mean"]], [yi], color=COLORS[m], s=90, zorder=5, edgecolor="white", linewidth=1.2)
        ax.axvline(TARGET, color="#111111", ls=":", lw=1.3)
        ax.set_yticks([])
        ax.set_xlim(0.0, 1.0)
        ax.set_title(metric if metric != "overall" else "Overall", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Coverage", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Bootstrap 95% CI across the 5-domain real stream (N={args.n_boot})",
                 fontsize=13, fontweight="bold", y=1.06)
    plt.tight_layout()
    out5 = OUT_FIG_DIR / "fig5_forest_5domain.png"
    plt.savefig(out5, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out5}")

    # ---- Figure 6: ridgeline distribution ----
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x_grid = np.linspace(0.05, 1.0, 500)
    offset_step = 1.15
    for i, m in enumerate(METHOD_ORDER):
        vals = bdf[bdf.method == m]["overall"].values
        kde = gaussian_kde(vals, bw_method=0.2)
        density = kde(x_grid)
        density = density / density.max() * 0.9
        base = i * offset_step
        ax.fill_between(x_grid, base, base + density, color=COLORS[m], alpha=0.75, lw=0, zorder=10 - i)
        ax.plot(x_grid, base + density, color=COLORS[m], lw=1.4, zorder=10 - i)
        mean_v = vals.mean()
        ax.plot([mean_v, mean_v], [base, base + kde(mean_v)[0] / kde(x_grid).max() * 0.9],
                color="white", lw=1.6, zorder=11)
        ax.text(0.06, base + 0.28, m, fontsize=9.5, fontweight="bold", color=COLORS[m], va="bottom")
        ax.text(0.06, base + 0.02, f"mean={mean_v:.3f}", fontsize=8, color="#555555", va="bottom")
    ax.axvline(TARGET, color="#111111", ls=":", lw=1.5, zorder=20)
    ax.text(TARGET, offset_step * len(METHOD_ORDER) + 0.15, f"target coverage\n({TARGET:.2f})",
            ha="center", fontsize=9, color="#333333")
    ax.set_yticks([])
    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(-0.1, offset_step * len(METHOD_ORDER) + 0.5)
    ax.set_xlabel(f"Bootstrap overall coverage ({args.n_boot} resamples)")
    ax.set_title("Full bootstrap distribution of overall coverage, by method", fontsize=12.5, fontweight="bold")
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    out6 = OUT_FIG_DIR / "fig6_ridgeline_5domain.png"
    plt.savefig(out6, dpi=300, bbox_inches="tight")
    print(f"Saved: {out6}")


if __name__ == "__main__":
    main()
