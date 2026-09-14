"""
02_ablation_hyperparameters.py
--------------------------------
Ablates the four hyperparameters of the drift-aware mechanism (gamma_min,
gamma_max, lambda, drift-estimation window size) on the same synthetic
stream as 01_synthetic_validation.py, varying one at a time around the
baseline configuration while holding the others fixed, with 15 seeds per
setting.

Produces: figures/fig2_ablation.png

Usage:
    python 02_ablation_hyperparameters.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
from calibration_methods import drift_aware_aci, coverage  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 1200
ALPHA = 0.10
CALIB_N = 200
N_SEEDS = 15
C_OURS, C_ACI = "#e0742a", "#3b6fb6"

BASELINE = dict(gamma_min=0.005, gamma_max=0.20, lam=4.0, window=30)
GRIDS = {
    "gamma_min": [0.001, 0.005, 0.01, 0.02],
    "gamma_max": [0.10, 0.20, 0.30, 0.50],
    "lam":       [1.0, 2.0, 4.0, 8.0],
    "window":    [10, 20, 30, 60],
}


def make_scores(seed):
    rng = np.random.default_rng(seed)
    scale = np.ones(T)
    scale[:400] = 1.0
    scale[400:550] = 3.0
    scale[550:800] = np.linspace(3.0, 1.4, 250)
    scale[800:] = 1.4
    return rng.exponential(scale=scale)


def run_once(seed, **cfg):
    scores = make_scores(seed)
    err, q, _ = drift_aware_aci(scores, alpha=ALPHA, calib_n=CALIB_N, gamma0=0.02, **cfg)
    return coverage(err), coverage(err, 400, 460), q.mean()


def main():
    rows = []
    for param, values in GRIDS.items():
        for v in values:
            cfg = dict(BASELINE)
            cfg[param] = v
            res = [run_once(s, **cfg) for s in range(N_SEEDS)]
            overall = np.array([r[0] for r in res])
            shift60 = np.array([r[1] for r in res])
            rows.append({
                "param": param, "value": v,
                "overall_mean": overall.mean(), "overall_std": overall.std(),
                "shift60_mean": shift60.mean(), "shift60_std": shift60.std(),
            })
    adf = pd.DataFrame(rows)
    print(adf.to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
    titles = {"gamma_min": r"$\gamma_{min}$", "gamma_max": r"$\gamma_{max}$",
              "lam": r"$\lambda$", "window": "drift window"}
    for ax, param in zip(axes, GRIDS.keys()):
        sub = adf[adf.param == param].sort_values("value")
        x = sub["value"].astype(str)
        ax.errorbar(x, sub["overall_mean"], yerr=sub["overall_std"], marker="o", color=C_OURS,
                    capsize=3, lw=1.8, label="Overall coverage")
        ax.errorbar(x, sub["shift60_mean"], yerr=sub["shift60_std"], marker="s", color=C_ACI,
                    capsize=3, lw=1.8, ls="--", label="Coverage, first 60 steps after shift")
        ax.axhline(1 - ALPHA, color="black", ls=":", lw=1.2)
        base_str = str(BASELINE[param])
        if base_str in list(x):
            ax.axvline(list(x).index(base_str), color="#cccccc", zorder=0, lw=8, alpha=0.5)
        ax.set_title(titles[param], fontsize=11, fontweight="bold")
        ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel("Coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Hyperparameter sensitivity (N={N_SEEDS} seeds/setting, shaded bar = baseline)",
                 fontsize=12, fontweight="bold", y=1.06)
    plt.tight_layout()
    out_path = OUT_DIR / "fig2_ablation.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
