"""
01_synthetic_validation.py
----------------------------
Validates the calibration mechanism (Sec. 3.3 of the paper) in isolation,
on a controlled synthetic nonconformity-score stream, BEFORE any
model-specific engineering. This is the cleanest test of the mechanism
itself: no VLM, no real data, just a score stream with an engineered
distribution shift.

Stream design (T=1200):
  - t in [0, 400)    : stationary phase
  - t in [400, 550)  : abrupt shift (scale jumps 1.0 -> 3.0)
  - t in [550, 800)  : gradual drift back down (3.0 -> 1.4)
  - t in [800, 1200) : new stationary regime

Runs N=30 independent random seeds and reports:
  - mean +/- std coverage for all three methods (overall, first 60 steps
    after the abrupt shift, and during the gradual-drift phase)
  - a paired t-test + Cohen's d comparing our method's shift-recovery
    coverage against standard ACI, to confirm the advantage is not a
    single-seed artifact

Produces: figures/fig1_synthetic_multiseed.png

Usage:
    python 01_synthetic_validation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
from calibration_methods import fixed_cp, standard_aci, drift_aware_aci, coverage  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 1200
ALPHA = 0.10
CALIB_N = 200
N_SEEDS = 30

C_FIXED, C_ACI, C_OURS = "#9e9e9e", "#3b6fb6", "#e0742a"


def make_scores(seed):
    rng = np.random.default_rng(seed)
    scale = np.ones(T)
    scale[:400] = 1.0
    scale[400:550] = 3.0
    scale[550:800] = np.linspace(3.0, 1.4, 250)
    scale[800:] = 1.4
    return rng.exponential(scale=scale)


def run_seed(seed):
    scores = make_scores(seed)
    err_f, q_f = fixed_cp(scores, alpha=ALPHA, calib_n=CALIB_N)
    err_a, q_a = standard_aci(scores, alpha=ALPHA, calib_n=CALIB_N, gamma=0.02)
    err_d, q_d, _ = drift_aware_aci(scores, alpha=ALPHA, calib_n=CALIB_N,
                                     gamma0=0.02, lam=4.0, gamma_min=0.005,
                                     gamma_max=0.20, window=30)
    return {
        "fixed_overall": coverage(err_f), "fixed_shift60": coverage(err_f, 400, 460),
        "aci_overall": coverage(err_a), "aci_shift60": coverage(err_a, 400, 460),
        "ours_overall": coverage(err_d), "ours_shift60": coverage(err_d, 400, 460),
        "ours_avg_q": q_d.mean(), "aci_avg_q": q_a.mean(),
        "err_f": err_f, "err_a": err_a, "err_d": err_d,
    }


def main():
    results = [run_seed(s) for s in range(N_SEEDS)]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("err_")} for r in results])

    print(f"=== Multi-seed validation (N={N_SEEDS} seeds) ===")
    for method in ["fixed", "aci", "ours"]:
        for metric in ["overall", "shift60"]:
            col = f"{method}_{metric}"
            print(f"  {col:<16} mean={df[col].mean():.3f}  std={df[col].std():.3f}")

    t_stat, p_val = stats.ttest_rel(df["ours_shift60"], df["aci_shift60"])
    d = (df["ours_shift60"] - df["aci_shift60"]).mean() / (df["ours_shift60"] - df["aci_shift60"]).std()
    print(f"\nPaired t-test (ours vs standard ACI, shift60 coverage): "
          f"t({N_SEEDS-1})={t_stat:.3f}, p={p_val:.2e}, Cohen's d={d:.3f}")

    # ---- figure: mean +/- std rolling coverage band across all seeds ----
    def rc(err, w=50):
        return np.convolve(1 - err, np.ones(w) / w, mode="same")

    traces = {"fixed": [], "aci": [], "ours": []}
    for r in results:
        traces["fixed"].append(rc(r["err_f"]))
        traces["aci"].append(rc(r["err_a"]))
        traces["ours"].append(rc(r["err_d"]))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axvspan(0, 400, color="#f0f0f0", zorder=0)
    ax.axvspan(400, 550, color="#fde8e0", zorder=0)
    ax.axvspan(550, 800, color="#fff6e0", zorder=0)
    ax.axvspan(800, 1200, color="#f0f0f0", zorder=0)
    for key, color, label in [("fixed", C_FIXED, "Fixed / Split CP"),
                               ("aci", C_ACI, "Standard ACI"),
                               ("ours", C_OURS, "Drift-Aware ACI (ours)")]:
        arr = np.array(traces[key])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(mean, color=color, lw=2.0, label=label)
        ax.fill_between(np.arange(T), mean - std, mean + std, color=color, alpha=0.18, lw=0)
    ax.axhline(1 - ALPHA, color="#111111", ls=":", lw=1.4, label="Target coverage (0.90)")
    for x, txt in [(200, "Stationary"), (475, "Abrupt\nshift"),
                   (675, "Gradual\ndrift"), (1000, "New\nstationary")]:
        ax.text(x, 1.06, txt, ha="center", va="bottom", fontsize=8.5, color="#555555")
    ax.set_ylim(0.3, 1.12)
    ax.set_xlim(0, T)
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Rolling coverage (window=50)")
    ax.set_title(f"Synthetic validation, N={N_SEEDS} seeds (mean \u00b1 1 std)", fontsize=12, fontweight="bold", pad=26)
    ax.legend(loc="lower center", ncol=4, frameon=False, fontsize=8.7, bbox_to_anchor=(0.5, -0.26))
    plt.tight_layout()
    out_path = OUT_DIR / "fig1_synthetic_multiseed.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
