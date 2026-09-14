"""
calibration_methods.py
-----------------------
The four online conformal-calibration methods compared throughout this
project. Each function takes a 1D array of nonconformity scores (a real
number per stream item, higher = more uncertain) and returns the per-step
miscoverage indicator array `err` (1 = the item was NOT covered by the
current threshold) and, where relevant, the threshold trace `q`.

All four methods share the same target miscoverage rate ALPHA (i.e. target
coverage 1 - ALPHA) and the same calibration-window size, so they are
directly comparable on any score stream.

Methods
-------
fixed_cp        : classical split conformal prediction, threshold frozen
                  after an initial calibration window (the "does nothing
                  about distribution shift" baseline).
standard_aci    : Adaptive Conformal Inference (Gibbs & Candes, 2021),
                  fixed step size.
drift_aware_aci : this paper's method -- step size scales with an estimated
                  distributional-shift signal.
pid_aci         : Conformal PI control (Angelopoulos, Candes & Tibshirani,
                  2023), the P+I configuration (integral term only; the D /
                  "scorecaster" term is omitted since it requires training
                  a separate per-stream forecasting model).

See the paper (Sec. 3.3, Sec. 5.5) for the full derivations and discussion.
"""

import numpy as np


def fixed_cp(scores, alpha=0.30, calib_n=30):
    """Split conformal prediction: calibrate once, freeze the threshold."""
    q0 = np.quantile(scores[:calib_n], 1 - alpha)
    err = (scores > q0).astype(float)
    q = np.full(len(scores), q0)
    return err, q


def standard_aci(scores, alpha=0.30, calib_n=30, gamma=0.03):
    """Adaptive Conformal Inference (Gibbs & Candes, 2021), fixed step size.

    q_{t+1} = q_t + gamma * (err_t - alpha)
    """
    T = len(scores)
    q0 = np.quantile(scores[:calib_n], 1 - alpha)
    q = np.zeros(T)
    q[0] = q0
    err = np.zeros(T)
    for t in range(T):
        err[t] = float(scores[t] > q[t])
        if t + 1 < T:
            q[t + 1] = max(0.0, q[t] + gamma * (err[t] - alpha))
    return err, q


def drift_aware_aci(scores, alpha=0.30, calib_n=30, gamma0=0.03, lam=4.0,
                     gamma_min=0.005, gamma_max=0.30, window=25):
    """This paper's method: step size scales with an estimated drift signal.

    q_{t+1} = q_t + gamma_t * (err_t - alpha)
    gamma_t = clip(gamma0 * (1 + lam * D_t), gamma_min, gamma_max)
    D_t     = |mean(recent window of scores) - calib_mean| / calib_std

    See Sec. 3.3 (definition) and Sec. 3.4 (Theorem 1: a finite-sample
    coverage bound for the gamma-weighted miscoverage rate) of the paper.
    """
    T = len(scores)
    calib = scores[:calib_n]
    q0 = np.quantile(calib, 1 - alpha)
    calib_mean, calib_std = calib.mean(), calib.std() + 1e-6

    q = np.zeros(T)
    q[0] = q0
    err = np.zeros(T)
    gamma_trace = np.zeros(T)
    for t in range(T):
        err[t] = float(scores[t] > q[t])
        lo = max(0, t - window)
        recent_mean = scores[lo:t + 1].mean()
        D_t = abs(recent_mean - calib_mean) / calib_std
        gamma_t = np.clip(gamma0 * (1 + lam * D_t), gamma_min, gamma_max)
        gamma_trace[t] = gamma_t
        if t + 1 < T:
            q[t + 1] = max(0.0, q[t] + gamma_t * (err[t] - alpha))
    return err, q, gamma_trace


def pid_aci(scores, alpha=0.30, calib_n=30, K_I=None, C_sat=None):
    """Conformal PI control (Angelopoulos, Candes & Tibshirani, NeurIPS 2023),
    P+I configuration -- the default they recommend in most experiments.
    The D ("scorecaster") term is omitted; it requires training a separate
    per-stream forecasting model, out of scope for this comparison.

    q_{t+1} = q0 + r_t( sum_{i=1}^t (err_i - alpha) )
    r_t(x)  = K_I * tan( x * log(t+1) / ((t+1) * C_sat) ), saturating at
              +-inf as |x| -> pi/2 (implemented as a large finite clip).

    C_sat and K_I have no universal default in the source paper; here they
    are set via a data-driven heuristic (a robust scale of the calibration
    window). A dedicated hyperparameter search may improve on this -- see
    Sec. 5.5 ("what remains open") in the paper.
    """
    T = len(scores)
    calib = scores[:calib_n]
    q0 = np.quantile(calib, 1 - alpha)
    if C_sat is None:
        C_sat = max(calib.max(), 1e-6)
    if K_I is None:
        K_I = 2.0 * (calib.std() + 1e-6)

    q = np.zeros(T)
    q[0] = q0
    err = np.zeros(T)
    integral = 0.0
    for t in range(T):
        err[t] = float(scores[t] > q[t])
        integral += (err[t] - alpha)
        arg = integral * np.log(t + 2) / ((t + 1) * C_sat)
        if abs(arg) >= np.pi / 2:
            r_t = np.sign(arg) * 1e3
        else:
            r_t = K_I * np.tan(arg)
        if t + 1 < T:
            q[t + 1] = max(0.0, q0 + r_t)
    return err, q


def coverage(err, a=None, b=None):
    """Empirical coverage (1 - miscoverage rate) over err[a:b]."""
    seg = err[a:b] if (a is not None or b is not None) else err
    return 1 - seg.mean()


METHODS = {
    "Fixed / Split CP": fixed_cp,
    "Standard ACI": standard_aci,
    "PID-ACI": pid_aci,
    "Drift-Aware ACI (ours)": drift_aware_aci,
}
