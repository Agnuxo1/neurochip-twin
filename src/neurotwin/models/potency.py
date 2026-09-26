"""Potency and activity derived from a dose x DIV trajectory forecast.

Developmental summary per feature (the NFA analogue of EPA's area-under-the-curve endpoints):
    A_f(c) = mean over DIV 5/7/9/12 of y_{f,DIV}(c)          (units: vehicle robust SD)
A chemical is *active* on feature f if |A_f| reaches the benchmark response BMR somewhere in the
tested range; its benchmark concentration BMC_f is the lowest log10 concentration at which |A_f|
first reaches BMR (linear interpolation on a fine log grid). The chemical-level potency is the
most sensitive feature: BMC = min_f BMC_f. BMR = 3 vehicle robust SD is fixed a priori.
"""
from __future__ import annotations

import numpy as np

from .trajectory import FEATURES, NF

BMR = 3.0


def dev_summary(traj: np.ndarray) -> np.ndarray:
    """(Q, ND, NF) trajectory -> (Q, NF) developmental mean effect."""
    return traj.mean(1)


def bmc_from_curve(grid: np.ndarray, A: np.ndarray, bmr: float = BMR):
    """grid (G,) log10 conc; A (G, NF). Returns (bmc_f (NF,) with nan if inactive, direction (NF,))."""
    bmc = np.full(NF, np.nan)
    sign = np.zeros(NF)
    for f in range(NF):
        a = np.abs(A[:, f])
        hit = np.where(a >= bmr)[0]
        if len(hit) == 0:
            continue
        i = hit[0]
        if i == 0:
            bmc[f] = grid[0]
        else:
            x0, x1, y0, y1 = grid[i - 1], grid[i], a[i - 1], a[i]
            bmc[f] = x0 + (bmr - y0) * (x1 - x0) / max(y1 - y0, 1e-9)
        sign[f] = np.sign(A[i, f])
    return bmc, sign


def chemical_potency(grid, A, bmr: float = BMR):
    bmc, sign = bmc_from_curve(grid, A, bmr)
    active = np.isfinite(bmc)
    if not active.any():
        return {"active": False, "bmc_log10": np.nan, "feature": None, "n_active_features": 0}
    f = int(np.nanargmin(bmc))
    return {"active": True, "bmc_log10": float(bmc[f]), "feature": FEATURES[f],
            "n_active_features": int(active.sum()), "direction": int(sign[f])}
