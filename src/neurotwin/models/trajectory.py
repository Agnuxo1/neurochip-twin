"""Dose x developmental-time trajectories of neural network formation (EPA NFA).

Each physical well is recorded at DIV 5, 7, 9 and 12 on a 48-well MEA plate.
We model the whole 4 x 17 trajectory of a well as a function of the log
concentration of the chemical applied to it.

Target transform (fixed a priori, fitted on vehicle wells only):
    g_f(x)   = asinh(x / c_f)                       variance-stabilising, defined at 0
    y_{f,d}  = (g_f(x) - med_veh(plate, date, d)) / s_{f,d}
where c_f is a quarter of the pooled DIV12 vehicle median of feature f,
med_veh is the median of g over the vehicle wells of the same plate, date and
DIV, and s_{f,d} is the pooled within-plate robust SD of g in vehicle wells.
Values are clipped to [-CLIP, CLIP]. Undefined raw values (e.g. burst duration
when a well has no bursts) are masked, never imputed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "firing_rate_mean", "burst_rate", "per_burst_interspike_interval",
    "per_burst_spike_percent", "burst_duration_mean", "interburst_interval_mean",
    "active_electrodes_number", "bursting_electrodes_number", "network_spike_number",
    "network_spike_peak", "spike_duration_mean", "per_network_spike_spike_percent",
    "inter_network_spike_interval_mean", "network_spike_duration_std",
    "per_network_spike_spike_number_mean", "correlation_coefficient_mean",
    "mutual_information_norm",
]
DIVS = [5, 7, 9, 12]
NF, ND = len(FEATURES), len(DIVS)
CLIP = 10.0
WELL_KEY = ["chemical", "spid", "plate", "date", "well", "concentration_uM"]


@dataclass
class Transform:
    c: dict[str, float]
    s: dict[str, list[float]]

    def to_json(self) -> dict:
        return {"c": self.c, "s": self.s, "clip": CLIP, "features": FEATURES, "divs": DIVS}


@dataclass
class ChemTask:
    chem: str
    fold: int
    logc: np.ndarray            # (n,) log10 µM
    y: np.ndarray               # (n, ND, NF) float32, 0 where masked
    m: np.ndarray               # (n, ND, NF) bool
    plate: np.ndarray           # (n,) plate|date
    label: str = "unknown"      # EPA DNT reference label if any
    levels: np.ndarray = field(default=None)

    def __post_init__(self):
        self.levels = np.unique(self.logc)
        self._lvidx = np.searchsorted(self.levels, self.logc)
        # cached masked mean trajectory per concentration level (all replicate wells)
        L = len(self.levels)
        num = np.zeros((L, ND, NF), np.float64)
        den = np.zeros((L, ND, NF), np.float64)
        np.add.at(num, self._lvidx, self.y * self.m)
        np.add.at(den, self._lvidx, self.m)
        self._ok = den > 0
        self._mu = np.where(self._ok, num / np.maximum(den, 1), 0.0).astype(np.float32)


def fit_transform(wells: pd.DataFrame) -> Transform:
    veh = wells[wells.concentration_uM == 0]
    c, s = {}, {}
    for f in FEATURES:
        v12 = veh.loc[(veh["div"] == 12) & (veh[f] > 0), f]
        c[f] = float(np.nanmedian(v12) / 4.0) if len(v12) else 1.0
        g = np.arcsinh(veh[f] / c[f])
        tmp = veh.assign(g=g)
        s_d = []
        for d in DIVS:
            grp = tmp[tmp["div"] == d].groupby(["plate", "date"])["g"]
            mad = grp.apply(lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))
                            if x.notna().sum() >= 3 else np.nan)
            val = float(np.nanmedian(mad.values)) if mad.notna().any() else 1.0
            s_d.append(max(val, 0.05))
        s[f] = s_d
    return Transform(c, s)


def transform_wells(wells: pd.DataFrame, tr: Transform) -> pd.DataFrame:
    out = wells.copy()
    for f in FEATURES:
        out["g_" + f] = np.arcsinh(out[f] / tr.c[f])
    veh = out[out.concentration_uM == 0]
    med = veh.groupby(["plate", "date", "div"])[["g_" + f for f in FEATURES]].median()
    out = out.join(med, on=["plate", "date", "div"], rsuffix="_veh")
    for f in FEATURES:
        s = out["div"].map(dict(zip(DIVS, tr.s[f])))
        out["y_" + f] = ((out["g_" + f] - out["g_" + f + "_veh"]) / s).clip(-CLIP, CLIP)
    return out


def load_tasks(root: Path, include_vehicle: bool = False):
    """Return (tasks, transform). One ChemTask per chemical with its fold."""
    wells = pd.read_parquet(root / "data/processed/epa_nfa/wells.parquet")
    splits = json.loads((root / "repo/results/splits_epa_nfa.json").read_text(encoding="utf-8"))
    fold_of = {}
    lab_of = {}
    for a in splits["assignments"]:
        fold_of[a["chemical"]] = a["fold"]
        lab = str(a.get("label", "")).strip().lower()
        if lab in ("positive", "negative"):
            lab_of[a["chemical"]] = lab
    tr = fit_transform(wells)
    tw = transform_wells(wells, tr)
    if not include_vehicle:
        tw = tw[tw.concentration_uM > 0]
    ycols = ["y_" + f for f in FEATURES]
    tasks = []
    for chem, g in tw.groupby("chemical", sort=True):
        if chem not in fold_of:
            continue
        piv = {}
        for key, gw in g.groupby(WELL_KEY, sort=True):
            arr = np.full((ND, NF), np.nan, np.float32)
            for _, r in gw.iterrows():
                if r["div"] in DIVS:
                    arr[DIVS.index(r["div"])] = r[ycols].to_numpy(np.float32)
            piv[key] = arr
        keys = list(piv)
        Y = np.stack([piv[k] for k in keys])
        M = np.isfinite(Y)
        tasks.append(ChemTask(
            chem=chem, fold=int(fold_of[chem]),
            logc=np.log10(np.array([k[5] for k in keys], dtype=np.float64)).astype(np.float32),
            y=np.nan_to_num(Y, nan=0.0).astype(np.float32), m=M,
            plate=np.array([f"{k[2]}|{k[3]}" for k in keys]),
            label=lab_of.get(chem, "unknown")))
    return tasks, tr


# ------------------------------------------------------------------ episodes
def split_context(task: ChemTask, ctx_levels: np.ndarray):
    """Context = every well at the chosen concentration levels; target = the rest."""
    is_ctx = np.isin(task.logc, ctx_levels)
    return is_ctx, ~is_ctx


def level_means(task: ChemTask, sel: np.ndarray):
    """Masked mean trajectory per concentration level among selected wells."""
    rows = np.unique(task._lvidx[sel])
    if np.array_equal(np.isin(task._lvidx, rows), sel):     # whole levels selected -> cached
        return task.levels[rows], task._mu[rows], task._ok[rows]
    lv = np.unique(task.logc[sel])
    mu = np.zeros((len(lv), ND, NF), np.float32)
    ok = np.zeros((len(lv), ND, NF), bool)
    for i, l in enumerate(lv):
        idx = sel & (task.logc == l)
        num = (task.y[idx] * task.m[idx]).sum(0)
        den = task.m[idx].sum(0)
        ok[i] = den > 0
        mu[i] = np.where(ok[i], num / np.maximum(den, 1), 0.0)
    return lv, mu, ok


# ------------------------------------------------------------------ baselines
def predict_zero(task, is_ctx, q_logc):
    return np.zeros((len(q_logc), ND, NF), np.float32)


def predict_ctx_mean(task, is_ctx, q_logc):
    if not is_ctx.any():
        return predict_zero(task, is_ctx, q_logc)
    num = (task.y[is_ctx] * task.m[is_ctx]).sum(0)
    den = task.m[is_ctx].sum(0)
    mu = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return np.broadcast_to(mu, (len(q_logc), ND, NF)).astype(np.float32).copy()


def interp_rows(lv, mu, ok, q):
    """Vectorised piecewise-linear interpolation of level means (L, ...) at queries q (flat outside),
    ignoring unobserved entries; returns (Q, ...) with 0 where a dimension has no observation."""
    q = np.asarray(q, np.float64)
    shp = mu.shape[1:]
    M, O = mu.reshape(len(lv), -1), ok.reshape(len(lv), -1)
    out = np.zeros((len(q), M.shape[1]), np.float32)
    full = O.all(0)
    if full.any():
        if len(lv) == 1:
            out[:, full] = M[0, full]
        else:
            i = np.clip(np.searchsorted(lv, q), 1, len(lv) - 1)
            x0, x1 = lv[i - 1], lv[i]
            w = np.clip((q - x0) / np.where(x1 > x0, x1 - x0, 1), 0, 1)[:, None]
            out[:, full] = (M[i - 1][:, full] * (1 - w) + M[i][:, full] * w)
    for j in np.where(~full & O.any(0))[0]:
        g = O[:, j]
        out[:, j] = np.interp(q, lv[g], M[g, j])
    return out.reshape((len(q),) + shp)


def predict_interp(task, is_ctx, q_logc):
    """Piecewise-linear in log10 concentration between context level means, flat outside."""
    if not is_ctx.any():
        return predict_zero(task, is_ctx, q_logc)
    lv, mu, ok = level_means(task, is_ctx)
    return interp_rows(lv, mu, ok, q_logc)


_AC50 = np.linspace(-3.5, 2.5, 31)
_HILL = np.array([0.7, 1.0, 1.5, 2.5, 4.0])


def predict_hill(task, is_ctx, q_logc):
    """Hill curve y = top * c^h / (c^h + ac50^h) (bottom fixed at vehicle = 0), fitted per
    output dimension by exhaustive search over (log ac50, h) with the linear 'top' solved in
    closed form (weighted by replicate counts). Needs >= 3 context levels, else falls back
    to piecewise-linear interpolation."""
    lv, mu, ok = level_means(task, is_ctx) if is_ctx.any() else (np.array([]), None, None)
    if len(lv) < 3:
        return predict_interp(task, is_ctx, q_logc)
    # basis[a, h, level] for context levels and for queries
    def basis(x):
        xx = x[None, None, :]
        return 1.0 / (1.0 + 10 ** (_HILL[None, :, None] * (_AC50[:, None, None] - xx)))
    Bc, Bq = basis(lv), basis(np.asarray(q_logc, np.float64))
    out = np.zeros((len(q_logc), ND, NF), np.float32)
    Y = mu.reshape(len(lv), -1)            # (L, 68)
    W = ok.reshape(len(lv), -1).astype(np.float64)
    num = np.einsum("ahl,lk->ahk", Bc, Y * W)
    den = np.einsum("ahl,lk->ahk", Bc ** 2, W) + 1e-9
    top = np.clip(num / den, -CLIP, CLIP)  # (A, H, 68)
    resid = np.einsum("lk,lk->k", W, Y ** 2)[None, None] - 2 * top * num + top ** 2 * den
    flat = resid.reshape(-1, Y.shape[1]).argmin(0)
    a_i, h_i = np.unravel_index(flat, resid.shape[:2])
    k = np.arange(Y.shape[1])
    pred = Bq[a_i, h_i, :].T * top[a_i, h_i, k][None, :]
    return pred.reshape(len(q_logc), ND, NF).astype(np.float32)


class AnalogKNN:
    """Few-shot analog baseline: find training chemicals whose (interpolated) responses at the
    context concentrations are closest to the test chemical's context, and average their
    responses at the query concentrations. With no context it returns the population mean."""

    def __init__(self, train: list[ChemTask], k: int = 10):
        self.k = k
        self.grid = np.linspace(-3.0, 2.5, 23)
        prof = []
        for t in train:
            prof.append(interp_rows(t.levels, t._mu, t._ok, self.grid))
        self.P = np.stack(prof)            # (T, G, ND, NF)

    def _at(self, x):
        idx = np.clip(np.searchsorted(self.grid, x), 1, len(self.grid) - 1)
        x0, x1 = self.grid[idx - 1], self.grid[idx]
        w = np.clip((x - x0) / (x1 - x0), 0, 1)
        return self.P[:, idx - 1] * (1 - w)[None, :, None, None] + self.P[:, idx] * w[None, :, None, None]

    def predict(self, task, is_ctx, q_logc):
        q = np.asarray(q_logc, np.float64)
        if not is_ctx.any():
            return self._at(q).mean(0).astype(np.float32)
        lv, mu, ok = level_means(task, is_ctx)
        A = self._at(lv)                   # (T, L, ND, NF)
        d2 = (((A - mu[None]) ** 2) * ok[None]).sum((1, 2, 3)) / max(ok.sum(), 1)
        nn = np.argsort(d2)[: self.k]
        w = 1.0 / (d2[nn] + 1e-3)
        return (np.tensordot(w / w.sum(), self._at(q)[nn], axes=1)).astype(np.float32)


def masked_mae(pred, task, sel):
    m = task.m[sel]
    return float(np.abs(pred - task.y[sel])[m].mean()) if m.any() else float("nan")
