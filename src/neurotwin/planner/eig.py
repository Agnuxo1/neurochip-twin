"""DoseCompass: choose the next concentration of a chemical by expected information about its potency.

State. A chemical has tested concentration levels (the support); some are *measured* (every replicate
well at that level, DIV 5-12) and the rest are candidates. The quantity of interest is the
full-information potency
    P = (active?, BMC)   neurotwin.models.potency applied to the level means at ALL tested levels,
interpolated linearly in log10 dose on a 120-point grid (the reference convention of R3).

Predictive samples. DoseCompass draws S plausible full-information datasets consistent with what has
been measured (developmental summary A = mean over DIV, units of vehicle robust SD):
    A_s(l) = observed summary at l, from measured wells only             l measured (fixed)
    A_s(l) = mu_A^(m_s)(l) + tau * sd_A^(m_s)(l) * e_s(l)                   l not measured
where m_s cycles over the ensemble members (NeuroTrajectory predictive given the measured context),
mu_A / sd_A are the DIV-means of the member's predicted location and standard deviation, and
e_s ~ GP(0, exp(-(x - x')^2 / (2 ell^2))) over log10 dose, independent across features. Each sample
yields P_s by the reference rule.

Information. Measuring candidate u reveals A(u); its outcome is summarised as
O_s(u) = bin of max_f |A_s(u, f)| / BMR with edges (0.5, 1, 2). With P_bin = 'inactive' or BMC in
0.25-log10 bins,
    EIG(u) = I(P_bin ; O(u))      (plug-in mutual information over the S samples, in bits)
and the next level is argmax EIG (ties: farther from the measured levels, then lower concentration).
The planner reads only wells at measured levels; the positions of the candidate levels are known a
priori (they are the tested concentrations), their responses are not.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..models.potency import BMR, chemical_potency, dev_summary
from ..models.trajectory import DIVS, ND, NF, ChemTask, interp_rows, level_means

GRID_POINTS = 120
BIN_WIDTH = 0.25                    # log10 units, potency variable
OUTCOME_EDGES = (0.5, 1.0, 2.0)     # multiples of BMR for max_f |A_f(level)|
SD_FACTOR = {"gaussian": 1.0, "laplace": float(np.sqrt(2.0))}   # predicted scale -> standard deviation
DEFAULT_TAU = 1.0
DEFAULT_ELL = 0.5
DEFAULT_SAMPLES = 2000
LEVEL_TOL = 1e-4


# ------------------------------------------------------------------ potency helpers
def reference_grid(levels) -> np.ndarray:
    levels = np.asarray(levels)
    return np.linspace(levels.min(), levels.max(), GRID_POINTS).astype(np.float32)


def full_information_potency(task: ChemTask) -> dict:
    """Reference potency: the rule applied to the level means at ALL tested levels (R3 convention)."""
    grid = reference_grid(task.levels)
    return chemical_potency(grid, dev_summary(interp_rows(task.levels, task._mu, task._ok, grid)))


def interp_matrix(levels, grid) -> np.ndarray:
    """(G, L) weights of piecewise-linear interpolation between levels (flat outside)."""
    levels = np.asarray(levels, np.float64)
    grid = np.asarray(grid, np.float64)
    L = len(levels)
    if L == 1:
        return np.ones((len(grid), 1))
    W = np.zeros((len(grid), L))
    for j in range(L):
        e = np.zeros(L)
        e[j] = 1.0
        W[:, j] = np.interp(grid, levels, e)
    return W


def bmc_batch(grid, A, bmr: float = BMR):
    """Vectorised potency rule (same as potency.chemical_potency). A (..., G, NF) summary on grid (G,).
    Returns (active (...,) bool, bmc (...,) log10 with nan where inactive)."""
    grid = np.asarray(grid, np.float64)
    A = np.asarray(A)
    a = np.abs(A if A.dtype in (np.float32, np.float64) else A.astype(np.float64))
    hit = a >= bmr
    anyf = hit.any(-2)
    i = hit.argmax(-2)
    im1 = np.maximum(i - 1, 0)
    y0 = np.take_along_axis(a, im1[..., None, :], -2)[..., 0, :].astype(np.float64)
    y1 = np.take_along_axis(a, i[..., None, :], -2)[..., 0, :].astype(np.float64)
    x0, x1 = grid[im1], grid[i]
    b = np.where(i == 0, grid[0], x0 + (bmr - y0) * (x1 - x0) / np.maximum(y1 - y0, 1e-9))
    b = np.where(anyf, b, np.inf)
    active = anyf.any(-1)
    return active, np.where(active, b.min(-1), np.nan)


def potency_codes(active, bmc, lo: float, hi: float, width: float = BIN_WIDTH):
    """Integer code of the potency variable: BMC bin (width log10 from lo) or 'inactive' (last code).
    Returns (codes, n_codes)."""
    K = int(np.floor((hi - lo) / width + 1e-9)) + 1
    b = np.nan_to_num(np.asarray(bmc, np.float64), nan=lo)
    k = np.clip(np.floor((b - lo) / width + 1e-9).astype(int), 0, K - 1)
    return np.where(np.asarray(active, bool), k, K), K + 1


def mutual_info_bits(x, Y) -> np.ndarray:
    """Plug-in mutual information (bits) between integer codes x (S,) and each column of Y (S, C)."""
    x = np.asarray(x)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    S = len(x)
    _, xi = np.unique(x, return_inverse=True)
    out = np.zeros(Y.shape[1])
    for c in range(Y.shape[1]):
        _, yi = np.unique(Y[:, c], return_inverse=True)
        J = np.zeros((xi.max() + 1, yi.max() + 1))
        np.add.at(J, (xi.ravel(), yi.ravel()), 1.0)
        P = J / S
        pxy = P.sum(1, keepdims=True) * P.sum(0, keepdims=True)
        nz = P > 0
        out[c] = float((P[nz] * np.log2(P[nz] / pxy[nz])).sum())
    return np.maximum(out, 0.0)


def log_score_bits(codes, target_code: int, n_codes: int) -> float:
    """log2 predictive probability of the target potency code (add-1/2 smoothing)."""
    codes = np.asarray(codes)
    return float(np.log2(((codes == target_code).sum() + 0.5) / (len(codes) + 0.5 * n_codes)))


# ------------------------------------------------------------------ model access
def load_fold_models(models_dir, fold: int, device="cpu"):
    """NeuroTrajectory ensemble of one outer fold (score a chemical only with the models of ITS fold).
    The checkpoint cfg decides the architecture (neurotwin.models.cnp.load_model); the likelihood is kept
    on each model so that its predicted scale can be turned into a standard deviation."""
    from ..models.cnp import load_model
    ms = []
    for p in sorted(Path(models_dir).glob(f"cnp_fold{fold}_seed*.pt")):
        m, cfg = load_model(p, device)
        m.likelihood = cfg.get("likelihood", "gaussian")
        ms.append(m)
    if not ms:
        raise FileNotFoundError(f"no cnp_fold{fold}_seed*.pt in {models_dir}")
    return ms


def member_predictions(models, task: ChemTask, is_ctx: np.ndarray, q, device="cpu"):
    """Per-member predictive location and standard deviation at queries q: two (M, Q, ND, NF) arrays."""
    from ..models.cnp import predict
    q = np.asarray(q, np.float32)
    mus, sds = [], []
    for m in models:
        mu, sc = predict(m, task, is_ctx, q, device=device)
        mus.append(mu)
        sds.append(sc * SD_FACTOR.get(getattr(m, "likelihood", "gaussian"), 1.0))
    return np.stack(mus).astype(np.float64), np.stack(sds).astype(np.float64)


def mask_divs(task: ChemTask, keep=(5, 7, 9)) -> ChemTask:
    """View of a chemical in which only the given DIVs have been recorded (the rest masked)."""
    keep_idx = [DIVS.index(d) for d in keep]
    m = np.zeros_like(task.m)
    m[:, keep_idx] = task.m[:, keep_idx]
    y = np.where(m, task.y, 0.0).astype(np.float32)
    return ChemTask(task.chem, task.fold, task.logc.copy(), y, m, task.plate.copy(), task.label)


# ------------------------------------------------------------------ core
def _wells_at(task: ChemTask, levels) -> np.ndarray:
    sel = np.zeros(len(task.logc), bool)
    for lv in np.asarray(levels, np.float64).ravel():
        hit = np.abs(task.logc.astype(np.float64) - lv) < LEVEL_TOL
        if not hit.any():
            raise ValueError(f"no measured wells at level {lv}")
        sel |= hit
    return sel


def _unique_levels(x) -> np.ndarray:
    x = np.sort(np.asarray(x, np.float64).ravel())
    if len(x) == 0:
        return x
    keep = np.concatenate([[True], np.diff(x) > LEVEL_TOL])
    return x[keep]


def _seed(chem: str, measured, seed: int) -> int:
    key = f"{chem}|{','.join(f'{v:.5f}' for v in sorted(np.asarray(measured, float).ravel()))}|{seed}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def measured_summary(task: ChemTask, is_ctx: np.ndarray, fill: np.ndarray):
    """Developmental summary (Lm, NF) at the measured levels from measured wells only. An entry unobserved
    at one level is interpolated from the other measured levels (as the reference does); an entry never
    observed at any measured level (e.g. DIV 12 before it is recorded) takes `fill` (Lm, ND, NF)."""
    lv, mu, ok = level_means(task, is_ctx)
    T = interp_rows(lv, mu, ok, lv).astype(np.float64)
    never = ~ok.any(0)
    T[:, never] = fill[:, never]
    return lv, T.mean(1)


def sample_datasets(support, meas_mask, A_meas, mu_m, sd_m, tau, ell, n_samples, rng):
    """S plausible full-information developmental summaries (S, L, NF) over the support levels."""
    support = np.asarray(support, np.float64)
    M, L = mu_m.shape[0], len(support)
    A = np.empty((n_samples, L, NF))
    A[:, meas_mask] = A_meas[None]
    U = np.where(~meas_mask)[0]
    if len(U):
        x = support[U]
        R = np.exp(-(x[:, None] - x[None]) ** 2 / (2.0 * ell ** 2)) + 1e-6 * np.eye(len(U))
        C = np.linalg.cholesky(R)
        z = rng.standard_normal((n_samples, len(U), NF))
        e = np.einsum("ij,sjf->sif", C, z)
        member = np.arange(n_samples) % M
        muA = mu_m[:, U].mean(2)           # (M, nU, NF) DIV-mean location
        sdA = sd_m[:, U].mean(2)           # (M, nU, NF) DIV-mean of per-DIV standard deviation
        A[:, U] = muA[member] + tau * sdA[member] * e
    return A


@dataclass
class Posterior:
    support: np.ndarray
    meas_mask: np.ndarray
    samples: np.ndarray        # (S, L, NF)
    active: np.ndarray         # (S,)
    bmc: np.ndarray            # (S,)
    codes: np.ndarray          # (S,)
    n_codes: int


def posterior_from_predictions(task, is_ctx, support, mu_m, sd_m, tau=DEFAULT_TAU, ell=DEFAULT_ELL,
                               n_samples=DEFAULT_SAMPLES, seed=0) -> Posterior:
    """Sampled posterior over full-information datasets and potency, given member predictions at the
    support (M, L, ND, NF). Reads only the wells selected by is_ctx."""
    support = np.asarray(support, np.float64)
    meas_lv = _unique_levels(task.logc[is_ctx])
    meas_mask = np.array([np.any(np.abs(meas_lv - s) < LEVEL_TOL) for s in support], bool)
    if meas_mask.sum() != len(meas_lv):
        raise ValueError("every measured level must belong to the support")
    A_meas = np.zeros((0, NF))
    if len(meas_lv):
        _, A_meas = measured_summary(task, is_ctx, mu_m[:, meas_mask].mean(0))
    rng = np.random.default_rng(_seed(task.chem, meas_lv, seed))
    A = sample_datasets(support, meas_mask, A_meas, mu_m, sd_m, tau, ell, n_samples, rng)
    grid = reference_grid(support)
    Ag = np.matmul(interp_matrix(support, grid).astype(np.float32), A.astype(np.float32))
    active, bmc = bmc_batch(grid, Ag)
    codes, n_codes = potency_codes(active, bmc, support.min(), support.max())
    return Posterior(support, meas_mask, A, active, bmc, codes, n_codes)


def eig_of_candidates(post: Posterior, cand_idx) -> np.ndarray:
    """Expected information (bits) about the potency code from measuring each candidate support index."""
    cand_idx = np.asarray(cand_idx, int)
    if len(cand_idx) == 0:
        return np.zeros(0)
    out = np.abs(post.samples[:, cand_idx]).max(-1) / BMR
    return mutual_info_bits(post.codes, np.digitize(out, OUTCOME_EDGES))


@dataclass
class PlanStep:
    chosen: float | None
    eig_bits: dict = field(default_factory=dict)
    p_active: float = float("nan")
    bmc_quantiles: list | None = None        # 5 / 50 / 95 % of BMC among active samples
    measured: list = field(default_factory=list)
    tau: float = DEFAULT_TAU
    ell: float = DEFAULT_ELL


def choose(post: Posterior, cand_idx, eig) -> float | None:
    if len(cand_idx) == 0:
        return None
    meas = post.support[post.meas_mask]
    x = post.support[np.asarray(cand_idx, int)]
    dist = np.abs(x[:, None] - meas[None]).min(1) if len(meas) else np.zeros(len(x))
    order = np.lexsort((x, -dist, -np.round(eig, 9)))
    return float(x[order[0]])


def plan_step(models, task: ChemTask, measured_levels, candidate_levels=None, device="cpu",
              tau=DEFAULT_TAU, ell=DEFAULT_ELL, n_samples=DEFAULT_SAMPLES, seed=0) -> PlanStep:
    measured = _unique_levels(measured_levels)
    if len(measured) == 0:
        raise ValueError("DoseCompass needs at least one measured level (start from a fixed first level)")
    is_ctx = _wells_at(task, measured)
    if candidate_levels is None:
        cands = np.array([l for l in _unique_levels(task.levels) if np.abs(measured - l).min(initial=9) > LEVEL_TOL])
    else:
        cands = _unique_levels(candidate_levels)
        cands = cands[[np.abs(measured - c).min(initial=9) > LEVEL_TOL for c in cands]] if len(cands) else cands
    support = _unique_levels(np.concatenate([np.asarray(task.levels, np.float64), cands]))
    mu_m, sd_m = member_predictions(models, task, is_ctx, support, device)
    post = posterior_from_predictions(task, is_ctx, support, mu_m, sd_m, tau, ell, n_samples, seed)
    cand_idx = [int(np.argmin(np.abs(support - c))) for c in cands]
    eig = eig_of_candidates(post, cand_idx)
    act = post.active
    q = [float(v) for v in np.percentile(post.bmc[act], [5, 50, 95])] if act.any() else None
    return PlanStep(chosen=choose(post, cand_idx, eig),
                    eig_bits={float(support[i]): float(e) for i, e in zip(cand_idx, eig)},
                    p_active=float(act.mean()), bmc_quantiles=q, measured=[float(v) for v in measured],
                    tau=tau, ell=ell)


def recommend_next(models, task: ChemTask, measured_levels, candidate_levels=None, device="cpu",
                   tau=DEFAULT_TAU, ell=DEFAULT_ELL, n_samples=DEFAULT_SAMPLES, seed=0):
    """Next concentration (log10 uM) to measure and the expected information (bits) of every candidate.

    models: NeuroTrajectory ensemble that never saw this chemical (e.g. load_fold_models of its fold);
    task: the chemical's wells (only wells at measured_levels are read); candidate_levels: log10 uM
    (default: the chemical's tested, not-yet-measured levels). Returns (chosen_level, {level: EIG bits})."""
    st = plan_step(models, task, measured_levels, candidate_levels, device, tau, ell, n_samples, seed)
    return st.chosen, st.eig_bits


def sequential_design(models, task: ChemTask, first_level: float, budget: int, device="cpu",
                      planner_task: ChemTask | None = None, **kw):
    """Greedy DoseCompass sequence of `budget` levels starting at first_level. planner_task is the view
    of the data available when deciding (default: task). Returns (levels in order, PlanStep per state);
    the last PlanStep describes the posterior after all `budget` levels."""
    view = planner_task if planner_task is not None else task
    seq = [float(first_level)]
    steps = []
    while True:
        cands = [float(l) for l in task.levels if min(abs(float(l) - s) for s in seq) > LEVEL_TOL]
        st = plan_step(models, view, seq, cands, device, **kw)
        steps.append(st)
        if len(seq) >= budget or st.chosen is None:
            break
        seq.append(st.chosen)
    return seq, steps
