"""Appendix: the team lead's own prior architectures, evaluated on real data.

Three of Francisco Angulo de Lafuente's earlier architectures were tested as candidate components of
NeuroChip Twin v2 on the same real data used by the main pipeline. None of them is used in the
headline model; this script reproduces the tests and reports them as parity or negative results.

  A  Eikonal (geodesic arrival-time) propagation model on the chip geometry, fitted to network-burst
     latency maps of the Brewer 4-compartment MEA (source: the eikonal solver repository, MIT).
  B  QESN lattice reservoir on the chip geometry (classical, Schrodinger-inspired lattice reservoir,
     corrected to an exactly unitary Crank-Nicolson step, with leak and a Kerr-type phase
     nonlinearity) forecasting Brewer NoStim spiking against persistence, a ridge VAR and a non-spatial
     echo-state network with the same number of readout features (source: QESN_MABe_V2_REPO, Apache-2.0).
  C  DoseLattice: the QESN mass-conserving diffusion stencil run over log-dose as a parameter-free
     encoder of the context set inside NeuroTrajectory. DEVELOPMENT FOLD 0 ONLY (the fold excluded
     from the reported folds 1-4); reference = the v1 cross-validated fold-0 models, the primary
     models (data/processed/models).

The code of A, B and C is a faithful port of the exploratory pilots written while deciding which of
these architectures to integrate (not published). A is rerun in full and checked byte-for-byte
against the pilot JSON. B is the expensive part (about 24 min of CPU for the 9 recordings in the pilot run): by default the
pilot's per-recording numbers are reused after this port reproduces the first two recordings
(byte-identical JSON serialisation, wall-clock field excluded) and the pilot summary is recomputed
byte-identically from them; `--b-mode full` reruns all 9 recordings. C is rerun with the v1 fold-0
configuration (the pilot used the v2 configuration).

Usage (from the workspace root; set PYTHONPATH to repo/src):
  python repo/scripts/appendix_own_architectures.py --parts A          # < 1 min CPU
  python repo/scripts/appendix_own_architectures.py --parts B          # reduced check, ~6 min CPU
  python repo/scripts/appendix_own_architectures.py --parts B --b-mode full   # ~24 min CPU
  python repo/scripts/appendix_own_architectures.py --parts C          # ~25 min GPU (RTX 3090, shared)
  python repo/scripts/appendix_own_architectures.py --summarize        # -> results/appendix_own_arch.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "repo"
RESULTS = REPO / "results"
PARTS = RESULTS / "appendix_own_arch_parts"
PILOT_DIR = ROOT / "colab/research/fran/_pilot"
BREWER = ROOT / "data/processed/brewer"
TASKS = ROOT / "data/processed/epa_nfa/tasks_cache.pkl"
OUT_NAME = "appendix_own_arch.json"

ROWS = "ABCDEFGHJKLM"
COMPARTMENTS = ("EC", "DG", "CA3", "CA1")

SOURCES = {
    "A": {"author": "Francisco Angulo de Lafuente",
          "repo": "Agnuxo1/Optical-Neuromorphic-Computing-for-Real-Time-Pathfinding-A-GPU-Accelerated-Eikonal-Solver",
          "licence": "MIT", "archive": "Zenodo record 17583729",
          "what_is_reused": ("the model class only: constant-speed eikonal arrival times on the device mask, which "
                             "equal geodesic distance divided by speed; solved here with 8-neighbour Dijkstra on a "
                             "48 x 48 lattice. The repository's GPU relaxation solver is not executed.")},
    "B": {"author": "Francisco Angulo de Lafuente", "repo": "Agnuxo1/QESN_MABe_V2_REPO", "licence": "Apache-2.0",
          "archive": "Zenodo record 17266084; PyPI qesn-mabe",
          "what_is_reused": ("the 2-D complex lattice reservoir with a linear readout, corrected: an exactly unitary "
                             "Crank-Nicolson step replaces the original 5-point update (which does not conserve "
                             "sum |psi|^2), plus explicit leak and a Kerr-type phase nonlinearity. Classical, "
                             "Schrodinger-inspired; nothing quantum is computed.")},
    "C": {"author": "Francisco Angulo de Lafuente", "repo": "Agnuxo1/QESN_MABe_V2_REPO", "licence": "Apache-2.0",
          "archive": "Zenodo record 17266084; PyPI qesn-mabe",
          "what_is_reused": ("the mass-conserving diffusion stencil of src/qesn/diffusion.py, in 1-D over log10 "
                             "concentration, as a fixed (parameter-free) encoder of the context wells.")},
}

VERDICT_WORDS = ("matches but does not beat", "worse", "no gain; worse at some k", "no gain over interpolation",
                 "better", "mixed")


# ================================================================== utilities
def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def as_written(text: str) -> bytes:
    """Bytes that Path.write_text / open(..., 'w') produce for `text` on this platform (the pilots wrote
    their JSON in text mode, so on Windows every newline is CRLF)."""
    return text.replace("\n", os.linesep).encode("utf-8")


def same_as_file(text: str, p: Path) -> bool:
    return bool(p.exists() and p.read_bytes() == as_written(text))


def max_rel_diff(a, b) -> float:
    """Largest relative difference between numeric leaves of two JSON-like objects (inf if the
    structures or non-numeric leaves differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return float("inf")
        return max([max_rel_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max([max_rel_diff(x, y) for x, y in zip(a, b)] or [0.0])
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        if a == b or (np.isnan(a) and np.isnan(b)):
            return 0.0
        return float(abs(a - b) / max(abs(b), 1e-300))
    return 0.0 if a == b else float("inf")


def env_info() -> dict:
    import scipy
    info = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch optional for A/B
        pass
    return info


def write_part(name: str, obj: dict, suffix: str = "") -> Path:
    PARTS.mkdir(parents=True, exist_ok=True)
    p = PARTS / f"{name}{suffix}.json"
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return p


def load_brewer():
    w = pd.read_parquet(BREWER / "wells.parquet")
    ax = pd.read_parquet(BREWER / "axons.parquet")
    return w, ax


# ================================================================== A: eikonal / geodesic latency
A_S, A_OFF = 4, 2
A_H = A_W = 12 * A_S
A_BW = 0.005


def a_mask(mode):
    m = np.zeros((A_H, A_W), bool)
    for r0 in (0, 7):
        for c0 in (0, 7):
            m[r0 * A_S:(r0 + 5) * A_S, c0 * A_S:(c0 + 5) * A_S] = True
    tun = np.zeros((A_H, A_W), bool)
    if mode == "chip":   # 20 one-cell tunnels in rows/cols F-G and 6-7, as on the device
        for r in list(range(5)) + list(range(7, 12)):
            tun[r * A_S + A_OFF, 5 * A_S:7 * A_S] = True
        for c in list(range(5)) + list(range(7, 12)):
            tun[5 * A_S:7 * A_S, c * A_S + A_OFF] = True
    if mode == "free":
        m[:] = True
    return m | tun, tun


def a_graph(m):
    from scipy.sparse import lil_matrix
    idx = -np.ones(m.shape, int)
    cells = np.argwhere(m)
    idx[m] = np.arange(len(cells))
    G = lil_matrix((len(cells), len(cells)))
    for k, (r, c) in enumerate(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < A_H and 0 <= cc < A_W and m[rr, cc]:
                    G[k, idx[rr, cc]] = np.hypot(dr, dc) / A_S       # in electrode pitches
    return G.tocsr(), idx


def a_elec_cell(e):
    return ROWS.index(e[0]) * A_S + A_OFF, (int(e[1:]) - 1) * A_S + A_OFF


def burst_latency_matrix(r):
    """Network bursts = 5 ms population bins above mean + 4 SD, merged if closer than 0.5 s; per burst
    and electrode, latency = first spike in [onset - 20 ms, onset + 150 ms] minus the earliest such
    spike (bursts with < 5 responding electrodes dropped). Returns (bursts, electrodes) in seconds."""
    trains = [np.sort(np.asarray(s, float)) for s in r.spike_times_s]
    allsp = np.concatenate(trains)
    rate = np.bincount((allsp / A_BW).astype(int), minlength=int(300 / A_BW) + 1).astype(float)
    thr = rate.mean() + 4 * rate.std()
    on = np.where((rate[1:] > thr) & (rate[:-1] <= thr))[0] + 1
    onsets = []
    for b in on * A_BW:
        if not onsets or b - onsets[-1] > 0.5:
            onsets.append(b)
    L = np.full((len(onsets), len(trains)), np.nan)
    for i, t0 in enumerate(onsets):
        for j, tr in enumerate(trains):
            k = np.searchsorted(tr, t0 - 0.020)
            if k < len(tr) and tr[k] <= t0 + 0.150:
                L[i, j] = tr[k]
        L[i] = L[i] - np.nanmin(L[i]) if np.isfinite(L[i]).sum() >= 5 else np.nan
    return L[np.isfinite(L).sum(1) >= 5]


def electrode_latency_ms(r):
    L = burst_latency_matrix(r)
    n = np.isfinite(L).sum(0)
    return np.where(n >= 10, np.nanmedian(L, 0) * 1000, np.nan)       # ms


def a_fit_dist(Dsrc, y, tr, extra=None):
    """Choose source (row of Dsrc) and linear coefficients minimising train abs error."""
    best = None
    for s in range(Dsrc.shape[0]):
        cols = [np.ones(len(y)), Dsrc[s]]
        if extra is not None:
            cols += list(extra(s))
        X = np.stack(cols, 1)
        coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        err = np.abs(X[tr] @ coef - y[tr]).mean()
        if best is None or err < best[0]:
            best = (err, s, coef, X)
    _, s, coef, X = best
    return X @ coef, s


def a_heldout(w):
    """Held-out electrode latency (port of the geodesic pilot). Returns (per_recording, summary, extra)."""
    from scipy.sparse.csgraph import dijkstra
    geo = {}
    for mode in ("chip", "free"):
        m, tun = a_mask(mode)
        G, idx = a_graph(m)
        geo[mode] = (G, idx, tun)
    out, allrows = [], []
    for fid in sorted(w[w.condition == "NoStim"].fid.unique()):
        r = w[(w.condition == "NoStim") & (w.fid == fid)].sort_values("electrode").reset_index(drop=True)
        y = electrode_latency_ms(r)
        ok = np.isfinite(y)
        els = list(r.electrode[ok]); comp = r.subregion[ok].values; y = y[ok]
        rc = np.array([a_elec_cell(e) for e in els])
        D = {}
        for mode in ("chip", "free"):
            G, idx, tun = geo[mode]
            nodes = idx[rc[:, 0], rc[:, 1]]
            D[mode] = dijkstra(G, indices=nodes)[:, nodes]
        Deuc = np.hypot(rc[:, None, 0] - rc[None, :, 0], rc[:, None, 1] - rc[None, :, 1]) / A_S
        # tunnel crossings between compartments: 0 same, 1 adjacent, 2 diagonal
        quad = {c: (rc[comp == c][:, 0].mean() > 6 * A_S, rc[comp == c][:, 1].mean() > 6 * A_S) for c in set(comp)}
        qa = np.array([quad[c] for c in comp], int)
        cross = np.abs(qa[:, None, 0] - qa[None, :, 0]) + np.abs(qa[:, None, 1] - qa[None, :, 1])
        onehot = np.stack([(comp == c).astype(float) for c in COMPARTMENTS], 1)[:, 1:]
        rng = np.random.default_rng(int(fid))
        folds = rng.permutation(len(y)) % 5
        pred = {k: np.full(len(y), np.nan) for k in ("comp_mean", "idw", "idw_comp", "free_eik", "chip_eik", "chip_eik_c")}
        for f in range(5):
            te, tr = folds == f, folds != f
            for j in np.where(te)[0]:
                same = tr & (comp == comp[j])
                pred["comp_mean"][j] = y[same].mean() if same.any() else y[tr].mean()
                wgt = 1 / Deuc[j, tr] ** 2
                pred["idw"][j] = (wgt * y[tr]).sum() / wgt.sum()
                if same.any():
                    ws = 1 / Deuc[j, same] ** 2
                    pred["idw_comp"][j] = (ws * y[same]).sum() / ws.sum()
                else:
                    pred["idw_comp"][j] = pred["idw"][j]
            pred["free_eik"][te] = a_fit_dist(Deuc, y, tr)[0][te]
            pred["chip_eik"][te] = a_fit_dist(D["chip"], y, tr, extra=lambda s: [cross[s]])[0][te]
            pred["chip_eik_c"][te] = a_fit_dist(D["chip"], y, tr, extra=lambda s: [cross[s]] + list(onehot.T))[0][te]
        mae = {k: float(np.abs(v - y).mean()) for k, v in pred.items()}
        out.append(dict(fid=int(fid), n_electrodes=int(len(y)), latency_sd_ms=float(y.std()), mae_ms=mae))
        for k, v in pred.items():
            for j in range(len(y)):
                allrows.append(dict(fid=int(fid), electrode=els[j], model=k, abs_err=float(abs(v[j] - y[j]))))
    df = pd.DataFrame(allrows)
    piv = df.pivot_table(index=["fid", "electrode"], columns="model", values="abs_err")
    summ = {}
    rng = np.random.default_rng(0)
    fids = piv.index.get_level_values(0).unique()

    def contrast(model, ref):
        d = (piv[model] - piv[ref])
        per = d.groupby(level=0).mean()          # bootstrap clustered by recording
        bs = [per.loc[rng.choice(fids, len(fids))].mean() for _ in range(4000)]
        return dict(mean=float(per.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    recordings_improved=int((per < 0).sum()), n=int(len(per)))

    for ref in ("idw_comp", "idw", "free_eik"):
        summ[f"chip_eik_c_minus_{ref}_ms"] = contrast("chip_eik_c", ref)
    summ["mean_mae_ms"] = {k: float(piv[k].groupby(level=0).mean().mean()) for k in piv.columns}
    # contrasts added in this appendix (same RNG stream continued, so the pilot's numbers are unchanged)
    extra = {"chip_eik_c_minus_comp_mean_ms": contrast("chip_eik_c", "comp_mean"),
             "chip_eik_minus_idw_comp_ms": contrast("chip_eik", "idw_comp"),
             "chip_eik_minus_free_eik_ms": contrast("chip_eik", "free_eik")}
    return out, summ, extra


def a_reliability(w):
    """Split-half reliability of the latency maps (port of the latency pilot)."""
    from scipy.stats import spearmanr
    res = []
    for fid in sorted(w[w.condition == "NoStim"].fid.unique()):
        r = w[(w.condition == "NoStim") & (w.fid == fid)].sort_values("electrode")
        L = burst_latency_matrix(r)
        odd, even = L[0::2], L[1::2]
        ok = (np.isfinite(odd).sum(0) >= 5) & (np.isfinite(even).sum(0) >= 5)
        rho = spearmanr(np.nanmedian(odd[:, ok], 0), np.nanmedian(even[:, ok], 0))[0] if ok.sum() >= 8 else float("nan")
        sub = r.subregion.values
        comp = {s: float(np.nanmedian(L[:, sub == s])) * 1000 for s in COMPARTMENTS}
        res.append(dict(fid=int(fid), n_bursts=int(len(L)), n_electrodes_reliable=int(ok.sum()),
                        split_half_spearman=float(rho), median_latency_ms_by_compartment=comp,
                        median_spread_ms=float(np.nanmedian(np.nanmax(L, 1)) * 1000) if len(L) else float("nan")))
    return res


def a_confound(w, ax):
    """Latency vs firing rate, and compartment lead order vs axon directionality (port of the extra pilot)."""
    from scipy.stats import spearmanr
    ax = ax[ax.condition == "NoStim"]
    conf, rows = [], []
    for fid in range(1, 10):
        r = w[(w.condition == "NoStim") & (w.fid == fid)].sort_values("electrode").reset_index(drop=True)
        y = electrode_latency_ms(r); ok = np.isfinite(y)
        conf.append(float(spearmanr(y[ok], np.log1p(r.n_spikes.values[ok]))[0]))
        lat = {s: float(np.nanmedian(y[ok & (r.subregion.values == s)])) for s in COMPARTMENTS}
        for x, z in [("EC", "DG"), ("DG", "CA3"), ("CA3", "CA1"), ("CA1", "EC")]:
            g = ax[(ax.fid == fid) & (ax.subregion_pair == f"{x}-{z}")]
            nff, nfb = int((g.direction == "ff").sum()), int((g.direction == "fb").sum())
            if nff + nfb:
                rows.append(dict(fid=fid, pair=f"{x}-{z}", lead_ff_ms=lat[z] - lat[x], d=(nff - nfb) / (nff + nfb)))
    df = pd.DataFrame(rows).dropna()
    rho = float(spearmanr(df.lead_ff_ms, df.d)[0])
    rng = np.random.default_rng(0)
    null = [spearmanr(df.lead_ff_ms, rng.permutation(df.d.values))[0] for _ in range(5000)]
    return {"status": "PILOT - exploratory",
            "latency_vs_log_spikecount_spearman_per_recording": conf,
            "median_confound_rho": float(np.median(conf)),
            "lead_vs_axon_d": {"n": int(len(df)), "spearman": rho, "perm_p": float(np.mean(np.abs(null) >= abs(rho)))}}


def run_part_a(pilot_dir: Path) -> dict:
    t0 = time.time()
    w, ax = load_brewer()
    per, summ, extra = a_heldout(w)
    rel = a_reliability(w)
    conf = a_confound(w, ax)
    # byte-level reconstruction of the three pilot files from the ported code
    recon = {
        "pilot_geodesic.json": json.dumps({"status": "PILOT - exploratory, not a submission result",
                                           "per_recording": per, "summary": summ}, indent=1),
        "pilot_latency.json": json.dumps({"status": "PILOT - feasibility, exploratory", "results": rel}, indent=1),
        "pilot_latency_extra.json": json.dumps(conf, indent=1),
    }
    check = {}
    for name, text in recon.items():
        p = pilot_dir / name
        check[name] = {"ported_file_sha256": hashlib.sha256(as_written(text)).hexdigest(), "pilot_file_sha256": sha256_file(p),
                       "byte_identical": same_as_file(text, p),
                       "max_rel_diff": (max_rel_diff(json.loads(text), json.loads(p.read_text(encoding="utf-8")))
                                        if p.exists() else None)}
    first = {}
    for r in rel:
        c = r["median_latency_ms_by_compartment"]
        first[str(r["fid"])] = min(c, key=c.get)
    rhos = [r["split_half_spearman"] for r in rel]
    return {
        "part": "A", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "full rerun of the ported code",
        "heldout": {"per_recording": per, "summary": summ, "summary_added": extra},
        "reliability": {"per_recording": rel, "split_half_spearman_median": float(np.median(rhos)),
                        "split_half_spearman_min": float(np.min(rhos)), "split_half_spearman_max": float(np.max(rhos)),
                        "earliest_compartment_by_recording": first,
                        "n_recordings_CA3_earliest": int(sum(v == "CA3" for v in first.values())),
                        "n_recordings": len(rel)},
        "confound": conf,
        "pilot_check": check,
        "elapsed_s": round(time.time() - t0, 1), "env": env_info(),
    }


# ================================================================== B: QESN-v2 chip-lattice reservoir
B_BIN, B_DUR = 0.025, 300.0
B_NB = int(B_DUR / B_BIN)
B_S = 2                      # lattice cells per electrode pitch
B_OFF = B_S // 2
B_HL = B_WL = 12 * B_S
B_HORIZONS = (1, 4)
B_ALPHAS = 10.0 ** np.arange(-2, 5)
B_SPLIT = (0.6, 0.7)
B_MODELS = ("lattice_chip", "lattice_full", "lattice_walls", "lattice_chip_shuffled", "esn_nonspatial")
B_REFS = ("var_ridge", "esn_nonspatial", "persistence")
B_PILOT_META = {"status": "PILOT - exploratory, not a submission result",
                "data": "Brewer NoStim (CC0), 25 ms bins, sqrt counts, 76 well electrodes",
                "split": "per recording, time-ordered: train 60 %, val 10 %, test 30 %"}


def b_elec_rc(e):
    return ROWS.index(e[0]), int(e[1:]) - 1


def b_bin_recording(df):
    df = df.sort_values("electrode")
    X = np.zeros((B_NB, len(df)), np.float32)
    for j, st in enumerate(df.spike_times_s):
        st = np.asarray(st, float)
        idx = np.minimum((st / B_BIN).astype(int), B_NB - 1)
        np.add.at(X[:, j], idx, 1)
    return np.sqrt(X), list(df.electrode), list(df.subregion)


def b_lattice_mask(mode):
    m = np.zeros((B_HL, B_WL), bool)
    for r0 in (0, 7):
        for c0 in (0, 7):
            m[r0 * B_S:(r0 + 5) * B_S, c0 * B_S:(c0 + 5) * B_S] = True
    if mode == "full":
        m[:] = True
    elif mode == "chip":   # 20 one-cell-wide tunnels, exactly where the device has them
        for r in list(range(5)) + list(range(7, 12)):
            m[r * B_S + B_OFF, 5 * B_S:7 * B_S] = True
        for c in list(range(5)) + list(range(7, 12)):
            m[5 * B_S:7 * B_S, c * B_S + B_OFF] = True
    elif mode == "walls":  # compartments isolated
        pass
    return m


def b_laplacian(mask):
    idx = -np.ones(mask.shape, int)
    cells = np.argwhere(mask)
    idx[mask] = np.arange(len(cells))
    n = len(cells)
    A = np.zeros((n, n), np.float64)
    for k, (r, c) in enumerate(cells):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < B_HL and 0 <= cc < B_WL and mask[rr, cc]:
                A[k, idx[rr, cc]] = 1.0
    L = np.diag(A.sum(1)) - A
    return L, idx


def b_cayley(L, c):
    """Crank-Nicolson step of i dpsi/dt = c L psi: (I + i c/2 L) psi' = (I - i c/2 L) psi.
    Exactly unitary for symmetric L (fixes the |psi|^2 non-conservation of the original update)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    n = len(L)
    Ls = sp.csc_matrix(L)
    I = sp.identity(n, format="csc")
    lu = splu((I + 0.5j * c * Ls).tocsc())
    M2 = (I - 0.5j * c * Ls).tocsr()
    return lambda v: lu.solve(M2 @ v)


def b_unitarity_check(c=2.0, seed=0) -> float:
    """|| U psi || / || psi || for a random complex state on the chip lattice (1.0 = unitary)."""
    L, _ = b_laplacian(b_lattice_mask("chip"))
    U = b_cayley(L, c)
    r = np.random.default_rng(seed)
    v = r.standard_normal(len(L)) + 1j * r.standard_normal(len(L))
    return float(np.linalg.norm(U(v)) / np.linalg.norm(v))


def b_run_lattice(X, electrodes, mode, c, gamma, g, a, nsub=2, shuffle=False):
    mask = b_lattice_mask(mode)
    L, idx = b_laplacian(mask)
    U = b_cayley(L, c)
    n = len(L)
    pos = [idx[r * B_S + B_OFF, q * B_S + B_OFF] for r, q in (b_elec_rc(e) for e in electrodes)]
    if shuffle:
        pos = list(np.random.default_rng(7).permutation(pos))
    B = np.zeros((n, X.shape[1]), np.complex64)
    for j, p in enumerate(pos):
        B[p, j] = a
    psi = np.zeros(n, np.complex128)
    F = np.zeros((len(X), 3 * n), np.float32)
    decay = np.float32(1.0 - gamma)
    for t in range(len(X)):
        psi = psi + B @ X[t]
        for _ in range(nsub):
            psi = decay * U(psi)
            psi = psi * np.exp(1j * g * (psi.real ** 2 + psi.imag ** 2))
        F[t, :n] = psi.real
        F[t, n:2 * n] = psi.imag
        F[t, 2 * n:] = psi.real ** 2 + psi.imag ** 2
    return F


def b_run_esn(X, N, rho, leak, a, seed=3):
    r = np.random.default_rng(seed)
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs
    W = sp.random(N, N, density=0.02, random_state=seed, data_rvs=r.standard_normal, format="csr")
    ev = np.abs(eigs(W, k=1, which="LM", return_eigenvectors=False)).max()
    W = (W * (rho / ev)).astype(np.float32)
    Win = (r.uniform(-1, 1, (N, X.shape[1])) * a).astype(np.float32)
    x = np.zeros(N, np.float32)
    F = np.zeros((len(X), N), np.float32)
    for t in range(len(X)):
        x = (1 - leak) * x + leak * np.tanh(W @ x + Win @ X[t])
        F[t] = x
    return F


def b_lags(X, L):
    T = len(X)
    out = np.zeros((T, X.shape[1] * L), np.float32)
    for l in range(L):
        out[l:, l * X.shape[1]:(l + 1) * X.shape[1]] = X[: T - l]
    return out


def b_ridge_eval(F, X, h, split):
    """features F[t] (known at t) predict X[t+h]. Returns val-best alpha, val MSE, test MSE per electrode."""
    tr, va = split
    Fx = np.hstack([F, X, np.ones((len(X), 1), np.float32)])[: len(X) - h]
    Y = X[h:]
    T = len(Y)
    i1, i2 = int(tr * T), int(va * T)
    mu = Fx[:i1].mean(0); sd = Fx[:i1].std(0) + 1e-6
    mu[-1], sd[-1] = 0.0, 1.0
    Z = ((Fx - mu) / sd).astype(np.float64)
    G = Z[:i1].T @ Z[:i1]; b = Z[:i1].T @ Y[:i1]
    ev, V = np.linalg.eigh(G)
    Vb = V.T @ b
    best, bestm = None, np.inf
    for al in B_ALPHAS:
        Wt = V @ (Vb / (ev + al)[:, None])
        m = ((Z[i1:i2] @ Wt - Y[i1:i2]) ** 2).mean()
        if m < bestm:
            best, bestm = al, m
    G2 = Z[:i2].T @ Z[:i2]; b2 = Z[:i2].T @ Y[:i2]
    Wt = np.linalg.solve(G2 + best * np.eye(len(G2)), b2)
    err = ((Z[i2:] @ Wt - Y[i2:]) ** 2).mean(0)
    return best, bestm, err


def b_persistence(X, h, split):
    Y = X[h:]; P = X[:-h]
    i2 = int(split[1] * len(Y))
    return ((P[i2:] - Y[i2:]) ** 2).mean(0)


def b_select_over(builder, grid, X, split):
    """Build features once per config; per horizon choose the config with best val MSE."""
    best = {h: None for h in B_HORIZONS}
    for cfg in grid:
        F = builder(**cfg)
        for h in B_HORIZONS:
            al, vm, err = b_ridge_eval(F, X, h, split)
            if best[h] is None or vm < best[h][0]:
                best[h] = (vm, cfg, err)
    return best


def b_run_recording(w, fid, log=print) -> dict:
    split = B_SPLIT
    lat_grid = [dict(c=c, gamma=gm, g=0.5, a=0.5) for c in (0.5, 2.0) for gm in (0.05, 0.2)]
    esn_grid = [dict(rho=r, leak=lk, a=0.5) for r in (0.8, 0.95) for lk in (0.3, 1.0)]
    ncell = int(b_lattice_mask("chip").sum())
    t0 = time.time()
    X, els, subs = b_bin_recording(w[(w.condition == "NoStim") & (w.fid == fid)])
    row = {"fid": int(fid), "n_bins": int(len(X)), "n_lattice_cells": ncell,
           "test_var": float(X[int(0.7 * len(X)):].var(0).mean())}
    res = {h: {"persistence": float(b_persistence(X, h, split).mean())} for h in B_HORIZONS}
    var = b_select_over(lambda Lg: b_lags(X, Lg)[:, X.shape[1]:], [dict(Lg=l) for l in (1, 2, 4, 8)], X, split)
    esn = b_select_over(lambda **k: b_run_esn(X, 3 * ncell, **k), esn_grid, X, split)
    lat = b_select_over(lambda **k: b_run_lattice(X, els, "chip", **k), lat_grid, X, split)
    for h in B_HORIZONS:
        res[h]["var_ridge"] = float(var[h][2].mean()); res[h]["var_L"] = var[h][1]["Lg"]
        res[h]["esn_nonspatial"] = float(esn[h][2].mean()); res[h]["esn_cfg"] = esn[h][1]
        res[h]["lattice_chip"] = float(lat[h][2].mean()); res[h]["lattice_cfg"] = lat[h][1]
    # geometry ablations with the chip-selected config (h=1 config reused for both horizons)
    cfg = lat[1][1]
    for mode, sh in (("full", False), ("walls", False), ("chip", True)):
        F = b_run_lattice(X, els, mode, shuffle=sh, **cfg)
        for h in B_HORIZONS:
            res[h][f"lattice_{mode}{'_shuffled' if sh else ''}"] = float(b_ridge_eval(F, X, h, split)[2].mean())
    for h in B_HORIZONS:
        row[f"h{h}"] = res[h]
    row["elapsed_s"] = round(time.time() - t0, 1)
    log(f"B fid {fid}: " + json.dumps({f"h{h}": {k: v for k, v in res[h].items() if isinstance(v, float)} for h in B_HORIZONS}))
    return row


def _rel_summary(a, b, rng, n=4000):
    rel = 100 * (np.array(a) - np.array(b)) / np.array(b)
    bs = [rel[rng.integers(0, len(rel), len(rel))].mean() for _ in range(n)]
    return dict(mean_rel_pct=round(float(rel.mean()), 2), median=round(float(np.median(rel)), 2),
                ci95=[round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
                n_better=int((rel < 0).sum()), n=len(rel))


def b_summary(results) -> dict:
    """Relative test-MSE change (%) per recording, mean over recordings, 95 % CI by bootstrap over
    recordings (one RNG, seed 0, 4000 resamples, fixed order). n_better = recordings where the first
    model has lower MSE. Identical to the pilot's summary."""
    rng = np.random.default_rng(0)
    out = {}
    for h in B_HORIZONS:
        for ref in B_REFS:
            for m in B_MODELS:
                if m == ref:
                    continue
                out[f"h{h}|{m}_vs_{ref}"] = _rel_summary([r[f"h{h}"][m] for r in results],
                                                         [r[f"h{h}"][ref] for r in results], rng)
    return out


def b_ablation_summary(results) -> dict:
    """Chip geometry vs geometry ablations (added in this appendix; separate RNG seed 1)."""
    rng = np.random.default_rng(1)
    out = {}
    for h in B_HORIZONS:
        chip = np.array([r[f"h{h}"]["lattice_chip"] for r in results])
        for ab in ("lattice_full", "lattice_walls", "lattice_chip_shuffled"):
            x = np.array([r[f"h{h}"][ab] for r in results])
            s = _rel_summary(chip, x, rng)
            rel_ab = 100 * (x - chip) / chip
            s["n_within_1pct"] = int((np.abs(rel_ab) < 1).sum())
            out[f"h{h}|lattice_chip_vs_{ab}"] = s
    out["mean_test_mse"] = {f"h{h}": {m: float(np.mean([r[f"h{h}"][m] for r in results]))
                                      for m in ("persistence", "var_ridge", "esn_nonspatial") + B_MODELS[:4]}
                            for h in B_HORIZONS}
    return out


def _strip_elapsed(row):
    return {k: v for k, v in row.items() if k != "elapsed_s"}


def run_part_b(pilot_dir: Path, mode: str, check_recs: list[int]) -> dict:
    t0 = time.time()
    w, _ = load_brewer()
    unit = b_unitarity_check()
    pilot_p = pilot_dir / "pilot_chiplattice.json"
    pilot_sum_p = pilot_dir / "pilot_chiplattice_summary.json"
    part = {"part": "B", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "crank_nicolson_norm_ratio": unit, "env": env_info()}
    if mode == "full":
        recs = sorted(int(x) for x in w[w.condition == "NoStim"].fid.unique())
        results = []
        for fid in recs:
            results.append(b_run_recording(w, fid))
            write_part("B_progress", {"results": results})
        part["mode"] = "full rerun of the ported code (all 9 NoStim recordings)"
        part["per_recording"] = results
        if pilot_p.exists():
            pil = json.loads(pilot_p.read_text(encoding="utf-8"))["results"]
            same = {str(r["fid"]): json.dumps(_strip_elapsed(r), indent=1) == json.dumps(_strip_elapsed(p), indent=1)
                    for r, p in zip(results, pil)}
            part["pilot_check"] = {"per_recording_identical_excluding_elapsed_s": same,
                                   "max_rel_diff": max_rel_diff(json.loads(json.dumps([_strip_elapsed(r) for r in results])),
                                                                [_strip_elapsed(p) for p in pil])}
    else:
        if not pilot_p.exists():
            raise SystemExit(f"--b-mode reuse needs {pilot_p}; run with --b-mode full instead")
        pil_text = pilot_p.read_text(encoding="utf-8")
        pil = json.loads(pil_text)
        check = []
        for fid in check_recs:
            row = b_run_recording(w, fid)
            ref = next(p for p in pil["results"] if p["fid"] == fid)
            a_txt, b_txt = json.dumps(_strip_elapsed(row), indent=1), json.dumps(_strip_elapsed(ref), indent=1)
            check.append({"fid": int(fid), "byte_identical_excluding_elapsed_s": a_txt == b_txt,
                          "ported_sha256": sha256_text(a_txt), "pilot_sha256": sha256_text(b_txt),
                          "max_rel_diff": max_rel_diff(json.loads(a_txt), json.loads(b_txt)),
                          "elapsed_s": row["elapsed_s"]})
        part["mode"] = ("reuse: per-recording numbers taken from the pilot run after the reduced check below "
                        f"(ported code rerun on recordings {check_recs} in a fresh process, i.e. the pilot's first "
                        "recordings in the pilot's order, so that process-level random state such as the ARPACK start "
                        "vectors of the ESN spectral-radius estimate is the same)")
        part["per_recording"] = pil["results"]
        part["reduced_check"] = check
        part["pilot_file"] = {"path": "colab/research/fran/_pilot/pilot_chiplattice.json (private working notes)",
                              "sha256": sha256_file(pilot_p)}
    summ = b_summary(part["per_recording"])
    part["summary"] = summ
    s_txt = json.dumps(summ, indent=1)
    part["summary_check"] = {"ported_file_sha256": hashlib.sha256(as_written(s_txt)).hexdigest(),
                             "pilot_file_sha256": sha256_file(pilot_sum_p),
                             "byte_identical": same_as_file(s_txt, pilot_sum_p)}
    part["ablations"] = b_ablation_summary(part["per_recording"])
    part["elapsed_s"] = round(time.time() - t0, 1)
    return part


# ================================================================== C: DoseLattice inside NeuroTrajectory
C_KS = (1, 2, 3, 4)
C_FOLD = 0


def _c_imports():
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    return torch, nn


def make_doselattice_class():
    torch, nn = _c_imports()
    from neurotwin.models.cnp import D_OUT, NeuroTrajectoryCNP, _mlp

    class DoseLatticeCNP(NeuroTrajectoryCNP):
        """NeuroTrajectory + a fixed 1-D diffusion lattice over log10 concentration. G=64 sites on
        [-3.5, 2.5]; each context well splats (y*m, m) linearly onto its two nearest sites; the QESN
        3-point mass-conserving stencil psi' = (1-2k)psi + k(psi_l + psi_r) (k=0.25, reflecting ends)
        runs 32 steps; snapshots after 2, 8, 32 steps give three bandwidths. At a query dose the
        snapshots are read by linear interpolation: sum(y m)/sum(m) (masked Nadaraya-Watson) and
        log(1 + density) per output dimension -> Linear(408, 64) -> extra decoder input."""

        def __init__(self, G=64, lo=-3.5, hi=2.5, snaps=(2, 8, 32), k=0.25, **kw):
            super().__init__(**kw)
            self.G, self.lo, self.hi, self.snaps, self.kd = G, lo, hi, snaps, k
            self.lp = nn.Linear(len(snaps) * 2 * D_OUT, 64)
            d = 128
            self.dec = _mlp(2 * d + 64 + (64 if self.use_interp else 0) + 64, 256,
                            (3 if self.use_interp else 2) * D_OUT, n=3)

        def lattice(self, cx, cy, cm, cmask, qx):
            B, N, _ = cx.shape
            G = self.G
            pos = ((cx[..., 0] - self.lo) / (self.hi - self.lo) * (G - 1)).clamp(0, G - 1)
            i0 = pos.floor().long().clamp(0, G - 2)
            w1 = (pos - i0.float()).unsqueeze(-1)
            w = cmask.float().unsqueeze(-1)
            vy, vm = cy * cm * w, cm * w
            field = torch.zeros(B, G, 2 * D_OUT, device=cx.device)
            src = torch.cat([vy, vm], -1)
            field.scatter_add_(1, i0.unsqueeze(-1).expand(-1, -1, 2 * D_OUT), src * (1 - w1))
            field.scatter_add_(1, (i0 + 1).unsqueeze(-1).expand(-1, -1, 2 * D_OUT), src * w1)
            psi = field.transpose(1, 2)                       # (B, C, G)
            qpos = ((qx[..., 0] - self.lo) / (self.hi - self.lo) * (G - 1)).clamp(0, G - 1)
            q0 = qpos.floor().long().clamp(0, G - 2)
            qw = (qpos - q0.float()).unsqueeze(1)
            feats, step = [], 0
            for s in self.snaps:
                while step < s:
                    pad = torch.cat([psi[..., :1], psi, psi[..., -1:]], -1)     # reflecting ends
                    psi = (1 - 2 * self.kd) * psi + self.kd * (pad[..., :-2] + pad[..., 2:])
                    step += 1
                a = torch.gather(psi, 2, q0.unsqueeze(1).expand(-1, psi.shape[1], -1))
                b = torch.gather(psi, 2, (q0 + 1).unsqueeze(1).expand(-1, psi.shape[1], -1))
                v = ((1 - qw) * a + qw * b).transpose(1, 2)            # (B, Q, 2*D_OUT)
                num, den = v[..., :D_OUT], v[..., D_OUT:]
                feats += [num / (den + 1e-3), torch.log1p(den)]
            return self.lp(torch.cat(feats, -1))

        def forward(self, cx, cy, cm, cmask, qx, qi=None):
            B = qx.shape[0]
            ex = self.emb(cx)
            h = self.enc(torch.cat([ex, cy * cm, cm], -1))
            w = cmask.float().unsqueeze(-1)
            n = w.sum(1)
            has = (n > 0).float()
            r_g = (h * w).sum(1) / n.clamp(min=1)
            r_g = has * r_g + (1 - has) * self.empty_g
            eq = self.emb(qx)
            if cmask.any() and self.use_attention:
                kpm = ~cmask
                kpm_safe = kpm.clone()
                kpm_safe[kpm.all(1)] = False
                r_l, _ = self.att(self.q(eq), self.k(ex), h, key_padding_mask=kpm_safe)
                r_l = has.unsqueeze(1) * r_l + (1 - has.unsqueeze(1)) * self.empty_l
            else:
                r_l = self.empty_l.expand(B, qx.shape[1], -1)
            parts = [r_g.unsqueeze(1).expand(-1, qx.shape[1], -1), r_l, eq]
            if self.use_interp:
                parts.append(self.ip(qi))
            parts.append(self.lattice(cx, cy, cm, cmask, qx))
            out = self.dec(torch.cat(parts, -1))
            if self.use_interp:
                delta, s, gate = out.chunk(3, -1)
                mu = torch.sigmoid(gate) * qi[..., :D_OUT] + delta
            else:
                mu, s = out.chunk(2, -1)
            return mu, 0.02 + nn.functional.softplus(s)

    return DoseLatticeCNP


def train_doselattice(cls, train_tasks, cfg, device):
    """Same loop as neurotwin.models.cnp.train_cnp, with the DoseLattice model class."""
    torch, nn = _c_imports()
    from neurotwin.models.cnp import LOSSES, make_batch, sample_episode
    torch.set_num_threads(int(os.environ.get("NT_THREADS", "3")))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = cls(use_interp=cfg.use_interp, n_freq=cfg.n_freq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    model.train()
    for step in range(cfg.steps):
        idx = rng.integers(0, len(train_tasks), cfg.batch)
        bt = [train_tasks[i] for i in idx]
        eps = [sample_episode(t, rng, cfg.max_ctx_levels) for t in bt]
        cx, cy, cm, cmask, qx, qy, qm, qmask, qi = make_batch(bt, [e[0] for e in eps], [e[1] for e in eps], device)
        mu, sig = model(cx, cy, cm, cmask, qx, qi)
        loss = LOSSES[cfg.likelihood](mu, sig, qy, qm, qmask)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    model.eval()
    return model


def c_port_fidelity(cls, tasks, pilot_dir: Path) -> dict:
    """Compare the ported DoseLatticeCNP with the pilot class: same parameter names/shapes, same
    initialisation under the same seed and identical forward outputs on a real batch (CPU)."""
    torch, _ = _c_imports()
    p = pilot_dir / "pilot_doselattice.py"
    if not p.exists():
        return {"available": False}
    import importlib.util
    spec = importlib.util.spec_from_file_location("pilot_doselattice", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from neurotwin.models.cnp import make_batch, sample_episode
    torch.manual_seed(0); a = cls(use_interp=True, n_freq=8)
    torch.manual_seed(0); b = mod.DoseLatticeCNP(use_interp=True, n_freq=8)
    sa, sb = a.state_dict(), b.state_dict()
    same_params = list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    a.eval(); b.eval()
    rng = np.random.default_rng(0)
    bt = tasks[:8]
    eps = [sample_episode(t, rng, 5) for t in bt]
    batch = make_batch(bt, [e[0] for e in eps], [e[1] for e in eps], "cpu")
    cx, cy, cm, cmask, qx, _, _, _, qi = batch
    with torch.no_grad():
        ma, sa_ = a(cx, cy, cm, cmask, qx, qi)
        mb, sb_ = b(cx, cy, cm, cmask, qx, qi)
    return {"available": True, "pilot_sha256": sha256_file(p), "identical_parameters_after_seeded_init": bool(same_params),
            "identical_forward_outputs": bool(torch.equal(ma, mb) and torch.equal(sa_, sb_)),
            "n_parameters": int(sum(v.numel() for v in a.parameters()))}


def _paired(a, b):
    from run_trajectory_cv import paired_boot
    pb = paired_boot(a, b)
    base = float(np.nanmean(np.asarray(b, float)[np.isfinite(a) & np.isfinite(b)]))
    pb["ci95_rel_pct"] = [round(100 * pb["ci95"][0] / base, 2), round(100 * pb["ci95"][1] / base, 2)]
    d = np.asarray(a, float) - np.asarray(b, float)
    pb["n_chem_improved"] = int((d[np.isfinite(d)] < 0).sum())
    return pb


def run_part_c(pilot_dir: Path, seeds: list[int], steps: int | None = None) -> dict:
    torch, _ = _c_imports()
    from neurotwin.models.cnp import TrainConfig, load_fold_models, predict, train_cnp
    from neurotwin.models.trajectory import AnalogKNN, split_context
    from run_trajectory_cv import BASELINES, curve_and_well_mae, designs
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tasks, _ = pickle.load(open(TASKS, "rb"))
    train = [t for t in tasks if t.fold != C_FOLD]
    test = [t for t in tasks if t.fold == C_FOLD]
    cls = make_doselattice_class()
    fidelity = c_port_fidelity(cls, tasks, pilot_dir)
    saved, cfg = load_fold_models(C_FOLD, device=dev, directory=ROOT / "data/processed/models")
    if not saved:
        raise SystemExit("v1 fold-0 models not found in data/processed/models")
    tc = dict(steps=steps or cfg["steps"], likelihood=cfg["likelihood"], use_interp=cfg["use_interp"], n_freq=cfg.get("n_freq", 8))
    timing = {}
    arms = {"neurotrajectory_v1_saved": saved}
    t1 = time.time()
    arms["neurotrajectory_retrained"] = [train_cnp(train, TrainConfig(seed=s, **tc), device=dev) for s in seeds]
    timing["neurotrajectory_retrained_train_s"] = round(time.time() - t1, 1)
    print(f"C: retrained NeuroTrajectory ({timing['neurotrajectory_retrained_train_s']} s)", flush=True)
    t1 = time.time()
    arms["doselattice"] = [train_doselattice(cls, train, TrainConfig(seed=s, **tc), dev) for s in seeds]
    timing["doselattice_train_s"] = round(time.time() - t1, 1)
    print(f"C: DoseLattice ({timing['doselattice_train_s']} s)", flush=True)
    knn = AnalogKNN(train)
    base_fns = dict(BASELINES)
    base_fns["analog_knn"] = knn.predict
    rows = []
    for t in test:
        for k in C_KS:
            if k >= len(t.levels):
                continue
            acc = {}
            for ctx_lv in designs(t, k):
                ic, it = split_context(t, ctx_lv)
                lv = np.unique(t.logc[it])
                for name, fn in base_fns.items():
                    acc.setdefault(name, []).append(curve_and_well_mae(fn(t, ic, lv), t, it)[0])
                for name, models in arms.items():
                    mu = np.mean([predict(m, t, ic, lv, device=dev)[0] for m in models], 0)
                    acc.setdefault(name, []).append(curve_and_well_mae(mu, t, it)[0])
            for name, v in acc.items():
                rows.append({"chemical": t.chem, "k": k, "method": name, "curve_mae": float(np.nanmean(v))})
    df = pd.DataFrame(rows)
    # integrity: the saved v1 ensemble and the baselines reproduce the v1 CV per-chemical log for fold 0
    integrity = {}
    csv = RESULTS / "trajectory_cv_per_chemical.csv"
    if csv.exists():
        ref = pd.read_csv(csv)
        ref = ref[(ref.fold == C_FOLD) & ref.k.isin(C_KS)]
        for ours, theirs in [("neurotrajectory_v1_saved", "neurotrajectory")] + [(b, b) for b in base_fns]:
            a = df[df.method == ours].set_index(["chemical", "k"]).curve_mae
            b = ref[ref.method == theirs].set_index(["chemical", "k"]).curve_mae
            j = a.index.intersection(b.index)
            integrity[ours] = {"n_matched": int(len(j)), "max_abs_diff_vs_trajectory_cv_csv": float((a[j] - b[j]).abs().max())}
    by_k = {}
    base_names = list(base_fns)
    for k in C_KS:
        p = df[df.k == k].pivot_table(index="chemical", columns="method", values="curve_mae")
        best = min(base_names, key=lambda c: p[c].mean())
        by_k[str(k)] = {
            "n_chemicals": int(len(p)),
            "mean_curve_mae": {c: round(float(p[c].mean()), 4) for c in p.columns},
            "best_baseline": best,
            "doselattice_vs_neurotrajectory_v1_saved": _paired(p["doselattice"], p["neurotrajectory_v1_saved"]),
            "doselattice_vs_neurotrajectory_retrained": _paired(p["doselattice"], p["neurotrajectory_retrained"]),
            "neurotrajectory_retrained_vs_v1_saved": _paired(p["neurotrajectory_retrained"], p["neurotrajectory_v1_saved"]),
            "doselattice_vs_best_baseline": _paired(p["doselattice"], p[best]),
            "neurotrajectory_v1_saved_vs_best_baseline": _paired(p["neurotrajectory_v1_saved"], p[best]),
        }
        print(f"C k={k}: " + json.dumps(by_k[str(k)]["doselattice_vs_neurotrajectory_v1_saved"]), flush=True)
    pilot = {}
    pj = pilot_dir / "pilot_doselattice.json"
    if pj.exists():
        pilot = json.loads(pj.read_text(encoding="utf-8"))
        pilot = {"note": ("exploratory pilot, superseded by this run: v2 fold-0 configuration (n_freq 3), 2 seeds, "
                          "both arms retrained; bootstrap CI on the absolute difference"),
                 "sha256": sha256_file(pj), "seeds": pilot.get("seeds"), "config": pilot.get("config"),
                 "by_k": pilot.get("by_k")}
    return {"part": "C", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fold": C_FOLD, "n_train_chemicals": len(train), "n_test_chemicals": len(test),
            "reference_models": "v1 cross-validated fold-0 models, data/processed/models (primary)",
            "config": {**tc, "seeds": seeds, "designs_per_k": 5, "ks": list(C_KS)},
            "device": dev, "timing": timing, "port_fidelity": fidelity, "integrity": integrity,
            "by_k": by_k, "per_chemical": rows, "pilot_reference": pilot,
            "elapsed_s": round(time.time() - t0, 1), "env": env_info()}


# ================================================================== verdicts and final JSON
def verdict_a(s) -> str:
    c = s["chip_eik_c_minus_idw_comp_ms"]
    if c["ci95"][1] < 0:
        return "better"
    return "worse" if c["ci95"][0] > 0 else "no gain over interpolation"


def verdict_b(summ) -> str:
    keys = [f"h{h}|lattice_chip_vs_{ref}" for h in B_HORIZONS for ref in ("var_ridge", "esn_nonspatial")]
    cis = [summ[k]["ci95"] for k in keys]
    if all(c[1] < 0 for c in cis):
        return "better"
    if any(c[0] > 0 for c in cis):
        return "worse"
    return "matches but does not beat"


def verdict_c(by_k) -> str:
    """Per k, the paired CI of DoseLattice minus the v1 NeuroTrajectory: 'worse' if it is above 0 at
    every k; 'no gain; worse at some k' if above 0 at some k and below 0 at none; 'matches but does not
    beat' if it includes 0 at every k; 'better' if below 0 at every k; 'mixed' otherwise."""
    cis = [by_k[str(k)]["doselattice_vs_neurotrajectory_v1_saved"]["ci95"] for k in C_KS]
    n_worse, n_better = sum(c[0] > 0 for c in cis), sum(c[1] < 0 for c in cis)
    if n_worse == len(cis):
        return "worse"
    if n_better == len(cis):
        return "better"
    if n_worse and not n_better:
        return "no gain; worse at some k"
    return "matches but does not beat" if not (n_worse or n_better) else "mixed"


def _ks_where(by_k, key, worse=True):
    return [k for k in C_KS if (by_k[str(k)][key]["ci95"][0] > 0 if worse else by_k[str(k)][key]["ci95"][1] < 0)]


def build_final(A: dict, B: dict, C: dict, env_checks: dict | None = None) -> dict:
    env_checks = env_checks or {}
    hs, hx = A["heldout"]["summary"], A["heldout"]["summary_added"]
    prim = hs["chip_eik_c_minus_idw_comp_ms"]
    free = hs["chip_eik_c_minus_free_eik_ms"]
    a_success = bool(prim["ci95"][1] < 0 and free["mean"] < 0)
    va, vb, vc = verdict_a(hs), verdict_b(B["summary"]), verdict_c(C["by_k"])
    bs, ab = B["summary"], B["ablations"]
    # ---- plain-words details, conditional on the numbers
    def beats(c):
        return c["ci95"][1] < 0

    lead = A["confound"]["lead_vs_axon_d"]
    a_detail = ("Latency maps are reproducible (median split-half Spearman above 0.9) but "
                if A["reliability"]["split_half_spearman_median"] > 0.9
                else "Latency maps are only moderately reproducible, and ")
    a_detail += ("track electrode firing rate; " if A["confound"]["median_confound_rho"] < -0.5
                 else "are not dominated by firing rate; ")
    if not any(beats(c) for c in (prim, hs["chip_eik_c_minus_idw_ms"], hx["chip_eik_c_minus_comp_mean_ms"])):
        a_detail += "the chip-geodesic model does not beat compartment means or inverse-distance interpolation on held-out electrodes"
    else:
        a_detail += "the chip-geodesic model beats at least one interpolation baseline on held-out electrodes"
    a_detail += ("; the compartment lead order does not follow axon directionality." if lead["perm_p"] > 0.05
                 else "; the compartment lead order follows axon directionality.")
    ab_keys = [f"h{h}|lattice_chip_vs_{x}" for h in B_HORIZONS for x in ("lattice_full", "lattice_walls", "lattice_chip_shuffled")]
    geometry_matters = any(ab[k]["ci95"][1] < 0 for k in ab_keys)
    if beats(bs["h1|lattice_chip_vs_persistence"]):
        b_detail = ("Beats persistence (as the ridge VAR and the ESN also do), but "
                    if beats(bs["h1|esn_nonspatial_vs_persistence"]) else "Beats persistence, but ")
    else:
        b_detail = "Does not beat persistence and "
    b_detail += {"matches but does not beat": "ties the ridge VAR and the matched non-spatial reservoir at both horizons",
                 "worse": "is worse than the ridge VAR or the matched non-spatial reservoir",
                 "better": "beats the ridge VAR and the matched non-spatial reservoir"}[vb]
    plain = {"lattice_full": "no walls", "lattice_walls": "isolated compartments", "lattice_chip_shuffled": "shuffled electrode positions"}
    chip_worse = {h: [plain[x] for x in plain if ab[f"h{h}|lattice_chip_vs_{x}"]["ci95"][0] > 0] for h in B_HORIZONS}
    if not geometry_matters:
        b_detail += "; removing walls, isolating compartments or shuffling electrode positions does not make it worse"
        worse_txt = ["at h=" + str(h) + " than " + " and ".join(v) for h, v in chip_worse.items() if v]
        if worse_txt:
            b_detail += " (the true chip geometry is even slightly worse " + "; ".join(worse_txt) + ")"
        b_detail += ", so the chip geometry adds nothing measurable."
    else:
        b_detail += "; at least one geometry ablation is worse than the chip geometry."
    key = "doselattice_vs_neurotrajectory_v1_saved"
    kw, kb = _ks_where(C["by_k"], key), _ks_where(C["by_k"], key, worse=False)
    all_k = ", ".join(map(str, C_KS))
    pos = all(C["by_k"][str(k)][key]["mean_diff"] > 0 for k in C_KS)
    c_detail = {"worse": ("Adding the diffusion encoder makes NeuroTrajectory's forecasts less accurate at every k ("
                          + all_k + "); likely redundant, because the model already interpolates in log-dose and "
                          "attends locally in dose."),
                "no gain; worse at some k": ("Adding the diffusion encoder never improves NeuroTrajectory"
                                             + (" (the point estimate is worse at every k)" if pos else "")
                                             + "; the CI excludes 0 only at k = " + ", ".join(map(str, kw))
                                             + ". Likely redundant, because the model already interpolates in "
                                             "log-dose and attends locally in dose."),
                "better": "The diffusion encoder improves the forecasts at k = " + ", ".join(map(str, kb)) + " (development fold only).",
                "matches but does not beat": "The diffusion encoder neither helps nor hurts at the tested numbers of measured concentrations.",
                "mixed": "The diffusion encoder helps at some k and hurts at others."}[vc]
    pc = pd.DataFrame(C["per_chemical"]).pivot_table(index=["chemical", "k"], columns="method", values="curve_mae")
    retrain_diff = float((pc["neurotrajectory_retrained"] - pc["neurotrajectory_v1_saved"]).abs().max())
    b_full_min = round(sum(r.get("elapsed_s", 0.0) for r in B["per_recording"]) / 60, 1)
    if B["mode"].startswith("reuse"):
        rc = B.get("reduced_check") or []
        ok = all(c["byte_identical_excluding_elapsed_s"] for c in rc)
        runtime_note = ("B reuses the pilot's per-recording numbers (a full rerun takes about b_full_rerun_cpu_min_estimate "
                        "minutes of CPU; use --b-mode full) after the ported code "
                        + ("reproduced" if ok else "did NOT reproduce") + " recordings "
                        + ", ".join(str(c["fid"]) for c in rc) + " byte-identically; A and C are complete reruns.")
    else:
        runtime_note = "A, B and C are complete reruns of the ported code."
    pr = (C.get("pilot_reference") or {}).get("by_k") or {}
    pilot_vs_run = {
        "note": ("The exploratory pilot (v2 fold-0 configuration, n_freq 3, 2 seeds) found DoseLattice worse with the "
                 "CI above 0 at the k listed below; with the primary v1 configuration (this run) the effect is smaller."),
        "pilot_k_ci_above_0": [int(k) for k in sorted(pr, key=int) if pr[k]["ci95"][0] > 0],
        "pilot_rel_change_pct": {k: round(pr[k]["rel_change_pct"], 2) for k in sorted(pr, key=int)},
        "this_run_k_ci_above_0": kw,
        "this_run_rel_change_pct": {str(k): C["by_k"][str(k)][key]["rel_change_pct"] for k in C_KS},
    } if pr else None
    arch = {
        "A_eikonal_latency": {
            "architecture": "Eikonal (geodesic arrival-time) propagation model on the chip geometry",
            "source": SOURCES["A"],
            "what_was_tested": ("Whether a chip-geodesic arrival-time model (homogeneous speed inside the device mask, "
                                "walls and the 20 tunnels exactly as on the chip, one source cell, a tunnel-crossing "
                                "term and per-compartment offsets; model 'chip_eik_c') predicts the median network-burst "
                                "latency of held-out electrodes better than spatial interpolation."),
            "data": ("Brewer 4-compartment hippocampal MEA (CC0): 9 NoStim recordings, 76 well electrodes, 300 s. "
                     "Per-electrode median burst latency: 5 ms population bins, bursts above mean + 4 SD merged "
                     "within 0.5 s, first spike in [onset - 20 ms, onset + 150 ms]; electrodes with >= 10 bursts."),
            "split": "5-fold leave-electrodes-out inside each recording (seed = recording id); every model is fitted on the training electrodes only.",
            "baselines": ["comp_mean: mean latency of the training electrodes of the same compartment",
                          "idw: inverse-distance weighting (power 2) over all training electrodes",
                          "idw_comp: inverse-distance weighting within the same compartment",
                          "free_eik: eikonal model in free space, no walls (geometry ablation)"],
            "metric": ("MAE (ms) of held-out electrode latency; contrast = chip_eik_c minus baseline, averaged per "
                       "recording; 95 % CI by bootstrap over recordings (4000 resamples)."),
            "result": {
                "primary_contrast": "chip_eik_c_minus_idw_comp_ms",
                "mean_mae_ms": hs["mean_mae_ms"],
                "chip_eik_c_minus_idw_comp_ms": prim,
                "chip_eik_c_minus_idw_ms": hs["chip_eik_c_minus_idw_ms"],
                "chip_eik_c_minus_free_eik_ms": free,
                "chip_eik_c_minus_comp_mean_ms": hx["chip_eik_c_minus_comp_mean_ms"],
                "chip_eik_minus_idw_comp_ms": hx["chip_eik_minus_idw_comp_ms"],
                "preregistered_success": {
                    "criterion": ("fit.md section 4.A: chip eikonal MAE below idw_comp with a recording-clustered "
                                  "bootstrap CI excluding 0, and below the free-space eikonal"),
                    "met": a_success},
                "precondition_latency_maps": {
                    "split_half_spearman_median": A["reliability"]["split_half_spearman_median"],
                    "split_half_spearman_min": A["reliability"]["split_half_spearman_min"],
                    "n_recordings_CA3_earliest": A["reliability"]["n_recordings_CA3_earliest"],
                    "n_recordings": A["reliability"]["n_recordings"],
                    "latency_vs_log_spike_count_spearman_median": A["confound"]["median_confound_rho"],
                    "compartment_lead_vs_axon_directionality": A["confound"]["lead_vs_axon_d"]},
            },
            "verdict": va,
            "verdict_detail": a_detail,
            "provenance": {"mode": A["mode"], "env": A["env"],
                           "pilot_byte_identical": {k: v["byte_identical"] for k, v in A["pilot_check"].items()},
                           "pilot_max_rel_diff": {k: v.get("max_rel_diff") for k, v in A["pilot_check"].items()},
                           "environment_check": env_checks.get("A"), "elapsed_s": A["elapsed_s"]},
        },
        "B_qesn_chip_lattice": {
            "architecture": ("QESN lattice reservoir on the chip geometry (classical, Schrodinger-inspired lattice "
                             "reservoir corrected to an exactly unitary Crank-Nicolson step, leak and Kerr-type phase "
                             "nonlinearity; 480 lattice cells, 1440 readout features [Re psi, Im psi, |psi|^2])"),
            "source": SOURCES["B"],
            "what_was_tested": ("Whether the chip-geometry lattice reservoir forecasts the square-root spike counts of "
                                "all 76 electrodes 25 ms (h=1) and 100 ms (h=4) ahead better than linear and "
                                "non-spatial reservoir baselines, and whether the chip geometry matters (ablations)."),
            "data": "Brewer 4-compartment MEA (CC0): 9 NoStim recordings x 76 well electrodes x 300 s; 25 ms bins, sqrt counts.",
            "split": ("per recording, time-ordered: train 60 %, validation 10 % (hyper-parameters and ridge alpha), "
                      "test 30 % (refit on train + validation). Every model sees the same inputs."),
            "baselines": ["persistence (last bin)",
                          "var_ridge: ridge VAR with L in {1, 2, 4, 8} lags chosen on validation",
                          "esn_nonspatial: sparse leaky echo-state network with 1440 nodes (same number of readout features)",
                          "geometry ablations: lattice_full (no walls), lattice_walls (compartments isolated), lattice_chip_shuffled (electrode positions permuted)"],
            "metric": ("test MSE of sqrt counts (mean over electrodes); relative change (%) per recording, mean over "
                       "the 9 recordings, 95 % CI by bootstrap over recordings (4000 resamples); n_better = recordings "
                       "where the lattice has lower MSE."),
            "result": {
                "primary_contrasts": ["h1|lattice_chip_vs_var_ridge", "h4|lattice_chip_vs_var_ridge",
                                      "h1|lattice_chip_vs_esn_nonspatial", "h4|lattice_chip_vs_esn_nonspatial"],
                "h1|lattice_chip_vs_var_ridge": bs["h1|lattice_chip_vs_var_ridge"],
                "h4|lattice_chip_vs_var_ridge": bs["h4|lattice_chip_vs_var_ridge"],
                "h1|lattice_chip_vs_esn_nonspatial": bs["h1|lattice_chip_vs_esn_nonspatial"],
                "h4|lattice_chip_vs_esn_nonspatial": bs["h4|lattice_chip_vs_esn_nonspatial"],
                "h1|lattice_chip_vs_persistence": bs["h1|lattice_chip_vs_persistence"],
                "h4|lattice_chip_vs_persistence": bs["h4|lattice_chip_vs_persistence"],
                "geometry_ablations": {k: v for k, v in ab.items() if k != "mean_test_mse"},
                "mean_test_mse": ab["mean_test_mse"],
                "crank_nicolson_norm_ratio": B["crank_nicolson_norm_ratio"],
                "all_contrasts": bs,
            },
            "verdict": vb,
            "verdict_detail": b_detail,
            "provenance": {"mode": B["mode"], "env": B["env"], "reduced_check": B.get("reduced_check"),
                           "summary_check": B["summary_check"], "pilot_file": B.get("pilot_file"),
                           "environment_check": env_checks.get("B"), "elapsed_s": B["elapsed_s"]},
        },
        "C_doselattice_encoder": {
            "architecture": ("DoseLattice: fixed 1-D diffusion lattice over log10 concentration (QESN mass-conserving "
                             "stencil, 64 sites, 3 bandwidths, masked Nadaraya-Watson readout) added as an extra "
                             "context encoder to the NeuroTrajectory decoder"),
            "source": SOURCES["C"],
            "what_was_tested": ("Whether adding the DoseLattice encoder to NeuroTrajectory lowers the curve MAE of the "
                                "forecast at unmeasured concentrations for unseen chemicals, with k = 1-4 measured levels."),
            "data": ("EPA NFA (public domain): 49 held-out chemicals of outer fold 0; training on the other 194 chemicals "
                     "(folds 1-4). DEVELOPMENT FOLD 0 ONLY: fold 0 is excluded from the reported folds 1-4, which are "
                     "not evaluated here."),
            "split": ("chemical-level 5-fold split (repo/results/splits_epa_nfa.json), outer fold 0 as test; 5 seeded "
                      "context designs per chemical and k (same designs as trajectory_cv)."),
            "baselines": ["neurotrajectory_v1_saved: the primary v1 fold-0 NeuroTrajectory ensemble (3 seeds, laplace, interpolation-informed decoder, n_freq 8), loaded from data/processed/models",
                          "neurotrajectory_retrained: the same configuration and seeds retrained in this run (training-noise control)",
                          "best simple baseline per k among zero, context_mean, loglinear_interp, hill_per_endpoint, analog_knn"],
            "metric": ("curve MAE (vehicle-normalised units) per chemical, averaged over designs; paired bootstrap over "
                       "chemicals (run_trajectory_cv.paired_boot, 4000 resamples, seed 0) of DoseLattice minus reference; "
                       "ci95_rel_pct = CI divided by the reference mean."),
            "result": {"primary_contrast": "doselattice_vs_neurotrajectory_v1_saved", "by_k": C["by_k"],
                       "retrained_vs_saved_max_abs_diff_curve_mae": retrain_diff,
                       "integrity_vs_trajectory_cv_csv": C["integrity"], "port_fidelity": C["port_fidelity"]},
            "verdict": vc,
            "verdict_detail": c_detail,
            "label": "development fold 0 only; not a headline result",
            "provenance": {"mode": "full rerun with the v1 fold-0 configuration", "env": C["env"], "config": C["config"], "device": C["device"],
                           "timing": C["timing"], "elapsed_s": C["elapsed_s"], "pilot_reference": C["pilot_reference"],
                           "pilot_vs_this_run": pilot_vs_run},
        },
    }
    table = []
    table.append({"architecture": "A_eikonal_latency", "comparison": "chip_eik_c minus idw_comp", "metric": "held-out latency MAE (ms)",
                  "chip_model_mae_ms": round(hs["mean_mae_ms"]["chip_eik_c"], 2),
                  "baseline_mae_ms": {k: round(hs["mean_mae_ms"][k], 2) for k in ("comp_mean", "idw", "idw_comp", "free_eik")},
                  "effect_ms": round(prim["mean"], 2), "ci95_ms": [round(x, 2) for x in prim["ci95"]],
                  "n_better": prim["recordings_improved"], "n": prim["n"], "verdict": va})
    table.append({"architecture": "A_eikonal_latency", "comparison": "chip_eik_c minus free_eik (geometry ablation)",
                  "metric": "held-out latency MAE (ms)", "effect_ms": round(free["mean"], 2),
                  "ci95_ms": [round(x, 2) for x in free["ci95"]], "n_better": free["recordings_improved"], "n": free["n"],
                  "verdict": va})
    for h in B_HORIZONS:
        for ref in ("var_ridge", "esn_nonspatial"):
            s = bs[f"h{h}|lattice_chip_vs_{ref}"]
            table.append({"architecture": "B_qesn_chip_lattice", "comparison": f"h{h}|lattice_chip_vs_{ref}",
                          "metric": "test MSE, relative change (%)", "effect_pct": s["mean_rel_pct"], "ci95_pct": s["ci95"],
                          "n_better": s["n_better"], "n": s["n"], "verdict": vb})
    for h in B_HORIZONS:
        for x in ("lattice_full", "lattice_walls", "lattice_chip_shuffled"):
            s = ab[f"h{h}|lattice_chip_vs_{x}"]
            table.append({"architecture": "B_qesn_chip_lattice", "comparison": f"h{h}|lattice_chip_vs_{x} (geometry ablation)",
                          "metric": "test MSE, relative change (%)", "effect_pct": s["mean_rel_pct"], "ci95_pct": s["ci95"],
                          "n_better": s["n_better"], "n": s["n"], "n_within_1pct": s["n_within_1pct"], "verdict": vb})
    for k in C_KS:
        s = C["by_k"][str(k)]["doselattice_vs_neurotrajectory_v1_saved"]
        table.append({"architecture": "C_doselattice_encoder", "comparison": f"k={k}|doselattice_vs_neurotrajectory_v1_saved",
                      "metric": "curve MAE, relative change (%)", "effect_pct": s["rel_change_pct"], "ci95_pct": s["ci95_rel_pct"],
                      "effect_abs": s["mean_diff"], "ci95_abs": s["ci95"],
                      "n_better": s["n_chem_improved"], "n": s["n_chemicals"], "verdict": vc})
    return {
        "title": "Own prior architectures, evaluated on real data",
        "status": ("APPENDIX - parity and negative results for the team lead's earlier architectures; exploratory "
                   "tests, not part of the headline; C is on the development fold 0 only"),
        "generated_by": "repo/scripts/appendix_own_architectures.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_models": ("NeuroTrajectory v1 cross-validated models (data/processed/models), the primary models; "
                             "the exploratory pilot of C used the v2 fold-0 configuration (n_freq 3) and is superseded here"),
        "verdict_vocabulary": list(VERDICT_WORDS),
        "environment_note": ("A and B were run with the libraries the pilots used (numpy " + A["env"]["numpy"] + ", pandas "
                             + A["env"]["pandas"] + ") so that byte identity with the pilot outputs can be checked; the "
                             "same code was rerun with the project's data/.pylibs (see each provenance.environment_check). "
                             "C was run with the project's data/.pylibs (numpy " + C["env"]["numpy"] + ", torch "
                             + C["env"].get("torch", "?") + ", device " + C["device"] + ")."),
        "runtime_note": runtime_note,
        "b_full_rerun_cpu_min_estimate": b_full_min,
        "language_rules": ["QESN is described as a classical, Schrodinger-inspired lattice reservoir (corrected to a unitary step); nothing quantum is computed",
                           "Brewer burst-latency maps are not called propagation velocity: they correlate with firing rate",
                           "no component in this appendix is used by the headline model"],
        "summary_table": table,
        "architectures": arch,
        "parts": {k: {"sha256": sha256_file(PARTS / f"{k}.json")} for k in ("A", "B", "C")},
    }


def summarize() -> Path:
    parts = {}
    for k in ("A", "B", "C"):
        p = PARTS / f"{k}.json"
        if not p.exists():
            raise SystemExit(f"missing part {p}; run --parts {k} first")
        parts[k] = json.loads(p.read_text(encoding="utf-8"))
    env_checks = {}
    pa = PARTS / "A_projenv.json"
    if pa.exists():
        x = json.loads(pa.read_text(encoding="utf-8"))
        env_checks["A"] = {"env": x["env"], "note": "same ported code rerun with the project's data/.pylibs libraries",
                           "pilot_byte_identical": {k: v["byte_identical"] for k, v in x["pilot_check"].items()},
                           "pilot_max_rel_diff": {k: v.get("max_rel_diff") for k, v in x["pilot_check"].items()},
                           "max_rel_diff_vs_primary_run": max_rel_diff(x["heldout"], parts["A"]["heldout"])}
    pb = PARTS / "B_projenv.json"
    if pb.exists():
        x = json.loads(pb.read_text(encoding="utf-8"))
        env_checks["B"] = {"env": x["env"], "note": "same reduced check rerun with the project's data/.pylibs libraries",
                           "reduced_check": x.get("reduced_check"), "summary_check": x["summary_check"]}
    final = build_final(parts["A"], parts["B"], parts["C"], env_checks)
    out = RESULTS / OUT_NAME
    out.write_text(json.dumps(final, indent=2), encoding="utf-8")
    for row in final["summary_table"]:
        print(json.dumps(row))
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--parts", nargs="*", default=[], choices=["A", "B", "C"])
    ap.add_argument("--b-mode", default="reuse", choices=["reuse", "full"])
    ap.add_argument("--b-check-recs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--c-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--c-steps", type=int, default=None, help="override training steps (smoke tests only)")
    ap.add_argument("--pilot-dir", default=str(PILOT_DIR))
    ap.add_argument("--suffix", default="", help="write parts as <part><suffix>.json (e.g. an environment check)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    pilot_dir = Path(a.pilot_dir)
    for part in a.parts:
        if part == "A":
            obj = run_part_a(pilot_dir)
        elif part == "B":
            obj = run_part_b(pilot_dir, a.b_mode, a.b_check_recs)
        else:
            obj = run_part_c(pilot_dir, a.c_seeds, a.c_steps)
        print(f"part {part} -> {write_part(part, obj, a.suffix)} ({obj['elapsed_s']} s)", flush=True)
    if a.summarize:
        summarize()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
