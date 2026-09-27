"""Make the ten report figures of NeuroChip Twin v2 from results/*.json and the data bundle.

    python scripts/make_figures.py                 # all figures
    python scripts/make_figures.py fig3 fig4       # a subset (figure numbers or full stems)

Sources: repo/results/*.json (and the per-chemical CSV written next to trajectory_cv.json),
repo/data_bundle/ (NFA tasks and the cross-validated NeuroTrajectory weights). No number is typed
by hand. Quantities that only the figures derive (which example chemicals the selection rule
picks, counts shown in panels, bootstrap bands) are written to results/figures_derived.json so the
captions (report/figures/captions.md, template syntax of scripts/render_docs.py) can cite them;
captions.md is then rendered to report/figures/captions_rendered.md to prove every reference
resolves. Outputs: report/figures/<stem>.pdf (vector) and <stem>.png (200 dpi), width 6.5 in.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "report" / "figures"
sys.path.insert(0, str(REPO / "src"))

W = 6.5            # full text width (in)
SEED = 20260926    # bootstrap seed for figure-only bands
N_BOOT = 4000

# ----------------------------------------------------------------------------- style
# Okabe-Ito hues in an order validated for adjacent CVD separation; grey is the neutral.
C = {"nt": "#0072B2", "interp": "#E69F00", "knn": "#009E73", "ref": "#D55E00",
     "ctx": "#56B4E9", "hill": "#CC79A7", "zero": "#8C8C8C", "ink": "#1A1A1A",
     "muted": "#5E5E5E", "grid": "#D9D9D9", "band": "#9CC3E6"}
METHODS = {  # key: (label, colour, marker)
    "neurotrajectory": ("NeuroTrajectory", C["nt"], "o"),
    "analog_knn": ("Analog kNN", C["knn"], "s"),
    "loglinear_interp": ("Log-linear interpolation", C["interp"], "D"),
    "hill_per_endpoint": ("Hill per endpoint", C["hill"], "^"),
    "context_mean": ("Context mean", C["ctx"], "v"),
    "zero": ("Zero effect", C["zero"], "x"),
}
FEATURE_LABELS = {
    "firing_rate_mean": "Firing rate", "burst_rate": "Burst rate",
    "per_burst_interspike_interval": "Within-burst ISI", "per_burst_spike_percent": "% spikes in bursts",
    "burst_duration_mean": "Burst duration", "interburst_interval_mean": "Inter-burst interval",
    "active_electrodes_number": "Active electrodes", "bursting_electrodes_number": "Bursting electrodes",
    "network_spike_number": "Network spikes (NS)", "network_spike_peak": "NS peak",
    "spike_duration_mean": "NS duration", "per_network_spike_spike_percent": "% spikes in NS",
    "inter_network_spike_interval_mean": "Inter-NS interval", "network_spike_duration_std": "NS duration SD",
    "per_network_spike_spike_number_mean": "Spikes per NS",
    "correlation_coefficient_mean": "Correlation (STTC)", "mutual_information_norm": "Mutual information",
}
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6, "legend.frameon": False,
    "legend.handlelength": 1.6, "legend.borderaxespad": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.labelcolor": C["ink"], "text.color": C["ink"], "xtick.color": "#333333",
    "ytick.color": "#333333", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.3, "lines.markersize": 4, "errorbar.capsize": 0, "axes.titlepad": 5,
    "axes.titlelocation": "left", "figure.dpi": 100, "savefig.dpi": 200, "pdf.fonttype": 42,
    "ps.fonttype": 42, "svg.fonttype": "none", "axes.axisbelow": True, "hatch.linewidth": 0.5,
}
plt.rcParams.update(STYLE)

# ----------------------------------------------------------------------------- data access
_CACHE: dict[str, object] = {}
DERIVED: dict[str, object] = {}
USED: set[str] = set()


def J(name: str):
    if name not in _CACHE:
        _CACHE[name] = json.loads((RES / name).read_text(encoding="utf-8"))
        USED.add(name)
    return _CACHE[name]


def jget(name: str, path: str):
    """Same path semantics as scripts/render_docs.py (dict keys, integer list indices)."""
    obj = J(name)
    for part in path.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def ptitle(ax, letter: str, text: str = "", **kw):
    ax.set_title(rf"$\bf{{{letter}}}$   {text}" if text else rf"$\bf{{{letter}}}$", loc="left", **kw)


def hline(ax, y, **kw):
    ax.axhline(y, color=kw.pop("color", "#9A9A9A"), lw=kw.pop("lw", 0.7), ls=kw.pop("ls", "--"), zorder=0, **kw)


def vline(ax, x, **kw):
    ax.axvline(x, color=kw.pop("color", "#9A9A9A"), lw=kw.pop("lw", 0.7), ls=kw.pop("ls", "--"), zorder=0, **kw)


def ci_err(val, ci):
    return np.array([[val - ci[0]], [ci[1] - val]])


def wilson(k: int, n: int, z: float = 1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, (c - h, c + h)


def boot_mean_ci(vals: np.ndarray, rng: np.random.Generator):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    idx = rng.integers(0, len(vals), (N_BOOT, len(vals)))
    m = vals[idx].mean(1)
    return float(vals.mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", **kw)
    fig.savefig(OUT / f"{stem}.png", dpi=200, **kw)
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def tasks_bundle():
    if "tasks" not in _CACHE:
        from neurotwin.bundle import load_tasks
        _CACHE["tasks"] = load_tasks()
    return _CACHE["tasks"]


def per_chem_cv(name="trajectory_cv_per_chemical.csv") -> pd.DataFrame:
    USED.add(name)
    return pd.read_csv(RES / name)


# ============================================================================= fig1
def _box(ax, x, y, w, h, title, lines, fc, ec, fs=6.4, tfs=7.1, gap=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.1",
                                fc=fc, ec=ec, lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=tfs,
            fontweight="bold", zorder=3)
    top = y + h - 1.3 - 0.27 * tfs * (title.count("\n") + 1) - gap     # just below the title block
    ax.text(x + w / 2, top, "\n".join(lines), ha="center", va="top", fontsize=fs,
            linespacing=1.3, color="#303030", zorder=3)


def _arrow(ax, a, b, style="-|>", ls="-", color="#4D4D4D", rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig1_system():
    n_chem = jget("trajectory_cv.json", "n_chemicals")
    man = json.loads((REPO / "data_bundle/manifest.json").read_text(encoding="utf-8"))
    n_wells = man["n_wells"]
    lv = [len(t.levels) for t in tasks_bundle()]
    med_lv = int(np.median(lv))
    n_rec = jget("data_audit_brewer.json", "n_recordings")
    n_ax = jget("data_audit_brewer.json", "n_axons")
    n_tun = jget("data_audit_brewer.json", "tunnels_per_recording")
    DERIVED["fig1"] = {"n_chemicals": n_chem, "n_wells": n_wells, "median_levels_per_chemical": med_lv}

    H = 58.0
    fig = plt.figure(figsize=(W, W * H / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, H)
    ax.axis("off")
    grey_f, grey_e = "#F2F2F2", "#8C8C8C"
    mod_f, mod_e = "#E3EEF8", C["nt"]
    out_f, out_e = "#FBEBDD", "#B85A12"
    for x_, t_ in ((0.6, "Inputs (public, licensed)"), (21.6, "Data asset"), (40.6, "Digital twin"),
                   (81.6, "Decisions and outputs")):
        ax.text(x_, H - 0.6, t_, fontsize=6.6, color=C["muted"], va="top")
    top_y, top_h, bot_y, bot_h = 31.0, 24.0, 3.0, 16.0
    ym_top, ym_bot = top_y + top_h / 2, bot_y + bot_h / 2

    _box(ax, 0.5, top_y, 18, top_h, "EPA Network\nFormation Assay",
         ["rat cortical networks,", "48-well MEA", f"{n_chem} chemicals, {n_wells:,} wells",
          f"median {med_lv} concentrations", "DIV 5/7/9/12 × 17 features"], grey_f, grey_e)
    _box(ax, 0.5, bot_y, 18, bot_h, "Brewer 4-compartment\nhippocampal MEA",
         ["EC → DG → CA3 → CA1", "→ EC (feed-forward loop)", f"{n_tun} axon tunnels per array", f"{n_rec} recordings,",
          f"{n_ax} sorted axons (CC0)"], grey_f, grey_e, fs=6.1)
    _box(ax, 0.5, 21.2, 18, 7.4, "Reference data", ["EPA DNT labels, cytotoxicity,", "Harrill, Kosnik (unpaired)"],
         "white", grey_e, fs=6.0)

    _box(ax, 21.5, bot_y, 16, top_y + top_h - bot_y, "Tidy, frozen\nasset",
         ["chemical × dose × DIV", "× well × feature", "", "vehicle-normalised", "effect units",
          "", "cohort, QC rules", "and all splits frozen", "by SHA-256", "", "provenance and", "licence per row",
          "", "preregistered", "protocols (hashed)"], grey_f, grey_e)

    _box(ax, 40.5, top_y, 19, top_h, "NeuroTrajectory",
         ["few-shot conditional", "neural process,", "meta-trained across", "chemicals",
          "", "k measured doses →", "dose × DIV × feature", "forecast + scale"], mod_f, mod_e)
    _box(ax, 62, top_y, 16.5, top_h, "Calibration\nand abstention",
         ["cross-conformal,", "Mondrian by", "cytotoxicity × k", "", "abstain when the", "epistemic score",
          "is high"], mod_f, mod_e)
    _box(ax, 40.5, bot_y, 19, bot_h, "Chip layer",
         ["directed propagation", "per tunnel: feed-", "forward fraction and", "conduction time",
          "(empirical Bayes,", "leave-one-recording-out)"], mod_f, mod_e)

    _box(ax, 81.5, 47.4, 18, 7.6, "Hazard call", ["DNT call from k = 3 doses"], out_f, out_e, fs=5.9, tfs=6.9, gap=0.7)
    _box(ax, 81.5, 39.2, 18, 7.6, "Potency", ["BMC by interpolation", "of measured doses"], out_f, out_e, fs=5.9, tfs=6.9, gap=0.7)
    _box(ax, 81.5, top_y, 18, 7.6, "DoseCompass", ["next concentration by", "expected information"],
         out_f, out_e, fs=5.9, tfs=6.9, gap=0.7)
    _box(ax, 62, bot_y, 37.5, bot_h, "Chip-level outputs",
         ["per-edge feed-forward fraction with uncertainty,",
          "conduction-time distributions,", "zero-lag synchrony between compartments",
          "", "descriptive; no cross-platform calibration"], out_f, out_e, fs=6.2)

    # flows
    for x0, x1 in ((18.5, 21.5), (37.5, 40.5), (59.5, 62)):
        _arrow(ax, (x0, ym_top), (x1, ym_top))
        _arrow(ax, (x0, ym_bot), (x1, ym_bot))
    _arrow(ax, (18.5, 24.9), (21.5, 24.9), color="#8C8C8C")
    _arrow(ax, (78.5, 49), (81.5, 51.35))
    _arrow(ax, (78.5, ym_top), (81.5, 43.0))
    _arrow(ax, (78.5, 37), (81.5, 34.65))
    # lab loop: DoseCompass -> new wells -> NeuroTrajectory
    ax.plot([90.5, 90.5, 50], [top_y, 28.2, 28.2], color=C["nt"], lw=0.9, ls=(0, (3, 2)), zorder=4)
    _arrow(ax, (50, 28.2), (50, top_y), color=C["nt"], ls=(0, (3, 2)))
    ax.text(70.2, 27.4, "measure the proposed concentration, retrain the twin in batch", ha="center", va="top",
            fontsize=6.0, color=C["nt"], zorder=5)
    # colour legend: what each box category is
    leg = [Patch(fc=grey_f, ec=grey_e, label="data / reference"), Patch(fc=mod_f, ec=mod_e, label="twin components"),
           Patch(fc=out_f, ec=out_e, label="decisions / outputs")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=6.2, frameon=False,
              handlelength=1.2, handleheight=1.2, columnspacing=1.2)
    save(fig, "fig1_system")


# ============================================================================= fig2
ROWS = "ABCDEFGHJKLM"


def _el_xy(e: str):
    r, c = e[0], int(e[1:])
    return c - 1, len(ROWS) - 1 - ROWS.index(r)


def draw_chip(ax, graph):
    """Electrode map of one Brewer array from the recording's well and tunnel tables."""
    region_col = {"EC": "#9CC3E6", "DG": "#F5CFA0", "CA3": "#A8DCC8", "CA1": "#E7BFD6"}
    region_edge = {"EC": C["nt"], "DG": "#B36B00", "CA3": "#00765A", "CA1": "#A0507F"}
    by_reg: dict[str, list] = {}
    for wl in graph["wells"]:
        by_reg.setdefault(wl["subregion"], []).append(_el_xy(wl["electrode"]))
    for reg, pts in by_reg.items():
        p = np.array(pts)
        x0, y0 = p.min(0) - 0.5
        x1, y1 = p.max(0) + 0.5
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=region_col[reg], ec=region_edge[reg], lw=0.8, alpha=0.55, zorder=1))
        ax.scatter(p[:, 0], p[:, 1], s=7, color="#3A3A3A", lw=0, zorder=3)
        # label in the compartment corner left empty by the triangular electrode layout
        occ = {(int(a), int(b)) for a, b in p}
        corners = [(x0 + 0.6, y1 - 0.6, "left", "top"), (x1 - 0.6, y1 - 0.6, "right", "top"),
                   (x0 + 0.6, y0 + 0.6, "left", "bottom"), (x1 - 0.6, y0 + 0.6, "right", "bottom")]
        free = [c for c in corners if not any(abs(px - c[0]) < 1.2 and abs(py - c[1]) < 1.2 for px, py in occ)]
        tx, ty, ha, va = (free or corners)[0]
        ax.text(tx, ty, reg, ha=ha, va=va, fontsize=7.6, color=C["ink"], fontweight="bold")
    centroid = {reg: np.array(pts).mean(0) for reg, pts in by_reg.items()}
    pairs, mids = {}, {}
    for tn in graph["tunnels"]:
        a, b = tn["tunnel"].split("-")
        pair = tn["subregion_pair"]
        ra, rb = pair.split("-")   # subregion_pair already runs source -> target of the feed-forward loop
        (xa, ya), (xb, yb) = _el_xy(a), _el_xy(b)
        # tunnel endpoints sit on boundary rows outside the recorded electrode grid (no direct region
        # lookup), so orient (xa,ya)->(xb,yb) along ra->rb by proximity to each region's centroid
        da = (xa - centroid[ra][0]) ** 2 + (ya - centroid[ra][1]) ** 2
        db = (xb - centroid[ra][0]) ** 2 + (yb - centroid[ra][1]) ** 2
        if db < da:
            (xa, ya), (xb, yb) = (xb, yb), (xa, ya)
        ax.plot([xa, xb], [ya, yb], color="#4D4D4D", lw=2.4, solid_capstyle="butt", zorder=2)
        ax.scatter([xa, xb], [ya, yb], s=9, marker="s", color="white", edgecolor="#4D4D4D", lw=0.6, zorder=3)
        pairs.setdefault(pair, []).append(tn["tunnel"])
        mids.setdefault(pair, []).append(((xa + xb) / 2, (ya + yb) / 2, xb - xa, yb - ya))
    # feed-forward direction, one small arrow per compartment edge, placed on its own tunnel bundle
    for pair, segs in mids.items():
        mx = float(np.mean([s[0] for s in segs])); my = float(np.mean([s[1] for s in segs]))
        dx = float(np.mean([s[2] for s in segs])); dy = float(np.mean([s[3] for s in segs]))
        norm = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
        ux, uy = dx / norm, dy / norm
        L = 0.55
        ax.add_patch(FancyArrowPatch((mx - ux * L, my - uy * L), (mx + ux * L, my + uy * L), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.2, color=C["nt"], shrinkA=0, shrinkB=0, zorder=6))
    ax.text(5.5, -1.15, r"feed-forward: EC $\rightarrow$ DG $\rightarrow$ CA3 $\rightarrow$ CA1 $\rightarrow$ EC",
            ha="center", va="bottom", fontsize=6.0, color=C["nt"])
    ax.set_xlim(-0.7, 11.7)
    ax.set_ylim(-1.9, 11.7)
    ax.set_aspect("equal")
    ax.axis("off")
    return pairs


def fig2_data():
    tasks = tasks_bundle()
    from neurotwin.models.trajectory import DIVS, FEATURES
    n_chem = len(tasks)
    assert n_chem == jget("trajectory_cv.json", "n_chemicals")
    lv = pd.Series([len(t.levels) for t in tasks]).value_counts().sort_index()
    # consistency with the independently written DoseCompass file
    assert {str(k): int(v) for k, v in lv.items()} == {k: int(v) for k, v in J("r6_dosecompass.json")["levels_per_chemical"].items()}
    reps = pd.Series(np.concatenate([np.bincount(t._lvidx) for t in tasks])).value_counts().sort_index()
    m = np.concatenate([t.m for t in tasks])            # (wells, DIV, feature)
    avail = m.mean(0).T                                 # (feature, DIV)
    n_wells = int(m.shape[0])
    DERIVED["fig2"] = {"n_chemicals": n_chem, "n_wells": n_wells,
                       "levels_per_chemical": {str(k): int(v) for k, v in lv.items()},
                       "median_levels_per_chemical": float(np.median([len(t.levels) for t in tasks])),
                       "replicate_wells_per_level": {str(k): int(v) for k, v in reps.items()},
                       "median_replicate_wells_per_level": float(np.median(np.concatenate([np.bincount(t._lvidx) for t in tasks]))),
                       "n_chemical_levels": int(reps.sum()),
                       "availability_min_pct": round(float(100 * avail.min()), 1),
                       "availability_min_cell": [FEATURES[int(np.argmin(avail) // 4)], int(DIVS[int(np.argmin(avail) % 4)])],
                       "availability_median_pct": round(float(100 * np.median(avail)), 1)}
    g = next(gr for gr in J("r8_chiplayer.json")["graphs"] if gr["condition"] == "NoStim" and gr["orientation"] == "clockwise")
    DERIVED["fig2"]["chip_recording_shown"] = {"condition": g["condition"], "fid": g["fid"], "orientation": g["orientation"]}

    fig = plt.figure(figsize=(W, 3.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 0.9, 1.6], height_ratios=[1, 1], wspace=0.42, hspace=0.72,
                          left=0.2, right=0.995, top=0.92, bottom=0.12)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.pcolormesh(np.arange(avail.shape[1] + 1) - 0.5, np.arange(avail.shape[0] + 1) - 0.5, 100 * avail,
                       cmap="Blues", vmin=0, vmax=100, edgecolors="white", linewidth=0.4)   # vector cells
    ax.set_ylim(avail.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(len(DIVS)), [f"DIV {d}" for d in DIVS], fontsize=6.4)
    ax.set_yticks(range(len(FEATURES)), [FEATURE_LABELS[f] for f in FEATURES], fontsize=6.4)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(avail.shape[0]):
        for j in range(avail.shape[1]):
            v = 100 * avail[i, j]
            if v < 99.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v > 60 else C["ink"])
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.06, aspect=22)
    cb.set_label("wells with the feature defined (%)", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8)
    cb.outline.set_visible(False)
    ptitle(ax, "a", "17 features × 4 DIV")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(lv.index, lv.values, width=0.7, color=C["nt"], edgecolor="white", lw=0)
    for x, v in zip(lv.index, lv.values):
        ax.text(x, v + 4, str(v), ha="center", va="bottom", fontsize=5.8, color=C["muted"])
    ax.set_xticks(lv.index)
    ax.set_xlabel("tested concentrations per chemical")
    ax.set_ylabel("chemicals")
    ax.set_ylim(0, lv.max() * 1.18)
    ptitle(ax, "b", f"{n_chem} chemicals")
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(reps.index, reps.values, width=0.7, color=C["nt"], edgecolor="white", lw=0)
    top2 = reps.sort_values(ascending=False).index[:3]
    for x, v in zip(reps.index, reps.values):
        if x in top2:
            ax.text(x, v + 12, str(v), ha="center", va="bottom", fontsize=5.8, color=C["muted"])
    ax.set_xticks([x for x in (3, 6, 9, 12, 15, 18) if x <= reps.index.max()])
    ax.set_xlabel("replicate wells per concentration")
    ax.set_ylabel("chemical × concentration")
    ax.set_ylim(0, reps.max() * 1.2)
    ptitle(ax, "c", f"{n_wells:,} exposed wells")

    ax = fig.add_subplot(gs[:, 2])
    pairs = draw_chip(ax, g)
    DERIVED["fig2"]["tunnels_per_pair"] = {k.replace("-", "_"): len(v) for k, v in pairs.items()}
    DERIVED["fig2"]["electrodes_per_compartment"] = {r: sum(w["subregion"] == r for w in g["wells"])
                                                     for r in ("EC", "DG", "CA3", "CA1")}
    DERIVED["fig2"]["n_tunnels_shown"] = sum(len(v) for v in pairs.values())
    ptitle(ax, "d", "Brewer 4-compartment MEA")
    save(fig, "fig2_data")


# ============================================================================= fig3
def select_examples():
    """Rule: among never-inspected folds 1-4 at k = 3, the chemical whose paired improvement
    (best baseline - NeuroTrajectory curve MAE) is closest to the median and to the 10th percentile
    (numpy linear quantile); ties broken alphabetically."""
    if "examples" in _CACHE:
        return _CACHE["examples"]
    cv = per_chem_cv()
    best = jget("trajectory_cv.json", "by_k_folds_1to4.3.best_baseline")
    k3 = cv[(cv.k == 3) & (cv.fold >= 1)].pivot_table(index="chemical", columns="method", values="curve_mae")
    imp = (k3[best] - k3["neurotrajectory"]).sort_index()
    ref = jget("trajectory_cv.json", "by_k_folds_1to4.3.paired_vs_best")
    assert len(imp) == ref["n_chemicals"], "per-chemical CSV disagrees with trajectory_cv.json"
    assert abs(-imp.mean() - ref["mean_diff"]) < 1e-4
    sel = {}
    for name, q in (("median", 0.5), ("p10", 0.1)):
        target = float(np.quantile(imp.values, q))
        dist = (imp - target).abs()
        chem = sorted(dist.index[dist == dist.min()])[0]
        sel[name] = {"quantile": q, "quantile_value": round(target, 4), "chemical": chem,
                     "fold": int(cv.loc[cv.chemical == chem, "fold"].iloc[0]),
                     "improvement": round(float(imp[chem]), 4),
                     "curve_mae_neurotrajectory_mean5designs": round(float(k3.loc[chem, "neurotrajectory"]), 4),
                     "curve_mae_baseline_mean5designs": round(float(k3.loc[chem, best]), 4)}
    out = {"rule": ("folds 1-4, k = 3; improvement = best-baseline minus NeuroTrajectory curve MAE per chemical "
                    "(mean over the 5 seeded designs); chemical closest to the median and to the 10th "
                    "percentile (numpy linear quantile), ties alphabetical"),
           "best_baseline": best, "n_chemicals": int(len(imp)),
           "frac_model_worse": round(float((imp < 0).mean()), 4), **sel}
    _CACHE["examples"] = (out, k3, imp)
    return _CACHE["examples"]


def fig3_forecast_main():
    cv = per_chem_cv()
    rng = np.random.default_rng(SEED)
    ks = sorted(cv.k.unique())
    fig = plt.figure(figsize=(W, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 1.0], wspace=0.42, left=0.075, right=0.99,
                          top=0.88, bottom=0.26)
    # a: error vs k
    ax = fig.add_subplot(gs[0])
    order = ["zero", "context_mean", "hill_per_endpoint", "loglinear_interp", "analog_knn", "neurotrajectory"]
    off = dict(zip(order, np.linspace(-0.2, 0.2, len(order))))
    bands = {}
    for mth in order:
        lab, col, mk = METHODS[mth]
        mus, lo, hi = [], [], []
        for k in ks:
            v = cv[(cv.k == k) & (cv.method == mth)].curve_mae.values
            mu, ci = boot_mean_ci(v, rng)
            assert abs(mu - jget("trajectory_cv.json", f"by_k.{k}.curve_mae.mean.{mth}")) < 6e-5
            mus.append(mu); lo.append(ci[0]); hi.append(ci[1])
        bands[mth] = {"mean": [round(m_, 4) for m_ in mus], "ci95": [[round(a, 4), round(b, 4)] for a, b in zip(lo, hi)]}
        x = np.array(ks) + off[mth]
        ax.errorbar(x, mus, yerr=[np.array(mus) - lo, np.array(hi) - mus], color=col, marker=mk,
                    ms=3.6 if mth != "neurotrajectory" else 4.2, lw=1.0 if mth != "neurotrajectory" else 1.6,
                    elinewidth=0.8, label=lab, zorder=5 if mth == "neurotrajectory" else 3,
                    mfc="white" if mk not in ("x",) and mth != "neurotrajectory" else col, mew=0.9)
    DERIVED["fig3"] = {"curve_mae_bootstrap": {"n_boot": N_BOOT, "seed": SEED, "unit": "chemical", "by_method_k0to4": bands}}
    ax.set_xticks(ks)
    ax.set_xlabel("measured concentrations given (k)")
    ax.set_ylabel("curve MAE (vehicle SD)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0),
               fontsize=6.3, columnspacing=1.4, handlelength=2.0)
    ptitle(ax, "a", f"Forecast error, all {jget('trajectory_cv.json', 'n_chemicals')} chemicals")
    # b: paired differences
    ax = fig.add_subplot(gs[1])
    series = [("v1, all folds", "trajectory_cv.json", "by_k.{k}.curve_mae.paired_vs_best", C["nt"], "o", "white"),
              ("v1, folds 1-4 (never inspected)", "trajectory_cv.json", "by_k_folds_1to4.{k}.paired_vs_best", C["nt"], "o", C["nt"]),
              ("v2 grid (dose smoothness)", "trajectory_cv_v2.json", "by_k.{k}.curve_mae.paired_vs_best", C["muted"], "s", "white")]
    for i, (lab, f, p, col, mk, mfc) in enumerate(series):
        for k in ks:
            d = jget(f, p.format(k=k))
            ax.errorbar(k + (i - 1) * 0.22, d["mean_diff"], yerr=ci_err(d["mean_diff"], d["ci95"]), color=col,
                        marker=mk, mfc=mfc, ms=3.8, mew=0.9, elinewidth=0.9, label=lab if k == 0 else None)
    hline(ax, 0)
    bb = [jget("trajectory_cv.json", f"by_k.{k}.curve_mae.best_baseline") for k in ks]
    short = {"analog_knn": "kNN", "loglinear_interp": "interp."}
    ax.set_xticks(ks, [f"{k}\n{short.get(b, b)}" for k, b in zip(ks, bb)])
    ax.set_xlabel("k (best baseline at that k)")
    ax.set_ylabel(r"$\Delta$ curve MAE, model $-$ best baseline")
    lo_ = min(jget(f, p.format(k=k))["ci95"][0] for _, f, p, *_ in series for k in ks)
    ax.set_ylim(lo_ - 0.01, 0.08)
    ax.legend(loc="upper right", fontsize=6.0, borderaxespad=0.1)
    ax.text(0.02, 0.02, "below 0: model better", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.0, color=C["muted"])
    ptitle(ax, "b", "Paired difference (95% CI)")
    # c: per-chemical scatter, k = 3, folds 1-4
    ex, k3, imp = select_examples()
    best = ex["best_baseline"]
    ax = fig.add_subplot(gs[2])
    better = imp >= 0
    ax.scatter(k3.loc[better.index[better], best], k3.loc[better.index[better], "neurotrajectory"], s=7,
               color=C["nt"], alpha=0.75, lw=0, label="model better")
    ax.scatter(k3.loc[better.index[~better], best], k3.loc[better.index[~better], "neurotrajectory"], s=9,
               color=C["ref"], alpha=0.85, lw=0, marker="^", label="model worse")
    lim = [0.25, float(max(k3[best].max(), k3["neurotrajectory"].max()) * 1.15)]
    ax.plot(lim, lim, color="#9A9A9A", lw=0.7, ls="--", zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    for key, xy in (("median", (0.34, 0.12)), ("p10", (0.08, 0.62))):
        chem = ex[key]["chemical"]
        x, y = k3.loc[chem, best], k3.loc[chem, "neurotrajectory"]
        ax.scatter([x], [y], s=34, facecolor="none", edgecolor=C["ink"], lw=0.9, zorder=6)
        ax.annotate("median" if key == "median" else "10th pct", (x, y), xytext=xy,
                    textcoords="axes fraction", fontsize=5.9, ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=C["ink"], shrinkA=1, shrinkB=3))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator([0.5, 1, 2, 4]))
        axis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(f"{METHODS[best][0].lower()} MAE")
    ax.set_ylabel("NeuroTrajectory MAE")
    ax.legend(loc="upper left", fontsize=6.0, handletextpad=0.2, borderaxespad=0.1)
    ptitle(ax, "c", "Per chemical, k = 3, folds 1-4")
    save(fig, "fig3_forecast_main")


# ============================================================================= fig4
FIG4_FEATURES = ["firing_rate_mean", "burst_rate", "network_spike_number"]   # fixed across the three chemicals


def select_examples_fig4():
    """Rule (fig4 only; fig3/fig10 keep the median/10th-pct pair from select_examples()):
    on outer folds 1-4 at k = 3, rank chemicals by the paired improvement (best-baseline minus
    NeuroTrajectory curve MAE, mean over the 5 seeded designs). For each of the 90th, 50th and 10th
    percentiles, take the chemicals within +/-2 percentile points of that quantile and prefer, among
    them, the one with a reference EPA label (positive/negative, not 'unknown'), breaking ties by
    distance to the quantile value and then alphabetically."""
    if "examples_fig4" in _CACHE:
        return _CACHE["examples_fig4"]
    cv = per_chem_cv()
    best = jget("trajectory_cv.json", "by_k_folds_1to4.3.best_baseline")
    k3 = cv[(cv.k == 3) & (cv.fold >= 1)].pivot_table(index="chemical", columns="method", values="curve_mae")
    imp = (k3[best] - k3["neurotrajectory"]).sort_index()
    tasks = {t.chem: t for t in tasks_bundle()}
    labels = {c: tasks[c].label for c in imp.index if c in tasks}
    ranks = imp.rank(pct=True) * 100.0
    chosen: set[str] = set()
    sel = {}
    for name, q in (("p90", 90.0), ("median", 50.0), ("p10", 10.0)):
        target = float(np.quantile(imp.values, q / 100.0))
        window = [c for c in imp.index[(ranks - q).abs() <= 2.0] if c not in chosen]
        if not window:
            window = [c for c in imp.index if c not in chosen]
        labeled = [c for c in window if labels.get(c, "unknown") != "unknown"]
        pool = labeled if labeled else window
        dist = {c: (abs(imp[c] - target), c) for c in pool}
        chem = min(dist, key=dist.get)
        chosen.add(chem)
        sel[name] = {"quantile": q / 100.0, "quantile_value": round(target, 4), "chemical": chem,
                     "fold": int(cv.loc[cv.chemical == chem, "fold"].iloc[0]),
                     "improvement": round(float(imp[chem]), 4), "epa_label": labels.get(chem, "unknown"),
                     "preferred_labeled_chemical_used": chem in labeled,
                     "curve_mae_neurotrajectory_mean5designs": round(float(k3.loc[chem, "neurotrajectory"]), 4),
                     "curve_mae_baseline_mean5designs": round(float(k3.loc[chem, best]), 4)}
    out = {"rule": ("folds 1-4, k = 3; improvement = best-baseline minus NeuroTrajectory curve MAE per chemical "
                    "(mean over the 5 seeded designs); for each of the 90th/50th/10th percentiles, the chemical "
                    "within +/-2 percentile points of that quantile with a reference EPA label (preferred) closest "
                    "to the quantile value, else the closest chemical in the window, else the closest overall"),
           "best_baseline": best, "n_chemicals": int(len(imp)),
           "frac_model_worse": round(float((imp < 0).mean()), 4), **sel}
    _CACHE["examples_fig4"] = (out, k3, imp)
    return _CACHE["examples_fig4"]


def _load_cv_module():
    spec = importlib.util.spec_from_file_location("run_trajectory_cv", REPO / "scripts/run_trajectory_cv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _conformal_qhat_k3():
    """Pooled (non-Mondrian) 90%-nominal conformal half-width at k = 3, vehicle-SD units, if it can
    be sourced from conformal_r7.json; otherwise None (caller must fall back to a pre-conformal band)."""
    try:
        w = J("conformal_r7.json")["results"]["alpha_0.1"]["k3"]["conformal"]["width"]
        return float(w) / 2.0, True
    except Exception:  # noqa: BLE001
        return None, False


def fig4_examples():
    import torch
    from neurotwin.bundle import BUNDLE
    from neurotwin.models.cnp import load_fold_models, predict
    from neurotwin.models.trajectory import DIVS, FEATURES, level_means, predict_interp, split_context
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    rtc = _load_cv_module()
    ex, k3, imp = select_examples_fig4()
    qhat90, have_conformal = _conformal_qhat_k3()
    tasks = {t.chem: t for t in tasks_bundle()}
    feats = [FEATURES.index(f) for f in FIG4_FEATURES]
    panels = []
    for key in ("p90", "median", "p10"):
        info = ex[key]
        t = tasks[info["chemical"]]
        assert t.fold == info["fold"]
        models, cfg = load_fold_models(t.fold, "cpu", BUNDLE / "models")

        def fc(is_ctx, q):
            with torch.no_grad():
                preds = [predict(mm, t, is_ctx, q, device="cpu") for mm in models]
            mu = np.mean([p[0] for p in preds], 0)
            return mu

        des = rtc.designs(t, 3)
        maes_nt, maes_ip = [], []
        for ctx_lv in des:
            ic, it = split_context(t, ctx_lv)
            lv = np.unique(t.logc[it])
            mu = fc(ic, lv)
            _, obs, ok = level_means(t, it)
            maes_nt.append(float(np.abs(mu - obs)[ok].mean()))
            maes_ip.append(float(np.abs(predict_interp(t, ic, lv) - obs)[ok].mean()))
        # reproduction check against the CV file (bundle stores y as float16, CPU vs GPU)
        info["recomputed_mean5_neurotrajectory"] = round(float(np.mean(maes_nt)), 4)
        info["recomputed_mean5_baseline"] = round(float(np.mean(maes_ip)), 4)
        if abs(np.mean(maes_nt) - info["curve_mae_neurotrajectory_mean5designs"]) > 0.02:
            print(f"  WARNING {t.chem}: recomputed NT MAE {np.mean(maes_nt):.4f} vs CSV "
                  f"{info['curve_mae_neurotrajectory_mean5designs']:.4f}")
        # design shown: the design with the median curve MAE among its 5 designs (never the worst)
        di = int(np.argsort(maes_nt, kind="stable")[len(maes_nt) // 2])
        ctx_lv = des[di]
        ic, it = split_context(t, ctx_lv)
        grid = np.linspace(t.levels.min() - 0.25, t.levels.max() + 0.25, 90).astype(np.float32)
        mu_g = fc(ic, grid)
        ip_g = predict_interp(t, ic, grid)
        info.update({"design_shown": "seeded design of run_trajectory_cv.designs(task, 3) with the median "
                                     "curve MAE among its 5 designs", "design_index": di,
                     "design_curve_mae_nt_all5": [round(float(g_), 4) for g_ in maes_nt],
                     "measured_concentrations_uM": [round(float(10 ** x), 4) for x in ctx_lv],
                     "n_levels": int(len(t.levels)), "dnt_label": t.label,
                     "design_shown_curve_mae_neurotrajectory": round(maes_nt[di], 4),
                     "design_shown_curve_mae_baseline": round(maes_ip[di], 4),
                     "features_shown": FIG4_FEATURES,
                     "feature_rule": "fixed across the three chemicals (mean firing rate, burst rate, network spikes)",
                     "likelihood": cfg["likelihood"], "n_ensemble": len(models)})
        panels.append((key, t, ic, it, grid, mu_g, ip_g, info))
    DERIVED["fig4"] = {**ex, "band": ("conformal 90% (pooled across Mondrian strata, k = 3)" if have_conformal
                                     else "pre-conformal (conformal_r7.json unavailable)"),
                       "conformal_halfwidth_vehicle_sd": qhat90}

    LAB = {"p90": "90th-percentile chemical (large gain)", "median": "median chemical (typical case)",
           "p10": "10th-percentile chemical (model loses)"}
    fig = plt.figure(figsize=(W, 13.2))
    sfs = fig.subfigures(3, 1, hspace=0.05)
    for sf, (key, t, ic, it, grid, mu_g, ip_g, info) in zip(sfs, panels):
        axs = sf.subplots(len(feats), 4, sharex=True, sharey="row",
                          gridspec_kw=dict(wspace=0.14, hspace=0.6, left=0.14, right=0.99, top=0.62, bottom=0.17))
        sf.text(0.56, 0.015, r"log$_{10}$ concentration ($\mu$M)", ha="center", va="bottom", fontsize=6.8)
        sf.text(0.005, 0.63, "vehicle SD", rotation=90, ha="left", va="center", fontsize=6.2, color=C["muted"])
        sf.text(0.1, 0.975, f"{LAB[key]}:  {t.chem}", fontsize=7.6, fontweight="bold", va="top")
        sf.text(0.1, 0.895,
                f"fold {t.fold}, EPA reference label: {t.label};  gain vs {ex['best_baseline']} "
                f"(mean, folds 1-4): {info['improvement']:+.2f}",
                fontsize=6.2, color=C["muted"], va="top")
        sf.text(0.1, 0.845,
                f"curve MAE, median design shown (of {len(info['design_curve_mae_nt_all5'])} designs): "
                f"NeuroTrajectory {info['design_shown_curve_mae_neurotrajectory']:.2f} vs interpolation "
                f"{info['design_shown_curve_mae_baseline']:.2f};  mean of all designs: "
                f"{info['curve_mae_neurotrajectory_mean5designs']:.2f} vs {info['curve_mae_baseline_mean5designs']:.2f}",
                fontsize=6.2, color=C["muted"], va="top")
        for r, f in enumerate(feats):
            ymin, ymax = [], []
            for d, div in enumerate(DIVS):
                ax = axs[r, d]
                if qhat90 is not None:
                    lo, hi = mu_g[:, d, f] - qhat90, mu_g[:, d, f] + qhat90
                else:
                    lo, hi = mu_g[:, d, f] - 2.0, mu_g[:, d, f] + 2.0
                ax.fill_between(grid, lo, hi, color=C["band"], alpha=0.55, lw=0)
                ax.plot(grid, ip_g[:, d, f], color=C["interp"], lw=1.1, ls=(0, (3.5, 1.5)))
                ax.plot(grid, mu_g[:, d, f], color=C["nt"], lw=1.5)
                mc = t.m[ic][:, d, f]
                mt = t.m[it][:, d, f]
                ax.scatter(t.logc[it][mt], t.y[it][mt, d, f], s=9, facecolor="white", edgecolor=C["ref"], lw=0.8, zorder=4)
                ax.scatter(t.logc[ic][mc], t.y[ic][mc, d, f], s=9, color=C["ink"], lw=0, zorder=5)
                hline(ax, 0, ls=":", lw=0.6)
                ymin += [lo.min(), t.y[:, d, f][t.m[:, d, f]].min() if t.m[:, d, f].any() else 0]
                ymax += [hi.max(), t.y[:, d, f][t.m[:, d, f]].max() if t.m[:, d, f].any() else 0]
                if r == 0:
                    ax.set_title(f"DIV {div}", loc="center", fontsize=7.2)
            pad = 0.08 * (max(ymax) - min(ymin))
            axs[r, 0].set_ylim(max(-10.5, min(ymin) - pad), min(10.5, max(ymax) + pad))
            short_lab = {"network_spike_number": "Network spikes"}.get(FEATURES[f], FEATURE_LABELS[FEATURES[f]])
            axs[r, 0].set_ylabel(short_lab, fontsize=6.4, labelpad=2)
    band_lab = "90% conformal band, k = 3 (pooled)" if qhat90 is not None else "pre-conformal band (+/-2 SD, illustrative)"
    handles = [Line2D([], [], color=C["ink"], marker="o", ls="none", ms=3.5, label="measured wells given to the model (k = 3)"),
               Line2D([], [], color=C["ref"], marker="o", mfc="white", ls="none", ms=3.5, label="held-out wells (revealed)"),
               Line2D([], [], color=C["nt"], lw=1.5, label="NeuroTrajectory forecast (mean)"),
               Patch(color=C["band"], alpha=0.55, label=band_lab),
               Line2D([], [], color=C["interp"], lw=1.1, ls=(0, (3.5, 1.5)), label="log-linear interpolation")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.54, -0.028), fontsize=6.4,
               columnspacing=1.2)
    save(fig, "fig4_examples")


# ============================================================================= fig5
def fig5_calibration():
    R = J("conformal_r7.json")["results"]
    alphas = [a for a in R]
    ks = ["k1", "k2", "k3"]
    fig = plt.figure(figsize=(W, 4.3))
    gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.5, left=0.085, right=0.99, top=0.94, bottom=0.1)
    # a: coverage gap
    ax = fig.add_subplot(gs[0, 0])
    kcol = {"k1": C["ctx"], "k2": C["knn"], "k3": C["nt"]}
    kmk = {"k1": "v", "k2": "s", "k3": "o"}
    for i, a in enumerate(alphas):
        nom = R[a]["nominal"]
        for j, k in enumerate(ks):
            for m, (kind, mfc) in enumerate((("parametric", "white"), ("conformal", None))):
                d = R[a][k][kind]
                x = i + (j - 1) * 0.24 + (m - 0.5) * 0.09
                ax.errorbar(x, 100 * (d["coverage"] - nom), yerr=100 * ci_err(d["coverage"], d["ci95"]),
                            color=kcol[k], marker=kmk[k], mfc=mfc or kcol[k], ms=3.6, mew=0.9, elinewidth=0.8)
    hline(ax, 0)
    ax.set_xticks(range(len(alphas)), [f"{100 * R[a]['nominal']:.0f}% nominal" for a in alphas])
    ax.set_ylabel("observed $-$ nominal coverage (pp)")
    h = [Line2D([], [], color=kcol[k], marker=kmk[k], ls="none", ms=3.6, label=f"k = {k[1]}") for k in ks]
    h += [Line2D([], [], color=C["muted"], marker="o", mfc="white", ls="none", ms=3.6, label="parametric (open)"),
          Line2D([], [], color=C["muted"], marker="o", ls="none", ms=3.6, label="conformal (filled)")]
    lo_a = min(100 * (R[a][k][kind]["ci95"][0] - R[a]["nominal"]) for a in alphas for k in ks
               for kind in ("parametric", "conformal"))
    ax.set_ylim(lo_a - 0.3, 2.6)
    ax.legend(handles=[h[0], h[3], h[1], h[4], h[2]], loc="upper right", ncol=3, fontsize=5.8, columnspacing=0.9,
              handletextpad=0.2, borderaxespad=0.1)
    ptitle(ax, "a", "Coverage of held-out wells (95% CI)")
    # b: Mondrian coverage by regime at 90 %
    a90 = next(a for a in alphas if abs(R[a]["nominal"] - 0.9) < 1e-9)
    regimes = ["non-cytotoxic", "cytotoxic", "indeterminate"]
    rmk = {"non-cytotoxic": "o", "cytotoxic": "s", "indeterminate": "D"}
    rcol = {"non-cytotoxic": C["nt"], "cytotoxic": C["ref"], "indeterminate": C["muted"]}
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    for j, k in enumerate(ks):
        for r_i, rg in enumerate(regimes):
            d = R[a90][k]["by_regime"][rg]
            x = j + (r_i - 1) * 0.2
            axb.plot(x, 100 * d["conformal_coverage"], marker=rmk[rg], color=rcol[rg], ms=4, ls="none",
                     label=f"{rg} (n = {d['n_chemicals']})" if j == 0 else None)
            axc.plot(x, d["width"], marker=rmk[rg], color=rcol[rg], ms=4, ls="none")
        axc.plot(j + 0.42, R[a90][k]["parametric"]["width"], marker="_", color=C["ink"], ms=7, mew=1.2)
    hline(axb, 100 * R[a90]["nominal"])
    axb.set_xticks(range(3), [f"k = {k[1]}" for k in ks])
    axb.set_ylabel("conformal coverage (%)")
    lo_ = min(100 * R[a90][k]["by_regime"][rg]["conformal_coverage"] for k in ks for rg in regimes)
    hi_ = max(100 * R[a90][k]["by_regime"][rg]["conformal_coverage"] for k in ks for rg in regimes)
    axb.set_ylim(min(lo_, 90) - 1.5, max(hi_, 90) + 1.0)
    axb.legend(loc="lower right", fontsize=6.0, title="Mondrian stratum", title_fontsize=6.0)
    ptitle(axb, "b", f"Per-stratum coverage at {100 * R[a90]['nominal']:.0f}% nominal")
    axc.set_xticks(range(3), [f"k = {k[1]}" for k in ks])
    axc.set_ylabel(f"mean {100 * R[a90]['nominal']:.0f}% interval width (SD)")
    w_all = [R[a90][k]["by_regime"][rg]["width"] for k in ks for rg in regimes] + [R[a90][k]["parametric"]["width"] for k in ks]
    axc.set_ylim(min(w_all) - 1.6, max(w_all) + 0.3)
    axc.plot([], [], marker="_", color=C["ink"], ms=7, mew=1.2, ls="none", label="parametric, pooled")
    for rg in regimes:
        axc.plot([], [], marker=rmk[rg], color=rcol[rg], ms=4, ls="none", label=f"conformal, {rg}")
    axc.legend(loc="lower center", fontsize=5.9, ncol=2, columnspacing=0.8, handletextpad=0.2, borderaxespad=0.1)
    ptitle(axc, "c", "Price of calibration: interval width")
    # d: abstention
    ab = J("conformal_r7.json")["abstention"]
    ax = fig.add_subplot(gs[1, 1])
    kk = sorted(ab["by_k"], key=int)
    for j, k in enumerate(kk):
        d = ab["by_k"][k]
        ax.plot([j, j], [d["mae_retained"], d["mae_abstained"]], color="#BDBDBD", lw=1.0, zorder=1)
        ax.plot(j, d["mae_retained"], marker="o", color=C["nt"], ms=4.5, ls="none")
        ax.plot(j, d["mae_abstained"], marker="X", color=C["ref"], ms=5, ls="none")
        if j == 0:
            ax.text(j + 0.1, d["mae_retained"], "retained", fontsize=6.0, va="center", color=C["muted"])
            ax.text(j + 0.1, d["mae_abstained"], "abstained", fontsize=6.0, va="center", color=C["muted"])
    ax.set_xticks(range(len(kk)), [f"k = {k}\n{100 * ab['by_k'][k]['rate']:.1f}% abstained" for k in kk])
    ax.set_xlim(-0.4, len(kk) - 0.5)
    ax.set_ylabel("curve MAE (vehicle SD)")
    ptitle(ax, "d", "Abstention flags the harder forecasts")
    # mirror of values whose source keys contain '.' (not addressable by the template path syntax)
    DERIVED["fig5"] = {"source": "conformal_r7.json results[alpha_*]", "by_nominal": {
        f"nominal{100 * R[a]['nominal']:.0f}": {k: {
            "parametric_coverage": R[a][k]["parametric"]["coverage"], "parametric_ci95": R[a][k]["parametric"]["ci95"],
            "parametric_width": R[a][k]["parametric"]["width"], "conformal_coverage": R[a][k]["conformal"]["coverage"],
            "conformal_ci95": R[a][k]["conformal"]["ci95"], "conformal_width": R[a][k]["conformal"]["width"],
            "by_regime": {rg.replace("-", "_"): R[a][k]["by_regime"][rg] for rg in regimes}} for k in ks}
        for a in alphas}}
    save(fig, "fig5_calibration")


# ============================================================================= fig6
def fig6_hazard():
    D = J("dnt_r4.json")
    mc3 = D["variants"]["k3"]["mcnemar_vs_epa"]
    rules = list(mc3)
    n_same = mc3[rules[0]]["n_chemicals"]
    rows = [("NeuroTrajectory, k = 3\n(pre-specified)", mc3[rules[0]]["ours"], C["nt"], "o", C["nt"]),
            ("NeuroTrajectory,\nall tested doses", D["variants"]["full"]["mcnemar_vs_epa"][rules[0]]["ours"], C["ctx"], "s", C["ctx"])]
    for r in rules:
        rows.append((f"EPA rule {r}", mc3[r]["epa_rule_same_chemicals"], C["ref"], "D", "white"))
    fig = plt.figure(figsize=(W, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], height_ratios=[1.05, 1], wspace=0.42, hspace=0.62,
                          left=0.17, right=0.985, top=0.93, bottom=0.1)
    ys = np.arange(len(rows))[::-1]
    for c_i, metric in enumerate(("sensitivity", "specificity")):
        ax = fig.add_subplot(gs[0, c_i])
        for y, (name, d, col, mk, mfc) in zip(ys, rows):
            ax.errorbar(d[metric], y, xerr=ci_err(d[metric], d[f"{metric}_ci95"]), color=col, marker=mk, ms=4,
                        elinewidth=0.9, mfc=mfc, mew=0.9)
        ax.set_yticks(ys, [r[0] for r in rows] if c_i == 0 else [], fontsize=6.4)
        ax.set_xlim(0.4, 1.03)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel(metric)
        ax.grid(axis="x", color=C["grid"], lw=0.5)
        if c_i == 0:
            ptitle(ax, "a", f"DNT hazard call, same {n_same} chemicals (95% CI)")
    # b: McNemar discordant pairs
    ax = fig.add_subplot(gs[0, 2])
    mx = max(max(mc3[r]["only_ours_correct"], mc3[r]["only_epa_correct"]) for r in rules)
    for i, r in enumerate(rules):
        d = mc3[r]
        y = len(rules) - 1 - i
        ax.barh(y, d["only_ours_correct"], color=C["nt"], height=0.5)
        ax.barh(y, -d["only_epa_correct"], color=C["ref"], height=0.5)
        ax.text(d["only_ours_correct"] + 0.4, y, f"{d['only_ours_correct']}", va="center", fontsize=6.2)
        ax.text(-d["only_epa_correct"] - 0.4, y, f"{d['only_epa_correct']}", va="center", ha="right", fontsize=6.2)
    ax.axvline(0, color="#333333", lw=0.6)
    ax.set_yticks(range(len(rules)), [f"vs {r}\np = {mc3[r]['exact_p']:.3f}" for r in rules][::-1], fontsize=6.3)
    ax.set_xlim(-mx - 3, mx + 3)
    ax.set_ylim(-0.5, len(rules) - 0.5)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.set_ylim(-0.5, len(rules) - 0.1)
    ax.text(-0.6, len(rules) - 0.45, "only EPA\ncorrect", ha="right", va="center", fontsize=6.0, color=C["muted"])
    ax.text(0.6, len(rules) - 0.45, "only k = 3\ncorrect", ha="left", va="center", fontsize=6.0, color=C["muted"])
    ax.set_xlabel("discordant chemicals", fontsize=6.8)
    ptitle(ax, "b", "Exact McNemar test")
    # c: potency from model-derived BMC (negative result)
    ax = fig.add_subplot(gs[1, 0:2])
    ks = sorted(J("potency_r3.json")["by_k"], key=int)
    for i, (lab, f, p, col, mk) in enumerate((
            ("R3 v2: BMC from the model's dense dose grid", "potency_r3.json",
             "by_k.{k}.replicate_split_paired_vs_all.loglinear_interp", C["nt"], "o"),
            ("R3b: measured doses completed by the model", "potency_r3b.json",
             "by_k.{k}.fill_paired_vs.loglinear_interp", C["knn"], "s"))):
        for k in ks:
            d = jget(f, p.format(k=k))
            ax.errorbar(int(k) + (i - 0.5) * 0.18, d["mean_diff"], yerr=ci_err(d["mean_diff"], d["ci95"]), color=col,
                        marker=mk, ms=4, elinewidth=0.9, label=lab if k == ks[0] else None)
    hline(ax, 0)
    ax.set_xticks([int(k) for k in ks])
    ax.set_xlim(int(ks[0]) - 0.45, int(ks[-1]) + 0.45)
    ax.set_xlabel("measured concentrations (k)")
    ax.set_ylabel(r"$\Delta$ composite potency error" + "\n(model $-$ interpolation)", fontsize=6.8)
    hi_ = max(jget(f, p.format(k=k))["ci95"][1] for f, p in (("potency_r3.json", "by_k.{k}.replicate_split_paired_vs_all.loglinear_interp"),
                                                              ("potency_r3b.json", "by_k.{k}.fill_paired_vs.loglinear_interp")) for k in ks)
    ax.set_ylim(min(-0.03, ax.get_ylim()[0]), hi_ * 1.6)
    ax.legend(loc="upper left", fontsize=6.0, ncol=1, borderaxespad=0.1)
    ax.text(0.99, 0.03, "above 0: interpolating the measured doses is better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.9, color=C["muted"])
    ptitle(ax, "c", "Potency against the replicate-split reference (negative result)")
    # d: discrimination of the two variants
    ax = fig.add_subplot(gs[1, 2])
    var = [("k = 3\n(pre-specified)", D["variants"]["k3"], C["nt"], "o"), ("all tested\ndoses", D["variants"]["full"], C["ctx"], "s")]
    for i, (lab, d, col, mk) in enumerate(var):
        ax.errorbar(i, d["auroc"], yerr=ci_err(d["auroc"], d["auroc_ci95"]), color=col, marker=mk, ms=4.5, elinewidth=0.9)
    hline(ax, 0.5)
    ax.set_xticks(range(len(var)), [v[0] for v in var], fontsize=6.4)
    ax.set_xlim(-0.6, len(var) - 0.4)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUROC (95% CI)")
    ptitle(ax, "d", f"AUROC ({D['n_positive']} pos / {D['n_negative']} neg)")
    DERIVED["fig6"] = {"source": "dnt_r4.json variants.*.mcnemar_vs_epa[rule]", "n_chemicals_mcnemar": n_same,
                       "mcnemar_k3": {r.replace(".", ""): mc3[r] for r in rules},
                       "mcnemar_full": {r.replace(".", ""): D["variants"]["full"]["mcnemar_vs_epa"][r] for r in rules}}
    save(fig, "fig6_hazard")


# ============================================================================= fig7
def fig7_dosecompass():
    V1 = J("r6_dosecompass.json")
    V2 = J("r6b_dosecompass_anchored.json")
    ex = V1["example_chemical"]
    bmr = V1["bmr_vehicle_sd"]
    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(1, 2, wspace=0.3, left=0.085, right=0.985, top=0.95, bottom=0.57)
    gb = fig.add_gridspec(1, 2, wspace=0.32, width_ratios=[1.05, 1], left=0.25, right=0.985, top=0.44, bottom=0.09)
    sub = gs[0].subgridspec(2, 1, height_ratios=[2.1, 1], hspace=0.08)
    ax = fig.add_subplot(sub[0])
    x = np.array(ex["levels_log10uM"])
    P = {int(k): np.array(v) for k, v in ex["sampled_max_abs_summary_pctl"].items()}
    ax.fill_between(x, P[5], P[95], color=C["band"], alpha=0.35, lw=0, label="twin, 5-95%")
    ax.fill_between(x, P[25], P[75], color=C["band"], alpha=0.8, lw=0, label="twin, 25-75%")
    ax.plot(x, P[50], color=C["nt"], lw=1.4, label="twin, median")
    obs = np.array(ex["observed_max_abs_summary_all_levels"])
    ax.plot(x, obs, ls="none", marker="o", mfc="white", mec=C["ref"], ms=4, mew=0.9, label="observed (hidden)")
    fi = ex["first_level_idx"]
    ax.plot(x[fi], obs[fi], ls="none", marker="o", color=C["ink"], ms=4.5, label="measured first")
    hline(ax, bmr, ls=":", color=C["muted"])
    ax.text(x[0], bmr + 0.15, f"BMR = {bmr:g} SD", fontsize=5.9, color=C["muted"], va="bottom")
    ax.set_ylabel("max |effect| (SD)")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=5.7, ncol=2, columnspacing=0.8, handlelength=1.3)
    ptitle(ax, "a", f"{ex['chemical']}: posterior after one measurement")
    ax2 = fig.add_subplot(sub[1], sharex=ax)
    eig = {int(k): v for k, v in ex["eig_bits"].items()}
    ch = int(ex["chosen_idx"])
    for i, v in eig.items():
        ax2.bar(x[i], v, width=0.3, color=C["ref"] if i == ch else "#BDBDBD", lw=0)
    ax2.annotate("proposed next", (x[ch], eig[ch]), xytext=(0, 3), textcoords="offset points", ha="center",
                 va="bottom", fontsize=5.9, color=C["muted"])
    ax2.set_ylabel("EIG (bits)")
    ax2.set_ylim(0, max(eig.values()) * 1.45)
    ax2.set_xlabel(r"log$_{10}$ concentration ($\mu$M)")
    # b: composite error vs budget
    ax = fig.add_subplot(gs[1])
    Bs = sorted(V1["results"], key=lambda b: int(b.split("=")[1]))
    bx = [int(b.split("=")[1]) for b in Bs]
    r_mean = [V1["results"][b]["neurotrajectory"]["strategies"]["random"]["over_200_replicates"]["composite_error_mean"] for b in Bs]
    ax.fill_between(bx, [r["p2.5"] for r in r_mean], [r["p97.5"] for r in r_mean], color="#D9D9D9", lw=0,
                    label="random (95% of 200 runs)")
    series = [("fixed log-spaced", V1, "fixed_logspaced", C["interp"], "D", "-"),
              ("DoseCompass R6 (median first)", V1, "dosecompass", C["nt"], "o", "-"),
              ("DoseCompass R6b (top first)", V2, "dosecompass_anchored", C["nt"], "s", "--"),
              ("oracle (upper bound)", V1, "oracle_upper_bound", C["muted"], "^", ":")]
    for lab, src, key, col, mk, ls in series:
        y = [src["results"][b]["neurotrajectory"]["strategies"][key]["composite_error_mean"] for b in Bs]
        ax.plot(bx, y, color=col, marker=mk, ls=ls, ms=3.8, mfc="white" if ls == "--" else col, mew=0.9, label=lab)
    ax.set_xticks(bx)
    ax.set_xlabel("measured concentrations (budget B)")
    ax.set_ylabel("composite potency error")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.38)
    ax.legend(loc="upper center", fontsize=5.8, ncol=2, columnspacing=0.9, borderaxespad=0.1)
    ptitle(ax, "b", "Error by budget (NeuroTrajectory estimator)")
    # c: paired differences at B = 3 (two registered protocols)
    ax = fig.add_subplot(gb[0])
    pb1 = V1["results"]["B=3"]["neurotrajectory"]["paired_bootstrap_composite"]
    pb2 = V2["results"]["B=3"]["neurotrajectory"]["paired_bootstrap_composite"]
    rows = [("R6 (median first) vs fixed log-spaced (primary)", pb1["dosecompass_minus_fixed_logspaced"], C["nt"], "o"),
            ("R6 (median first) vs random", pb1["dosecompass_minus_random"], C["nt"], "o"),
            ("R6b (top first) vs fixed log-spaced (primary)", pb2["dosecompass_anchored_minus_fixed_logspaced"], C["nt"], "s"),
            ("R6b (top first) vs random, top-first order", pb2["dosecompass_anchored_minus_random_anchored"], C["nt"], "s"),
            ("R6b (top first) vs random, median-first order", pb2["dosecompass_anchored_minus_random_v1_median_first"], C["nt"], "s")]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, d, col, mk) in zip(ys, rows):
        prim = "primary" in lab
        ax.errorbar(d["mean_diff"], y, xerr=ci_err(d["mean_diff"], d["ci95"]), color=col, marker=mk, ms=4.2,
                    mfc=col if prim else "white", mew=0.9, elinewidth=1.1 if prim else 0.8)
    vline(ax, 0)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel(r"$\Delta$ composite error (95% CI)" + "\n< 0 favours DoseCompass")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ptitle(ax, "c", "Paired differences at B = 3")
    # d: activity detection at B = 3
    ax = fig.add_subplot(gb[1])
    S1 = V1["results"]["B=3"]["neurotrajectory"]["strategies"]
    S2 = V2["results"]["B=3"]["neurotrajectory"]["strategies"]
    strat = [("fixed log-spaced", S1["fixed_logspaced"], C["interp"], "D", None),
             ("DoseCompass R6 (median first)", S1["dosecompass"], C["nt"], "o", None),
             ("DoseCompass R6b (top first)", S2["dosecompass_anchored"], C["nt"], "s", "white"),
             ("random, median first", S1["random"]["over_200_replicates"], "#8C8C8C", "v", None),
             ("random, anchored", S2["random_anchored"]["over_200_replicates"], "#8C8C8C", "^", "white")]
    for m_i, metric in enumerate(("activity_accuracy", "activity_kappa")):
        for s_i, (lab, d, col, mk, mfc) in enumerate(strat):
            xx = m_i + (s_i - 2) * 0.13
            v = d[metric]
            if isinstance(v, dict):
                ax.errorbar(xx, v["mean"], yerr=[[v["mean"] - v["p2.5"]], [v["p97.5"] - v["mean"]]], color=col,
                            marker=mk, mfc=mfc or col, ms=4, mew=0.9, elinewidth=0.8, label=lab if m_i == 0 else None)
            else:
                ax.plot(xx, v, color=col, marker=mk, mfc=mfc or col, ms=4.2, mew=0.9, ls="none",
                        label=lab if m_i == 0 else None)
    ax.set_xticks([0, 1], ["activity accuracy", "activity kappa"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("agreement with reference call")
    ax.legend(loc="lower left", fontsize=5.9)
    ptitle(ax, "d", "Activity detection at B = 3")
    DERIVED["fig7"] = {"source": "r6_dosecompass.json / r6b_dosecompass_anchored.json results[B=*].neurotrajectory",
                       "B3_v1_paired": pb1, "B3_r6b_paired": pb2,
                       "composite_by_budget": {b.replace("=", ""): {
                           "fixed_logspaced": V1["results"][b]["neurotrajectory"]["strategies"]["fixed_logspaced"]["composite_error_mean"],
                           "dosecompass_v1": V1["results"][b]["neurotrajectory"]["strategies"]["dosecompass"]["composite_error_mean"],
                           "dosecompass_r6b": V2["results"][b]["neurotrajectory"]["strategies"]["dosecompass_anchored"]["composite_error_mean"],
                           "oracle_v1": V1["results"][b]["neurotrajectory"]["strategies"]["oracle_upper_bound"]["composite_error_mean"],
                           "random_v1": V1["results"][b]["neurotrajectory"]["strategies"]["random"]["over_200_replicates"]["composite_error_mean"]}
                           for b in Bs},
                       "B3_activity": {"fixed_logspaced": {m: S1["fixed_logspaced"][m] for m in ("activity_accuracy", "activity_kappa")},
                                       "dosecompass_v1": {m: S1["dosecompass"][m] for m in ("activity_accuracy", "activity_kappa")},
                                       "dosecompass_r6b": {m: S2["dosecompass_anchored"][m] for m in ("activity_accuracy", "activity_kappa")}},
                       "example_eig_bits_chosen": eig[ch], "example_chemical": ex["chemical"],
                       "example_proposed_uM": round(float(10 ** x[ch]), 4), "example_first_uM": round(float(10 ** x[fi]), 4)}
    save(fig, "fig7_dosecompass")


# ============================================================================= fig8
EDGE_ORDER = ["EC-DG", "DG-CA3", "CA3-CA1", "CA1-EC"]


def fig8_chip():
    P = J("chiplayer_params.json")
    R8 = J("r8_chiplayer.json")
    cv = R8["cv"]
    fig = plt.figure(figsize=(W, 4.5))
    gs = fig.add_gridspec(2, 6, wspace=1.3, hspace=0.6, left=0.085, right=0.98, top=0.94, bottom=0.1)
    # a: feed-forward fraction per edge
    ax = fig.add_subplot(gs[0, 0:2])
    pooled = {e["pair"]: e for e in P["fit_pair_edges"]}
    cond_mk = {"NoStim": ("o", C["nt"]), "HFS5": ("s", C["knn"]), "HFS40": ("^", C["hill"])}
    rng = np.random.default_rng(SEED)
    wil = {}
    for i, pair in enumerate(EDGE_ORDER):
        for rec in P["observed_per_recording"]:
            e = next(ed for ed in rec["edges"] if ed["pair"] == pair)
            if e["ff_fraction"] is None:
                continue
            mk, col = cond_mk[rec["condition"]]
            j = {"NoStim": -0.22, "HFS5": 0.0, "HFS40": 0.22}[rec["condition"]] + rng.uniform(-0.06, 0.06)
            ax.plot(i + j, e["ff_fraction"], marker=mk, color=col, ms=2.8, alpha=0.65, ls="none", mew=0)
        pe = pooled[pair]
        p, ci = wilson(pe["n_axons_ff"], pe["n_axons_ff"] + pe["n_axons_fb"])
        wil[pair] = {"ff_fraction": round(p, 4), "wilson95": [round(ci[0], 4), round(ci[1], 4)],
                     "n_axons": pe["n_axons_ff"] + pe["n_axons_fb"]}
        ax.errorbar(i + 0.42, p, yerr=ci_err(p, ci), color=C["ink"], marker="_", ms=8, mew=1.4, elinewidth=1.0)
    hline(ax, 0.5)
    ax.set_xticks(range(4), [p.replace("-", chr(10) + "→ ") for p in EDGE_ORDER], fontsize=6.2)
    ax.set_ylim(-0.05, 1.42)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("feed-forward fraction of axons")
    h = [Line2D([], [], marker=m, color=c, ls="none", ms=3, label=k) for k, (m, c) in cond_mk.items()]
    h.append(Line2D([], [], marker="_", color=C["ink"], ms=8, mew=1.4, ls="none", label="pooled NoStim, 95% CI"))
    ax.legend(handles=h, loc="upper left", fontsize=5.6, ncol=2, columnspacing=0.6, handletextpad=0.2,
              borderaxespad=0.0)
    ptitle(ax, "a", "Direction per edge")
    # b: conduction times per edge (pooled NoStim)
    ax = fig.add_subplot(gs[0, 2:4])
    for i, pair in enumerate(EDGE_ORDER):
        v = np.array(pooled[pair]["conduction"]["values_ms"])
        xs = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(xs, v + rng.uniform(-0.012, 0.012, len(v)), s=4, color=C["nt"], alpha=0.45, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=C["ink"], lw=1.4)
    ax.set_xticks(range(4), [p.replace("-", chr(10) + "→ ") for p in EDGE_ORDER], fontsize=6.2)
    ax.set_ylabel("conduction time (ms)")
    ax.set_ylim(0, None)
    ptitle(ax, "b", "Tunnel conduction (NoStim)")
    # c: burst lag
    ax = fig.add_subplot(gs[0, 4:6])
    lags = np.array([e["burst_propagation"]["lag_ms"] for g in R8["graphs"] for e in g["edges"]])
    corr = np.array([e["burst_propagation"]["correlation"] for g in R8["graphs"] for e in g["edges"]])
    bins = np.arange(-2050, 2051, 100)
    ax.hist(lags, bins=bins, color=C["nt"], lw=0)
    ax.set_xlabel("lag of peak burst correlation (ms)")
    ax.set_ylabel("edge × recording")
    ax.set_xlim(-2100, 2100)
    n0 = int((lags == 0).sum())
    ax.text(0.97, 0.95, f"{n0} of {len(lags)}\nat zero lag", transform=ax.transAxes, ha="right", va="top", fontsize=6.0)
    DERIVED["fig8"] = {"burst_lag_zero_n": n0, "burst_edges_n": int(len(lags)),
                       "burst_corr_median": round(float(np.median(corr)), 4),
                       "pooled_nostim_ff_fraction_wilson": wil}
    ptitle(ax, "c", "Population bursts are synchronous")
    # d/e: leave-one-recording-out CV
    folds = cv["folds"]
    for pi, (metric, lab, methods, letter, title) in enumerate((
            ("ff_fraction_mae", "feed-forward fraction MAE", ["tunnel_mean", "permuted_direction", "model"], "d",
             "LORO: direction per tunnel"),
            ("conduction_mae_ms", "conduction-time MAE (ms)", ["tunnel_mean", "model"], "e",
             "LORO: conduction time"))):
        ax = fig.add_subplot(gs[1, pi * 3:(pi + 1) * 3])
        names = {"tunnel_mean": "tunnel mean", "permuted_direction": "direction-\npermuted", "model": "shrinkage\nmodel"}
        cols = {"tunnel_mean": C["interp"], "permuted_direction": C["muted"], "model": C["nt"]}
        vals = np.array([[f["scores"][m][metric] for m in methods] for f in folds])
        for row in vals:
            ax.plot(range(len(methods)), row, color="#C8C8C8", lw=0.7, zorder=1)
        for j, m in enumerate(methods):
            ax.scatter(np.full(len(vals), j), vals[:, j], s=12, color=cols[m], zorder=3, lw=0)
            ax.plot([j - 0.2, j + 0.2], [cv["summary"][m][metric]] * 2, color=C["ink"], lw=1.4, zorder=4)
        ax.set_xticks(range(len(methods)), [names[m] for m in methods], fontsize=6.3)
        ax.set_xlim(-0.4, len(methods) + 1.25)
        ax.set_ylabel(lab)
        txt = ["model − baseline", "(95% CI over arrays)", ""]
        plain = {"tunnel_mean": "tunnel mean", "permuted_direction": "direction-permuted"}
        for base in [m for m in methods if m != "model"]:
            c = cv["comparisons"][base][metric]
            txt += [f"vs {plain[base]}:", f"{c['difference_model_minus_baseline']:+.3f} "
                    f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]", ""]
        ax.text(len(methods) - 0.35, np.mean(ax.get_ylim()), "\n".join(txt[:-1]), ha="left", va="center",
                fontsize=5.8, color=C["muted"], linespacing=1.25)
        ptitle(ax, letter, f"{title} ({len(folds)} arrays)")
    save(fig, "fig8_chip")


# ============================================================================= fig9
def fig9_crossassay():
    X2 = J("x2_harrill.json")
    X3 = J("x3_kosnik.json")
    fig = plt.figure(figsize=(W, 5.0))
    gt = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.35, left=0.09, right=0.98, top=0.95, bottom=0.56)
    gb = fig.add_gridspec(1, 1, left=0.3, right=0.98, top=0.43, bottom=0.08)
    # a: potency scatter, primary overlap, both active
    ax = fig.add_subplot(gt[0])
    pc = pd.DataFrame(X2["per_chemical"])
    prim = pc[pc.match_type.isin(["exact", "synonym"])]
    assert len(prim) == X2["primary_overlap"]["n_chemicals"]
    both = prim[prim.nfa_active.astype(bool) & prim.morph_active.astype(bool)].dropna(subset=["nfa_bmc_log10", "morph_bmc_log10"])
    hd = X2["headline"]
    assert len(both) == hd["n_both_active"]
    off = float(np.median(both.nfa_bmc_log10 - both.morph_bmc_log10))
    assert abs(off - hd["potency_offset_nfa_minus_morph_log10_median"]) < 1e-3
    ax.scatter(both.morph_bmc_log10, both.nfa_bmc_log10, s=13, color=C["nt"], lw=0, alpha=0.85, zorder=3)
    lo = float(min(both.morph_bmc_log10.min(), both.nfa_bmc_log10.min())) - 0.3
    hi = float(max(both.morph_bmc_log10.max(), both.nfa_bmc_log10.max())) + 1.5
    ax.plot([lo, hi], [lo, hi], color="#9A9A9A", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"morphology BMC, log$_{10}$ $\mu$M (Harrill imaging)")
    ax.set_ylabel(r"network BMC, log$_{10}$ $\mu$M (NFA)")
    ax.text(0.03, 0.97, f"n = {hd['n_both_active']} active in both;  median offset "
            f"{hd['potency_offset_nfa_minus_morph_log10_median']:+.2f} log$_{{10}}$\n"
            f"95% CI [{hd['potency_offset_ci95'][0]:+.2f}, {hd['potency_offset_ci95'][1]:+.2f}];  "
            f"Spearman {hd['spearman_potency_both_active']:.2f}",
            transform=ax.transAxes, va="top", fontsize=5.9)
    ptitle(ax, "a", "Network function changes before morphology")
    # c (top right): Kosnik discrimination
    ax = fig.add_subplot(gt[1])
    mc = X3["measured_acute_call"]
    rows = [("NFA reference (all doses)", mc["nfa_reference"], C["ref"], "o"), ("twin, k = 3", mc["twin_k3"], C["nt"], "s")]
    for m_i, metric in enumerate(("balanced_accuracy", "auroc")):
        for r_i, (name, d, col, mk) in enumerate(rows):
            x = m_i + (r_i - 0.5) * 0.36
            ax.errorbar(x, d[metric], yerr=ci_err(d[metric], d[f"{metric}_ci95"]), color=col,
                        marker=mk, ms=4.2, elinewidth=0.9, label=name if m_i == 0 else None)
            ax.text(x, d[f"{metric}_ci95"][1] + 0.012, f"p = {d[f'{metric}_perm_p']:.3f}",
                    fontsize=5.6, color=C["muted"], va="bottom", ha="center")
    hline(ax, mc["nfa_reference"]["chance"])
    ax.set_xticks([0, 1], ["balanced accuracy", "AUROC"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel(f"vs Kosnik acute MEA call (n = {mc['nfa_reference']['n']})")
    ax.legend(loc="upper left", fontsize=5.9, borderaxespad=0.1)
    ax.text(0.99, 0.02, "permutation p; dashed line: chance", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.6, color=C["muted"])
    ptitle(ax, "b", "Acute neuroactivity (Kosnik), unpaired")
    # c: correlation / agreement statistics
    ax = fig.add_subplot(gb[0])
    po = X2["primary_overlap"]["activity_nfa_vs_harrill_morph"]
    fam = X2["primary_by_family_spearman"]
    tw = X2["twin_k3_vs_harrill_primary_overlap"]["activity"]
    kp = mc["potency_spearman_both_active"]
    fam_names = {"hN2_NOG": "hN2 neurite outgrowth", "Cortical_NOG": "cortical neurite outgrowth",
                 "Cortical_Synaptogenesis": "cortical synaptogenesis"}
    rows = [(f"Harrill activity, Cohen's kappa (n = {po['n']})", po["kappa"], po["kappa_ci95"], C["nt"], "o", C["nt"]),
            (f"twin k = 3 vs Harrill activity, kappa (n = {tw['n']})", tw["kappa"], tw["kappa_ci95"], C["nt"], "s", "white"),
            (f"Harrill potency, Spearman, pooled (n = {hd['n_both_active']})", hd["spearman_potency_both_active"],
             hd["spearman_ci95"], C["knn"], "o", C["knn"])]
    for k_, v in fam.items():
        rows.append((f"    {fam_names.get(k_, k_)} (n = {v['n']})", v["spearman"], v["ci95"], C["knn"], "D", "white"))
    rows.append((f"Kosnik acute potency, Spearman (n = {kp['n']})", kp["spearman"], kp["ci95"], C["ref"], "o", C["ref"]))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, ci, col, mk, mfc) in zip(ys, rows):
        ax.errorbar(v, y, xerr=ci_err(v, ci), color=col, marker=mk, ms=4, elinewidth=0.9, mfc=mfc, mew=0.9)
    vline(ax, 0)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=6.3)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("agreement (bootstrap 95% CI); 0 = no association")
    ax.grid(axis="x", color=C["grid"], lw=0.5)
    ptitle(ax, "c", "Concordance with unpaired assays")
    save(fig, "fig9_crossassay")


# ============================================================================= fig10
def fig10_ablations_failures():
    A = J("r9_ablations.json")
    F = J("r10_failures.json")
    S = J("r5_selectivity.json")
    fig = plt.figure(figsize=(W, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.08], wspace=0.32, left=0.09, right=0.985, top=0.95, bottom=0.6)
    gc = fig.add_gridspec(1, 1, left=0.33, right=0.6, top=0.45, bottom=0.08)
    gd = fig.add_gridspec(1, 1, left=0.79, right=0.985, top=0.45, bottom=0.08)
    # a: ablations
    ax = fig.add_subplot(gs[0])
    comps = [("no_interp_decoder_minus_full", "without interpolation-\ninformed decoder", C["interp"], "D"),
             ("no_dose_attention_minus_full", "without dose-local\nattention", C["knn"], "s"),
             ("single_seed_minus_ensemble", "single seed instead\nof 3-seed ensemble", C["muted"], "o")]
    ks = sorted(A["by_k"], key=int)
    for c_i, (key, lab, col, mk) in enumerate(comps):
        for k in ks:
            d = A["by_k"][k][key]
            ax.errorbar(int(k) + (c_i - 1) * 0.22, d["mean_diff"], yerr=ci_err(d["mean_diff"], d["ci95"]), color=col,
                        marker=mk, ms=4, elinewidth=0.9, label=lab if k == ks[0] else None)
    hline(ax, 0)
    ax.set_xticks([int(k) for k in ks])
    ax.set_xlabel("measured concentrations (k)")
    ax.set_ylabel(r"$\Delta$ curve MAE (ablated $-$ full)")
    ax.legend(loc="upper left", fontsize=5.9, labelspacing=0.5)
    ax.set_ylim(min(-0.02, min(A["by_k"][k][c[0]]["ci95"][0] for k in ks for c in comps) - 0.01),
                max(A["by_k"][k][c[0]]["ci95"][1] for k in ks for c in comps) * 1.9)
    ax.text(0.99, 0.02, "above 0: component helps", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9,
            color=C["muted"])
    ptitle(ax, "a", "Ablations (95% CI)")
    # b: per-chemical improvement distribution, k = 3, folds 1-4
    ex, k3, imp = select_examples()
    ax = fig.add_subplot(gs[1])
    bins = np.arange(np.floor(imp.min() * 4) / 4, np.ceil(imp.max() * 4) / 4 + 0.25, 0.1)
    ax.hist(imp[imp >= 0], bins=bins, color=C["nt"], lw=0, label="model better")
    ax.hist(imp[imp < 0], bins=bins, color=C["ref"], lw=0, label="model worse")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    for key, lab in (("median", "median"), ("p10", "10th pct")):
        vline(ax, ex[key]["improvement"], color=C["ink"], ls=":")
        ax.text(ex[key]["improvement"] + (0.04 if key == "median" else -0.04), ax.get_ylim()[1] * 0.98, lab,
                fontsize=5.9, va="top", ha="left" if key == "median" else "right")
    ax.set_xlabel(f"per-chemical gain vs {METHODS[ex['best_baseline']][0].lower()}\n(curve MAE, k = 3, folds 1-4)")
    ax.set_ylabel("chemicals")
    ax.legend(loc="upper left", fontsize=5.9)
    ptitle(ax, "b", f"Where the model loses ({100 * ex['frac_model_worse']:.1f}% of chemicals)")
    # c: failures by class
    ax = fig.add_subplot(gc[0])
    nmin = 5
    cls = [(c, d) for c, d in F["classes"].items() if d["n"] >= nmin]
    cls.sort(key=lambda cd: cd[1]["neurotrajectory_curve_mae"] - cd[1]["baseline_curve_mae"])
    DERIVED["fig10"] = {"class_min_n": nmin, "classes_shown": [c for c, _ in cls],
                        "classes_hidden_n_lt_min": sorted(c for c, d in F["classes"].items() if d["n"] < nmin)}
    ys = np.arange(len(cls))[::-1]
    for y, (c, d) in zip(ys, cls):
        diff = d["neurotrajectory_curve_mae"] - d["baseline_curve_mae"]
        col = C["nt"] if diff < 0 else C["ref"]
        ax.plot([0, diff], [y, y], color=col, lw=1.0)
        ax.plot(diff, y, marker="o" if diff < 0 else "^", color=col, ms=2.2 + 0.9 * np.sqrt(d["n"]), ls="none", mew=0)
    vline(ax, 0, ls="-", color="#333333", lw=0.6)
    ax.set_yticks(ys, [f"{c} (n = {d['n']}, {100 * d['frac_improved']:.0f}% better)" for c, d in cls], fontsize=6.0)
    ax.set_xlabel(r"$\Delta$ mean curve MAE, model $-$ " + METHODS[F["best_baseline"]][0].lower())
    ax.set_ylim(-0.7, len(cls) - 0.3)
    ptitle(ax, "c", f"By annotated class, k = 3 (n $\\geq$ {nmin}; no CI)")
    # d: selectivity (negative)
    ax = fig.add_subplot(gd[0])
    rows = [("selectivity index", S["auroc_selectivity_index"], S["auroc_si_ci95"], C["nt"], "o"),
            ("network potency", S["auroc_network_potency_only"], S["auroc_network_ci95"], C["knn"], "s"),
            ("cytotoxic potency", S["auroc_cytotoxic_potency_only"], S["auroc_cyto_ci95"], C["ref"], "D")]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, ci, col, mk) in zip(ys, rows):
        ax.errorbar(v, y, xerr=ci_err(v, ci), color=col, marker=mk, ms=4.2, elinewidth=0.9)
    vline(ax, 0.5)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=6.3)
    ax.set_xlim(0.3, 0.9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(f"AUROC, DNT label\n({S['n_positive']} pos / {S['n_negative']} neg)")
    ptitle(ax, "d", "Selectivity (negative)")
    save(fig, "fig10_ablations_failures")


# ============================================================================= figA2 (appendix)
def figA2_darwin():
    """Redraws the Darwin-Cage honest-ceiling audit (results/r10_residual_audit.json) in the house
    style; the previous figA2/figA3 were raw copies of run_residual_audit.py's own matplotlib figures."""
    A = J("r10_residual_audit.json")
    fig = plt.figure(figsize=(W, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.1, 1.05], wspace=0.48, left=0.075, right=0.99,
                          top=0.85, bottom=0.25)
    # a: confirmed programs per confirmation fold
    ax = fig.add_subplot(gs[0])
    folds = sorted(A["search"], key=int)
    x = np.arange(len(folds))
    orig = [A["search"][f]["n_pass_original_rule"] for f in folds]
    conf = [A["search"][f]["n_confirmed"] for f in folds]
    ax.bar(x - 0.19, orig, 0.36, color=C["muted"], label="z-rule only")
    ax.bar(x + 0.19, conf, 0.36, color=C["nt"], label="+ sign-flip (used)")
    ax.set_xticks(x, [f"fold {f}" + ("\n(primary)" if int(f) == 4 else "") for f in folds], fontsize=6.1)
    ax.set_ylabel("confirmed programs")
    ax.legend(fontsize=5.6, loc="upper right", borderaxespad=0.1, handlelength=1.2)
    ptitle(ax, "a", "Programs confirmed by fold")
    # b: power curves for the two planted-effect classes
    ax = fig.add_subplot(gs[1])
    eff = A["protocol"]["plant_effects_vehicle_sd"]
    plants = [("plate_shift", "plate|date shift", C["nt"], "o"),
              ("feature_x_div_x_class", "feature x DIV x class", C["ref"], "s")]
    for name, lab, col, mk in plants:
        pw = A["power"][name]
        xe = np.asarray(eff, float) * (0.96 if name == "plate_shift" else 1.04)
        ax.plot(xe, [pw[str(e)]["power_exact"] for e in eff], color=col, marker=mk, ms=4.2, lw=1.3,
                label=f"{lab}: exact recovery")
        ax.plot(xe, [pw[str(e)]["power_oracle"] for e in eff], color=col, marker=mk, mfc="white", ls="--",
                lw=1.0, ms=3.8, label=f"{lab}: oracle program")
    hline(ax, 0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(eff, [str(e) for e in eff])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("planted effect (vehicle SD)")
    ax.set_ylabel("power (of 5 plants)")
    ax.legend(fontsize=5.1, loc="upper left", borderaxespad=0.1, handlelength=1.3)
    ptitle(ax, "b", "Audit power by plant")
    # c: nested curve-MAE change, Darwin program vs generic learner
    ax = fig.add_subplot(gs[2])
    sec = A["secondary_nested_correction"]
    series = [("confirmed Darwin program", sec["darwin_confirmed_correction"]["vs_uncorrected"], C["nt"], "o"),
              ("generic learner (HGB)", sec["baseline_generic_learner_prediction_time_covariates"]["vs_uncorrected"],
               C["muted"], "s")]
    ks = sorted(series[0][1], key=int)
    for i, (lab, d, col, mk) in enumerate(series):
        y = [d[k]["mean_diff"] for k in ks]
        lo = [d[k]["ci95"][0] for k in ks]
        hi = [d[k]["ci95"][1] for k in ks]
        x = np.array([int(k) for k in ks]) + (i - 0.5) * 0.12
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], color=col, marker=mk, ms=4.0,
                    elinewidth=0.9, label=lab)
    hline(ax, 0)
    ax.set_xticks([int(k) for k in ks])
    ax.set_xlabel("measured concentrations (k)")
    ax.set_ylabel(r"$\Delta$ curve MAE vs uncorrected")
    ax.text(0.02, 0.98, "above 0: correction hurts", transform=ax.transAxes, ha="left", va="top", fontsize=5.6,
            color=C["muted"])
    ax.legend(fontsize=5.5, loc="lower right", borderaxespad=0.1)
    ptitle(ax, "c", "Subtracting the learned residual")
    DERIVED["figA2"] = {"n_confirmed_by_fold": {f: A["search"][f]["n_confirmed"] for f in folds},
                        "mde_power0.8_search_exact": A["search"][folds[-1]].get("threshold"),
                        "darwin_vs_generic_k": {k: {"darwin_mean_diff": sec["darwin_confirmed_correction"]
                                                     ["vs_uncorrected"][k]["mean_diff"],
                                                     "generic_mean_diff": sec[
                                                         "baseline_generic_learner_prediction_time_covariates"]
                                                     ["vs_uncorrected"][k]["mean_diff"]} for k in ks}}
    save(fig, "figA2_darwin")


# ============================================================================= figA3 (appendix)
def figA3_viability_voi():
    """Redraws the viability value-of-information audit (results/r10_viability_voi.json). V1/V2 use
    viability read at all doses (a same-plate scenario with leakage w.r.t. the forecast); V3 uses only
    the wells at the measured context concentrations, so it alone is leakage-free."""
    V = J("r10_viability_voi.json")
    fig = plt.figure(figsize=(W, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.4, left=0.09, right=0.99, top=0.85, bottom=0.26)
    ax = fig.add_subplot(gs[0])
    allc = V["all_chemicals"]
    ks = sorted(allc["V0_vs_uncorrected"], key=int)
    arms = [("V0, no viability", allc["V0_vs_uncorrected"], C["nt"], "o", False),
            ("V1, scenario, leakage", allc["V1_vs_uncorrected"], C["ref"], "s", True)]
    for i, (lab, d, col, mk, scen) in enumerate(arms):
        y = [d[k]["mean_diff"] for k in ks]
        lo = [d[k]["ci95"][0] for k in ks]
        hi = [d[k]["ci95"][1] for k in ks]
        x = np.array([int(k) for k in ks]) + (i - 0.5) * 0.14
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], color=col, marker=mk,
                    mfc="white" if scen else col, mew=1.0, ms=4.4, elinewidth=0.9, label=lab)
    hline(ax, 0)
    ax.set_xticks([int(k) for k in ks])
    ax.set_xlabel("measured concentrations (k)")
    ax.set_ylabel(r"$\Delta$ curve MAE vs uncorrected")
    ax.legend(fontsize=5.5, loc="upper right", borderaxespad=0.1)
    ptitle(ax, "a", f"All {allc['n_chemicals']} chemicals")
    ax = fig.add_subplot(gs[1])
    ns = V["ntp_subset"]
    arms2 = [("V0, no viability", ns["V0_prediction_time_vs_uncorrected"], C["nt"], "o", False),
             ("V1, AB/LDH, scenario", ns["V1_pubchem_AB_LDH_calls_vs_uncorrected"], C["ref"], "s", True),
             ("V2, NTP summary, scenario", ns["V2_ntp_chemical_summary_vs_uncorrected"], C["hill"], "D", True),
             ("V3, NTP context wells only", ns["V3_ntp_context_only_vs_uncorrected"], C["knn"], "^", False)]
    ks2 = sorted(arms2[0][1], key=int)
    for i, (lab, d, col, mk, scen) in enumerate(arms2):
        y = [d[k]["mean_diff"] for k in ks2]
        lo = [d[k]["ci95"][0] for k in ks2]
        hi = [d[k]["ci95"][1] for k in ks2]
        x = np.array([int(k) for k in ks2]) + (i - 1.5) * 0.11
        ax.errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], color=col, marker=mk,
                    mfc="white" if scen else col, mew=1.0, ms=4.0, elinewidth=0.85, label=lab)
    hline(ax, 0)
    ax.set_xticks([int(k) for k in ks2])
    ax.set_xlabel("measured concentrations (k)")
    ax.set_ylabel(r"$\Delta$ curve MAE vs uncorrected")
    ax.legend(fontsize=5.0, loc="upper right", ncol=1, borderaxespad=0.1)
    ptitle(ax, "b", f"{ns['n_chemicals']} chemicals with NTP viability")
    DERIVED["figA3"] = {"all_chemicals_n": allc["n_chemicals"], "ntp_subset_n": ns["n_chemicals"],
                        "v1_vs_v0_k3": allc["V1_vs_V0"]["3"], "leakage_caveat": V["leakage_caveat"]}
    save(fig, "figA3_viability_voi")


# ============================================================================= captions
def render_captions():
    spec = importlib.util.spec_from_file_location("render_docs", REPO / "scripts/render_docs.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    src = OUT / "captions.md"
    if not src.exists():
        print("  captions.md not found; skipped")
        return
    errors, cache = [], {}

    def repl(m):
        fname, path, fmt_ = m.group(1), m.group(2), m.group(3)
        try:
            if fname not in cache:
                cache[fname] = json.loads((RES / fname).read_text(encoding="utf-8"))
            return rd.fmt(rd.lookup(cache[fname], path), fmt_)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fname}:{path} -> {e}")
            return "??"

    text = rd.PAT.sub(repl, src.read_text(encoding="utf-8"))
    (OUT / "captions_rendered.md").write_text(text, encoding="utf-8")
    print(f"  rendered captions ({len(rd.PAT.findall(src.read_text(encoding='utf-8')))} references)")
    if errors:
        raise SystemExit("caption references that do not resolve:\n  " + "\n  ".join(errors))


FIGS = [fig1_system, fig2_data, fig3_forecast_main, fig4_examples, fig5_calibration, fig6_hazard,
        fig7_dosecompass, fig8_chip, fig9_crossassay, fig10_ablations_failures, figA2_darwin, figA3_viability_voi]


def main(argv: list[str]):
    want = [f for f in FIGS if not argv or any(f.__name__ == a or f.__name__.startswith(a + "_") for a in argv)]
    dpath = RES / "figures_derived.json"
    prev = json.loads(dpath.read_text(encoding="utf-8")) if dpath.exists() else {}
    for f in want:
        print(f.__name__)
        f()
    derived = {k: v for k, v in prev.items() if k not in ("generated_by", "inputs_sha256")}
    derived.update(DERIVED)
    inputs = sorted(set(prev.get("inputs_sha256", {})) | USED | {"../data_bundle/nfa_tasks.npz"})
    derived["inputs_sha256"] = {n: hashlib.sha256((RES / n).read_bytes()).hexdigest() for n in inputs
                                if (RES / n).exists()}
    derived["generated_by"] = "scripts/make_figures.py"
    dpath.write_text(json.dumps(derived, indent=1, sort_keys=True), encoding="utf-8")
    render_captions()


if __name__ == "__main__":
    main(sys.argv[1:])
