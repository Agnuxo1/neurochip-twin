"""Presentation-grade visuals for NeuroChip Twin v2 (graphical abstract + representative figures).

    python repo/scripts/make_visuals.py                 # all four visuals
    python repo/scripts/make_visuals.py graphical_abstract twin_in_action

Every number drawn comes from repo/results/*.json or from data/processed/*; nothing is
hand-typed. Outputs go to repo/report/figures/visuals/<stem>.pdf, <stem>.png (report width,
6.5 in) and <stem>_16x9.png (1920x1080, for the video/Writeup).

Sources:
  - graphical_abstract: repo/results/trajectory_cv.json, conformal_r7.json, dnt_r4.json,
    r11_early_exit.json, potency_r3.json
  - real_network_raster: data/processed/brewer/wells.parquet (Brewer 4-compartment
    hippocampal MEA, condition NoStim = unstimulated), electrode -> compartment from the
    'subregion' column written by repo/src/neurotwin/data/brewer.py
  - network_development: data/processed/epa_nfa/tasks_cache.pkl (ChemTask per chemical,
    vehicle-SD units; repo/src/neurotwin/models/trajectory.py)
  - twin_in_action: the v1 cross-validated NeuroTrajectory models
    (data/processed/models/cnp_fold<f>_seed*.pt) via neurotwin.models.cnp.load_fold_models,
    run on the chemical's own held-out fold with a k=3 context design taken from
    repo/scripts/run_trajectory_cv.py:designs() (CPU only).
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
RES = REPO / "results"
OUT = REPO / "report" / "figures" / "visuals"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))

W = 6.5          # report figure width, inches (matches make_figures.py)
VIDEO_W_PX, VIDEO_H_PX = 1920, 1080
VIDEO_DPI = 160  # -> figsize (12, 6.75) exactly renders to 1920x1080

# --------------------------------------------------------------------------- palette / style
# Import the validated Okabe-Ito palette + typography from make_figures.py so every visual
# matches the report figures exactly. make_figures.py has no module-level side effects
# (figures are only drawn inside main()), so importing it is safe.
try:
    spec = importlib.util.spec_from_file_location("make_figures", REPO / "scripts" / "make_figures.py")
    MF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MF)
    C, STYLE, FEATURES_LBL = MF.C, dict(MF.STYLE), MF.FEATURE_LABELS
except Exception as exc:  # pragma: no cover - fallback keeps this script independently runnable
    print(f"  (could not import make_figures.py palette, using copied values: {exc})")
    C = {"nt": "#0072B2", "interp": "#E69F00", "knn": "#009E73", "ref": "#D55E00",
         "ctx": "#56B4E9", "hill": "#CC79A7", "zero": "#8C8C8C", "ink": "#1A1A1A",
         "muted": "#5E5E5E", "grid": "#D9D9D9", "band": "#9CC3E6"}
    STYLE = {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333", "axes.labelcolor": "#1A1A1A", "text.color": "#1A1A1A",
        "xtick.color": "#333333", "ytick.color": "#333333", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "figure.dpi": 100, "savefig.dpi": 200,
    }
    FEATURES_LBL = {}
# extra colours only needed for these visuals (kept close in hue family to the imported palette)
C = dict(C, ec="#0072B2", dg="#E69F00", ca3="#009E73", ca1="#D55E00",
         hazard_lo="#009E73", hazard_mid="#E69F00", hazard_hi="#D55E00", box="#F2F4F7", boxedge="#B9C2CE")
plt.rcParams.update(STYLE)
BASE_FONT = STYLE["font.size"]

DERIVED: dict[str, object] = {}


def J(name: str):
    return json.loads((RES / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- rendering helpers
def save_report(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_video(fig, stem: str):
    fig.savefig(OUT / f"{stem}_16x9.png", dpi=VIDEO_DPI, bbox_inches=None)
    plt.close(fig)


def render(build, stem: str, report_wh: tuple[float, float], video_scale: float | None = None):
    """Call build(fig, scale) twice: once at report size, once at 1920x1080 for video."""
    w, h = report_wh
    fig = plt.figure(figsize=(w, h))
    build(fig, 1.0)
    save_report(fig, stem)

    vscale = video_scale if video_scale is not None else (VIDEO_W_PX / VIDEO_DPI) / w
    fig = plt.figure(figsize=(VIDEO_W_PX / VIDEO_DPI, VIDEO_H_PX / VIDEO_DPI))
    build(fig, vscale)
    save_video(fig, stem)
    print(f"  {stem}: .pdf, .png, _16x9.png")


def fs(x, scale):
    return x * scale


def rbox(ax, xy, w, h, fc, ec_, lw=1.0, alpha=1.0, rounding=0.06, z=2, shadow=False, sc=1.0):
    if shadow:
        sh = FancyBboxPatch((xy[0] + 0.35 * sc, xy[1] - 0.35 * sc), w, h,
                            boxstyle=f"round,pad=0,rounding_size={rounding}",
                            fc="#00000018", ec="none", zorder=z - 0.5)
        ax.add_patch(sh)
    b = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                        fc=fc, ec=ec_, lw=lw, alpha=alpha, zorder=z, mutation_aspect=1)
    ax.add_patch(b)
    return b


def arrow(ax, p0, p1, color=None, lw=1.6, z=3, shrinkA=2, shrinkB=2, mutation_scale=10):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale, lw=lw,
                        color=color or C["muted"], zorder=z, shrinkA=shrinkA, shrinkB=shrinkB)
    ax.add_patch(a)
    return a


# ============================================================================ 1. graphical abstract
def graphical_abstract():
    traj = J("trajectory_cv.json")
    conf = J("conformal_r7.json")
    dnt = J("dnt_r4.json")
    exit_ = J("r11_early_exit.json")
    pot = J("potency_r3.json")

    rel_err = traj["by_k"]["3"]["curve_mae"]["paired_vs_best"]["rel_change_pct"]
    coverage = conf["results"]["alpha_0.1"]["k3"]["conformal"]["coverage"]
    v = dnt["variants"]["k3"]["mcnemar_vs_epa"]["DIV12"]
    tp_ours, tp_epa = v["ours"]["tp"], v["epa_rule_same_chemicals"]["tp"]
    exit_rate = exit_["methods"]["persistence"]["overall_exit_rate"]
    wells_saved = pot["by_k"]["3"]["wells_saved_median_pct"]

    DERIVED["graphical_abstract"] = {
        "rel_curve_mae_change_pct_k3": rel_err, "conformal_coverage_alpha0.1_k3": coverage,
        "dnt_tp_ours_DIV12_k3": tp_ours, "dnt_tp_epa_rule_DIV12": tp_epa,
        "early_exit_overall_rate": exit_rate, "wells_saved_median_pct_k3": wells_saved,
    }

    def build(fig, sc):
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 37)
        ax.axis("off")
        # key numbers are read at a glance in the 16:9 video crop, so bump them up a notch there
        big_boost = 1.18 if sc > 1.3 else 1.0

        ax.text(1, 35.3, "NeuroChip Twin v2", fontsize=fs(11, sc), fontweight="bold", color=C["ink"])
        ax.text(1, 33.2, "few-shot forecasting of neural network development on a chip",
                 fontsize=fs(6.6, sc), color=C["muted"])

        doses = [("C1", C["ctx"]), ("C2", C["nt"]), ("C3", C["ref"])]

        # --- stage 1: chip
        rbox(ax, (2, 7.5), 16, 22.5, C["box"], C["boxedge"], lw=fs(1.0, sc), shadow=True, sc=sc)
        ax.text(10, 28.7, "MEA chip", ha="center", fontsize=fs(BASE_FONT, sc), fontweight="bold", color=C["ink"])
        grid_x = np.linspace(4.5, 15.5, 5)
        grid_y = np.linspace(13.5, 25.2, 4)
        for gy in grid_y:
            for gx in grid_x:
                ax.add_patch(Circle((gx, gy), 0.45, fc="white", ec=C["muted"], lw=fs(0.5, sc), zorder=3))
        for i, (lab, col) in enumerate(doses):
            x = 5.5 + i * 4.5
            ax.add_patch(Circle((x, 10.6), 1.05, fc=col, ec="white", lw=fs(0.8, sc), zorder=4))
            ax.text(x, 10.6, lab, ha="center", va="center", fontsize=fs(5.6, sc), color="white",
                     fontweight="bold", zorder=5)
        ax.text(10, 8.3, "3 measured concentrations", ha="center", fontsize=fs(6.0, sc), color=C["muted"])

        # --- arrow 1
        arrow(ax, (18.3, 18.7), (23.5, 18.7), color=C["muted"], lw=fs(1.6, sc))

        # --- stage 2: NeuroTrajectory twin -- shows the few-shot idea directly: 3 measured
        # doses (dots) condition a smooth forecast curve whose uncertainty band is narrow at
        # the measured doses and widens over the unmeasured dose range.
        rbox(ax, (24, 7.5), 25, 22.5, "#EAF2FA", C["nt"], lw=fs(1.3, sc), shadow=True, sc=sc)
        ax.text(36.5, 28.7, "NeuroTrajectory", ha="center", fontsize=fs(BASE_FONT, sc),
                 fontweight="bold", color=C["nt"])
        ax.text(36.5, 26.2, "few-shot twin (CNP): 3 doses in, full curve out",
                 ha="center", fontsize=fs(5.7, sc), color=C["muted"])

        px0, px1, py0, py_top = 27, 46, 10.2, 22.5
        meas_x = np.array([30.5, 36.5, 42.5])
        meas_dose = np.array([-1.0, 0.3, 1.3])          # illustrative log-dose positions
        meas_y = np.array([1.0, 5.6, 7.6]) + py0

        def curve_y(xd):
            return py0 + 0.7 + 7.6 / (1 + np.exp(-(xd - 0.15) / 0.55))

        xs_all = np.linspace(px0 + 1, px1 - 1, 60)
        ys_all = curve_y(np.interp(xs_all, meas_x, meas_dose, left=meas_dose[0] - (meas_x[0] - xs_all.min()) / 3,
                                    right=meas_dose[-1] + (xs_all.max() - meas_x[-1]) / 3))
        dist = np.min(np.abs(xs_all[:, None] - meas_x[None, :]), axis=1)
        halfw = 0.35 + 0.55 * (dist / dist.max())
        ax.plot([px0, px1], [py0, py0], color=C["muted"], lw=fs(0.8, sc), zorder=3)
        ax.fill_between(xs_all, ys_all - halfw * 3, ys_all + halfw * 3, color=C["band"], alpha=0.55,
                         zorder=3, lw=0)
        ax.plot(xs_all, ys_all, color=C["nt"], lw=fs(1.6, sc), zorder=4)
        for (mx, my), (lab, col) in zip(zip(meas_x, meas_y), doses):
            ax.add_patch(Circle((mx, my), 0.75, fc=col, ec="white", lw=fs(0.7, sc), zorder=5))
        ax.text((px0 + px1) / 2, py0 - 1.6, "dose ->", ha="center", fontsize=fs(5.6, sc), color=C["muted"])
        ax.text(px0 - 0.6, (py0 + py_top) / 2, "response", ha="center", va="center", rotation=90,
                 fontsize=fs(5.6, sc), color=C["muted"])

        # --- arrow 2
        arrow(ax, (49.3, 18.7), (54.5, 18.7), color=C["muted"], lw=fs(1.6, sc))

        # --- stage 3: outputs (2x2), tighter grid, less dead space
        out_boxes = [
            (55, 19.3, "Dose x day forecast", "band = uncertainty"),
            (77.5, 19.3, "Hazard call", "safe / caution / hazard"),
            (55, 7.5, "Next dose", "DoseCompass"),
            (77.5, 7.5, "Stop-at-DIV7 gate", "throughput gate"),
        ]
        bw, bh = 21, 10.7
        for x, y, title, sub in out_boxes:
            rbox(ax, (x, y), bw, bh, "white", C["boxedge"], lw=fs(1.0, sc), shadow=True, sc=sc)
            ax.text(x + bw / 2, y + bh - 2.3, title, ha="center", fontsize=fs(6.6, sc),
                     fontweight="bold", color=C["ink"])
            ax.text(x + bw / 2, y + 0.85, sub, ha="center", va="bottom", fontsize=fs(5.4, sc), color=C["muted"])

        # forecast mini heatmap
        hm = np.array([[0.1, 0.3, 0.6, 0.9], [0.2, 0.5, 0.8, 1.0], [0.4, 0.7, 0.95, 1.0]])
        x0, y0, hw, hh = 58.5, 26.9, 13, 4.1
        nr, nc = hm.shape
        for r in range(nr):
            for c in range(nc):
                cx = x0 + c * hw / nc
                cy = y0 - (r + 1) * hh / nr
                col = plt.cm.Blues(0.25 + 0.65 * hm[r, c])
                ax.add_patch(Rectangle((cx, cy), hw / nc * 0.88, hh / nr * 0.78, fc=col, ec="white",
                                       lw=fs(0.4, sc), zorder=4))

        # hazard call: traffic-light rings
        cx, cy = 88, 24.6
        for i, (col, r) in enumerate(zip([C["hazard_lo"], C["hazard_mid"], C["hazard_hi"]], [2.3, 1.55, 0.8])):
            ax.add_patch(Circle((cx, cy), r, fc=col if r == 0.8 else "none",
                                ec=col, lw=0 if r == 0.8 else fs(1.6, sc), zorder=4 + i))

        # DoseCompass: arrow to next concentration on an axis
        ax0x, ax0y = 58, 11.6
        ax.plot([ax0x, ax0x + 15], [ax0y, ax0y], color=C["muted"], lw=fs(1.0, sc), zorder=4)
        for xx_ in np.linspace(ax0x + 1, ax0x + 14, 5):
            ax.plot([xx_, xx_], [ax0y - 0.4, ax0y + 0.4], color=C["muted"], lw=fs(0.8, sc), zorder=4)
        arrow(ax, (ax0x + 4.5, ax0y + 2.4), (ax0x + 11, ax0y + 0.5), color=C["nt"], lw=fs(1.6, sc))

        # stop gate: DIV7 flag (offset from box centre so it doesn't cross the caption text)
        gx, gy = 85, 11.1
        ax.plot([gx, gx], [gy - 2.2, gy + 2.2], color=C["muted"], lw=fs(1.2, sc), zorder=4)
        ax.add_patch(Rectangle((gx, gy + 0.35), 4.4, 1.9, fc=C["hazard_mid"], ec="none", zorder=4))
        ax.text(gx + 2.2, gy + 1.3, "DIV 7", ha="center", va="center", fontsize=fs(5.6, sc),
                 color="white", fontweight="bold", zorder=5)

        # --- bottom key-number strip
        ax.plot([2, 98], [5.6, 5.6], color=C["grid"], lw=fs(0.8, sc), zorder=1)
        stats = [
            (f"{abs(rel_err):.0f}%", "lower forecast error at k=3"),
            (f"{100*coverage:.1f}%", "conformal coverage (target 90%)"),
            (f"{tp_ours} vs {tp_epa}", "neurotoxicants flagged, ours vs EPA rule"),
            (f"{100*exit_rate:.0f}%", "chemicals stop at DIV 7 (throughput gate)"),
            (f"{wells_saved:.0f}%", "median wells saved (k=3)"),
        ]
        xs = np.linspace(9, 91, len(stats))
        for x, (big, small) in zip(xs, stats):
            ax.text(x, 3.8, big, ha="center", fontsize=fs(9.5, sc) * big_boost, fontweight="bold", color=C["nt"])
            ax.text(x, 1.1, small, ha="center", va="top", fontsize=fs(5.0, sc), color=C["muted"], wrap=True)

    render(build, "graphical_abstract", (W, W * 9 / 16 * 1.15))


# ============================================================================ 2. real network raster
def real_network_raster():
    wells = pd.read_parquet(ROOT / "data/processed/brewer/wells.parquet")
    condition, fid = "NoStim", 1
    sub = wells[(wells.condition == condition) & (wells.fid == fid)].copy()
    order = ["EC", "DG", "CA3", "CA1"]
    colors = {"EC": C["ec"], "DG": C["dg"], "CA3": C["ca3"], "CA1": C["ca1"]}
    sub["region_rank"] = sub.subregion.map({r: i for i, r in enumerate(order)})
    sub = sub.sort_values(["region_rank", "electrode"]).reset_index(drop=True)

    t0, t1 = 60.0, 90.0  # 30 s unstimulated window
    rows, ys, cols_, rate_all = [], [], [], []
    for i, r in sub.iterrows():
        st = np.asarray(r["spike_times_s"], float)
        st = st[(st >= t0) & (st < t1)]
        rows.append(st)
        ys.append(i)
        cols_.append(colors[r["subregion"]])
        rate_all.append(st)
    bins = np.arange(t0, t1 + 0.1, 0.1)
    pop_rate, _ = np.histogram(np.concatenate(rate_all) if len(rate_all) else np.array([]), bins=bins)
    pop_rate = pop_rate / (0.1 * len(sub))  # Hz per electrode
    n_e = len(sub)
    DERIVED["real_network_raster"] = {"condition": condition, "fid": int(fid), "window_s": [t0, t1],
                                        "n_electrodes": int(n_e),
                                        "n_spikes_window": int(sum(len(s) for s in rows))}

    def build(fig, sc):
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4.3, 1.9],
                              hspace=0.08, wspace=0.32, left=0.09, right=0.98, top=0.90, bottom=0.11)
        ax_rate = fig.add_subplot(gs[0, 0])
        ax_rate.plot(bins[:-1] - t0, pop_rate, color=C["ink"], lw=fs(0.9, sc))
        ax_rate.fill_between(bins[:-1] - t0, 0, pop_rate, color=C["muted"], alpha=0.25, lw=0)
        ax_rate.set_xlim(0, t1 - t0)
        ax_rate.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)
        ax_rate.set_ylabel("pop. rate\n(Hz/electrode)", fontsize=fs(6.4, sc))
        ax_rate.set_title("$\\bf{a}$   Representative 30 s recording, Brewer 4-compartment MEA "
                          f"(NoStim, FID {fid})", loc="left", fontsize=fs(8, sc))
        for s in ("top", "right"):
            ax_rate.spines[s].set_visible(False)
        ax_rate.tick_params(labelsize=fs(6, sc))

        ax = fig.add_subplot(gs[1, 0], sharex=ax_rate)
        for y, st, col in zip(ys, rows, cols_):
            if len(st):
                ax.vlines(st - t0, y - 0.42, y + 0.42, color=col, lw=fs(0.6, sc))
        boundaries = np.cumsum([np.sum(sub.subregion == r) for r in order])
        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color="white", lw=fs(1.6, sc), zorder=5)
        ax.set_ylim(-1, n_e)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0, t1 - t0 + 1, 5))
        ax.set_xlabel(f"time relative to window start (s)  --  absolute window {t0:.0f}-{t1:.0f} s",
                     fontsize=fs(6.8, sc))
        ax.set_ylabel("electrode (grouped by compartment)", fontsize=fs(6.8, sc))
        ax.tick_params(labelsize=fs(6, sc))
        yt, ytl = [], []
        prev = 0
        for r, b in zip(order, boundaries):
            yt.append((prev + b - 1) / 2)
            ytl.append(r)
            prev = b
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(yt)
        ax2.set_yticklabels(ytl, fontsize=fs(6.8, sc), fontweight="bold")
        for tick, r in zip(ax2.get_yticklabels(), order):
            tick.set_color(colors[r])
        ax2.tick_params(length=0)
        ax2.spines[:].set_visible(False)

        # inset: chip layout schematic
        axi = fig.add_subplot(gs[1, 1])
        axi.set_xlim(0, 10)
        axi.set_ylim(-1.6, 10)
        # equal aspect: the column is much narrower than tall, so without this the "circular"
        # nodes render as tall ellipses and a uniform point-based arrow shrink clears them in
        # one direction but not the other (arrowheads silently hidden inside the node)
        axi.set_aspect("equal", adjustable="box")
        axi.axis("off")
        axi.set_title("chip layout", fontsize=fs(6.8, sc), loc="center", pad=fs(3, sc))
        # route CA1 back near EC to show the closed 4-compartment loop (EC->DG->CA3->CA1->EC)
        centers = {"EC": (2, 8.6), "DG": (7.5, 8.6), "CA3": (7.5, 2.2), "CA1": (2, 2.2)}
        path = ["EC", "DG", "CA3", "CA1", "EC"]
        node_r = 1.15
        for a, b in zip(path[:-1], path[1:]):
            # shrink well past the node radius (in points) so the arrowhead clears the circle
            # and the feed-forward direction EC->DG->CA3->CA1->EC is actually visible
            arrow(axi, centers[a], centers[b], color=C["muted"], lw=fs(1.1, sc),
                  shrinkA=fs(13, sc), shrinkB=fs(13, sc), mutation_scale=fs(11, sc))
        for r, (x, y) in centers.items():
            axi.add_patch(Circle((x, y), node_r, fc=colors[r], ec="white", lw=fs(0.8, sc), zorder=4))
            axi.text(x, y, r, ha="center", va="center", fontsize=fs(5.6, sc), color="white",
                     fontweight="bold", zorder=5)
        axi.text(4.75, -1.0, "single-axon tunnels\n(feed-forward)", ha="center", fontsize=fs(5.2, sc),
                 color=C["muted"])

    render(build, "real_network_raster", (W, W * 0.62))


# ============================================================================ 3. network development
def network_development():
    from neurotwin.models.trajectory import DIVS, FEATURES

    tasks, tr = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))

    # Selection rule (explicit, deterministic): among EPA reference chemicals labelled
    # "positive" whose chemical fold is 1-4 (not the fold-0 sanity split), take the one with the
    # single largest |mean level effect| (vehicle-SD units) over (DIV, feature, concentration
    # level), restricted to level means that do NOT sit on the +-10 SD hard clip (clip ties would
    # make the "largest effect" pick arbitrary/saturated rather than informative).
    candidates = [t for t in tasks if t.label == "positive" and t.fold in (1, 2, 3, 4)]
    best_chem, best_val, best_fold = None, -1.0, None
    for t in candidates:
        vals = np.abs(t._mu)[t._ok]
        vals = vals[vals < 9.99]
        if len(vals) == 0:
            continue
        mx = float(vals.max())
        if mx > best_val:
            best_val, best_chem, best_fold, best_task = mx, t.chem, t.fold, t

    t = best_task
    lv = t.levels  # ascending log10 uM
    # top effect features for this chemical
    eff = np.zeros(len(FEATURES))
    for f in range(len(FEATURES)):
        vals = np.abs(t._mu[:, :, f])[t._ok[:, :, f]]
        vals = vals[vals < 9.99]
        eff[f] = vals.max() if len(vals) else 0
    top = list(np.argsort(-eff)[:6])
    DERIVED["network_development"] = {"selection_rule": "positive reference chemical, fold in 1-4, "
                                       "largest |mean level effect| below the +-10 SD clip",
                                       "chemical": best_chem, "fold": int(best_fold),
                                       "n_levels": int(len(lv)),
                                       "features_shown": [FEATURES[i] for i in top]}

    cmap = plt.cm.viridis
    norm = plt.Normalize(lv.min(), lv.max())

    def build(fig, sc):
        gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.32, left=0.08, right=0.86, top=0.86, bottom=0.10)
        fig.suptitle(f"Developmental trajectory, {best_chem} (EPA NFA, fold {best_fold}, "
                     f"{len(lv)} tested concentrations)", fontsize=fs(8.2, sc), x=0.06, ha="left",
                     fontweight="bold")
        letters = "abcdef"
        for k, fidx in enumerate(top):
            ax = fig.add_subplot(gs[k // 3, k % 3])
            band = ax.axhspan(-1, 1, color=C["grid"], alpha=0.6, lw=0, zorder=1,
                              label="vehicle band ($\\pm$1 SD)" if k == 0 else None)
            for li, level in enumerate(lv):
                y = t._mu[li, :, fidx]
                ok = t._ok[li, :, fidx]
                col = cmap(norm(level))
                if ok.any():
                    xs = np.array(DIVS)[ok]
                    ax.plot(xs, y[ok], "-o", color=col, lw=fs(1.1, sc), ms=fs(2.6, sc), zorder=3)
            ax.set_title(f"$\\bf{{{letters[k]}}}$   {FEATURES_LBL.get(FEATURES[fidx], FEATURES[fidx])}",
                        loc="left", fontsize=fs(7.2, sc))
            ax.set_xticks(DIVS)
            ax.tick_params(labelsize=fs(6.2, sc))
            ax.set_xlabel("DIV", fontsize=fs(6.4, sc))
            if k % 3 == 0:
                ax.set_ylabel("effect (vehicle SD)", fontsize=fs(6.4, sc))
            ax.axhline(0, color=C["muted"], lw=fs(0.5, sc), ls="--", zorder=1)
            if k == 0:
                # legend lives on panel a itself, right next to the data it explains, rather
                # than floating above the (unrelated) concentration colorbar
                ax.legend(handles=[band], loc="upper right", fontsize=fs(5.0, sc), frameon=False,
                         handlelength=1.2, handleheight=0.9, borderaxespad=0.2)

        cax = fig.add_axes((0.89, 0.15, 0.018, 0.65))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("log$_{10}$ concentration (µM)", fontsize=fs(6.4, sc))
        cb.ax.tick_params(labelsize=fs(5.8, sc))

    render(build, "network_development", (W, W * 0.62))


# ============================================================================ 4. twin in action
def twin_in_action():
    from neurotwin.models.trajectory import FEATURES, DIVS, split_context
    from neurotwin.models.cnp import load_fold_models, predict

    spec = importlib.util.spec_from_file_location("run_trajectory_cv", REPO / "scripts" / "run_trajectory_cv.py")
    rtc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtc)

    tasks, tr = pickle.load(open(ROOT / "data/processed/epa_nfa/tasks_cache.pkl", "rb"))
    # same chemical/fold as network_development, chosen there by the documented rule; re-derive
    # it here from the same deterministic rule so this script's figures stay independent to run.
    candidates = [x for x in tasks if x.label == "positive" and x.fold in (1, 2, 3, 4)]
    best_chem, best_val, task = None, -1.0, None
    for x in candidates:
        vals = np.abs(x._mu)[x._ok]
        vals = vals[vals < 9.99]
        if len(vals) == 0:
            continue
        mx = float(vals.max())
        if mx > best_val:
            best_val, best_chem, task = mx, x.chem, x

    fidx = int(np.argmax([np.abs(task._mu[:, :, f])[task._ok[:, :, f]][
        np.abs(task._mu[:, :, f])[task._ok[:, :, f]] < 9.99].max()
        if (task._ok[:, :, f]).any() else 0 for f in range(len(FEATURES))]))
    feat_name = FEATURES[fidx]

    ctx_lv = rtc.designs(task, 3)[0]  # k=3 design, seed 0, same protocol as run_trajectory_cv.py
    ic, it = split_context(task, ctx_lv)
    models, cfg = load_fold_models(task.fold, "cpu", directory=str(ROOT / "data/processed/models"))
    preds = [predict(m, task, ic, task.levels, device="cpu") for m in models]
    mu = np.mean([p[0] for p in preds], axis=0)          # (L, ND, NF) ensemble mean
    sig = np.sqrt(np.mean([p[1] ** 2 for p in preds], axis=0) + np.var([p[0] for p in preds], axis=0))

    lv = task.levels
    meas = np.where(task._ok[:, :, fidx], task._mu[:, :, fidx], np.nan)
    pred = mu[:, :, fidx]
    unc = sig[:, :, fidx]
    ctx_idx = [int(np.where(lv == c)[0][0]) for c in ctx_lv]

    DERIVED["twin_in_action"] = {"chemical": best_chem, "fold": int(task.fold), "feature": feat_name,
                                  "k": 3, "context_log10_conc_uM": [round(float(c), 3) for c in ctx_lv]}

    # symmetric limits around 0 so a TwoSlopeNorm puts white exactly at "no effect vs vehicle";
    # panels a and b share this one norm so their colours are directly comparable
    vmax_sym = float(np.nanmax(np.abs(np.concatenate([meas[~np.isnan(meas)], pred.ravel()]))))
    vmax_sym = max(vmax_sym, 1e-6)
    norm_ab = matplotlib.colors.TwoSlopeNorm(vmin=-vmax_sym, vcenter=0.0, vmax=vmax_sym)

    def heat(ax, mat, title, sc, show_ctx=True, cmap="RdBu_r", norm=None, vmin_=None, vmax_=None):
        im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=cmap, norm=norm, vmin=vmin_, vmax=vmax_,
                       extent=(-0.5, len(lv) - 0.5, -0.5, len(DIVS) - 0.5))
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([f"{v:.1f}" for v in lv], fontsize=fs(5.8, sc), rotation=0)
        ax.set_yticks(range(len(DIVS)))
        ax.set_yticklabels(DIVS, fontsize=fs(6.2, sc))
        ax.set_xlabel("log$_{10}$ concentration (µM)", fontsize=fs(6.4, sc))
        ax.set_ylabel("DIV", fontsize=fs(6.4, sc))
        ax.set_title(title, fontsize=fs(7.4, sc), loc="left")
        if show_ctx:
            for ci in ctx_idx:
                ax.add_patch(Rectangle((ci - 0.5, -0.5), 1, len(DIVS), fill=False,
                                       ec=C["ink"], lw=fs(1.6, sc), zorder=5))
        return im

    def build(fig, sc):
        gs = fig.add_gridspec(1, 3, wspace=0.75, left=0.07, right=0.95, top=0.80, bottom=0.20)
        fig.suptitle(f"Twin in action - {best_chem} (fold {task.fold} held out), "
                     f"{FEATURES_LBL.get(feat_name, feat_name)}, k=3 measured context (boxed)",
                     fontsize=fs(8.2, sc), x=0.06, ha="left", fontweight="bold")

        ax1 = fig.add_subplot(gs[0, 0])
        heat(ax1, meas, "$\\bf{a}$   Measured", sc, cmap="RdBu_r", norm=norm_ab)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = heat(ax2, pred, "$\\bf{b}$   Twin forecast", sc, cmap="RdBu_r", norm=norm_ab)
        # one shared colorbar for a+b: same norm/cmap, so white = no effect vs vehicle in both
        cb = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.06, location="right")
        cb.set_label("effect (vehicle SD, clipped at $\\pm$10)", fontsize=fs(6.0, sc))
        cb.ax.tick_params(labelsize=fs(5.6, sc))

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = heat(ax3, unc, "$\\bf{c}$   Uncertainty ($\\sigma$)", sc, cmap="Purples", vmin_=0,
                  vmax_=float(np.nanmax(unc)))
        cb2 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.06)
        cb2.set_label("$\\sigma$ (vehicle SD)", fontsize=fs(6.0, sc))
        cb2.ax.tick_params(labelsize=fs(5.6, sc))

    render(build, "twin_in_action", (W, W * 0.42))


# ============================================================================ main
ALL = {"graphical_abstract": graphical_abstract, "real_network_raster": real_network_raster,
       "network_development": network_development, "twin_in_action": twin_in_action}


def main(argv):
    want = argv or list(ALL)
    for name in want:
        print(name)
        ALL[name]()
    if DERIVED:
        (OUT / "visuals_derived.json").write_text(json.dumps(DERIVED, indent=1, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
