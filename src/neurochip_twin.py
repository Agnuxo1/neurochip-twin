"""Reproducible interpretable temporal phenotype benchmark.

The benchmark is intentionally self-contained: no network, API key, or hidden
dataset is required. The generated image sequences are a stress-test proxy for
organ-on-chip microscopy, not a biological claim.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "count_ratio",
    "area_ratio",
    "intensity_ratio",
    "elongation_ratio",
    "motility",
    "persistence",
    "count_slope",
    "area_slope",
    "intensity_slope",
]


@dataclass(frozen=True)
class Sequence:
    frames: np.ndarray
    dose: float
    label: int
    seed: int


def _draw_gaussian(image: np.ndarray, cy: float, cx: float, sy: float, sx: float, amp: float) -> None:
    h, w = image.shape
    y0, y1 = max(0, int(cy - 4 * sy)), min(h, int(cy + 4 * sy + 1))
    x0, x1 = max(0, int(cx - 4 * sx)), min(w, int(cx + 4 * sx + 1))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    image[y0:y1, x0:x1] += amp * np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


def generate_sequence(seed: int, dose: float, *, frames: int = 12, shape: tuple[int, int] = (128, 160)) -> Sequence:
    """Generate an organ-on-chip-like cell movie with a known perturbation."""
    rng = np.random.default_rng(seed)
    h, w = shape
    n_cells = int(rng.integers(14, 24))
    y = rng.uniform(18, h - 18, n_cells)
    x = rng.uniform(18, w - 18, n_cells)
    sy = rng.uniform(2.0, 4.3, n_cells)
    sx = sy * rng.uniform(0.8, 1.8, n_cells)
    drift = rng.normal(0, 0.8, (n_cells, 2))
    toxicity = np.clip(dose * 1.35 + rng.normal(0, 0.08), 0, 1)
    label = int(toxicity > 0.42)
    out = np.zeros((frames, h, w), dtype=np.float32)
    for t in range(frames):
        image = rng.normal(0.025, 0.012, shape).astype(np.float32)
        vitality = np.clip(1.0 - toxicity * (t / max(frames - 1, 1)) * 1.20, 0.12, 1.0)
        for i in range(n_cells):
            cy = y[i] + drift[i, 0] * t + rng.normal(0, 0.25)
            cx = x[i] + drift[i, 1] * t + rng.normal(0, 0.25)
            if rng.random() < toxicity * (t / max(frames, 1)) * 0.42:
                continue
            amp = (0.42 + rng.uniform(0.0, 0.2)) * vitality
            elongation = 1.0 + toxicity * 1.8 * (t / max(frames - 1, 1)) + rng.normal(0, 0.08)
            _draw_gaussian(image, cy, cx, sy[i] * elongation, sx[i] / max(elongation, 0.7), amp)
        if toxicity > 0.3 and t > frames // 2:
            # A faint debris-like background signal makes threshold choice nontrivial.
            image += ndimage.gaussian_filter(rng.random(shape).astype(np.float32), 4) * toxicity * 0.018
        out[t] = np.clip(ndimage.gaussian_filter(image, 0.6), 0, 1)
    return Sequence(out, float(dose), label, seed)


def segment(frame: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Segment bright cells and return a labelled image plus interpretable objects."""
    threshold = max(float(np.quantile(frame, 0.985) * 0.35), 0.10)
    mask = frame > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    labels, count = ndimage.label(mask)
    objects: list[dict[str, float]] = []
    for idx in range(1, count + 1):
        ys, xs = np.where(labels == idx)
        area = len(xs)
        if area < 8 or area > 1200:
            labels[labels == idx] = 0
            continue
        vals = frame[ys, xs]
        cy, cx = float(ys.mean()), float(xs.mean())
        cov = np.cov(np.stack([ys, xs])) if area > 2 else np.eye(2)
        eig = np.linalg.eigvalsh(np.atleast_2d(cov))
        elongation = float(np.sqrt(max(eig[-1], 1e-6) / max(eig[0], 1e-6)))
        objects.append({"label": float(idx), "cy": cy, "cx": cx, "area": float(area), "intensity": float(vals.mean()), "elongation": elongation})
    return labels, objects


def track_objects(detections: list[list[dict[str, float]]], max_distance: float = 16.0) -> list[list[dict[str, float]]]:
    """Associate cells frame-to-frame with a gated Hungarian assignment."""
    next_id = 0
    previous: list[dict[str, float]] = []
    result: list[list[dict[str, float]]] = []
    for current in detections:
        for obj in current:
            obj["track_id"] = -1.0
        if previous and current:
            cost = np.array([[np.hypot(a["cy"] - b["cy"], a["cx"] - b["cx"]) for b in current] for a in previous])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= max_distance:
                    current[c]["track_id"] = previous[r]["track_id"]
        for obj in current:
            if obj["track_id"] < 0:
                obj["track_id"] = float(next_id)
                next_id += 1
        result.append(current)
        previous = current
    return result


def phenotype_features(sequence: Sequence) -> tuple[np.ndarray, pd.DataFrame]:
    detections: list[list[dict[str, float]]] = []
    masks = []
    for frame in sequence.frames:
        mask, objects = segment(frame)
        masks.append(mask)
        detections.append(objects)
    tracks = track_objects(detections)
    rows = []
    for t, objects in enumerate(tracks):
        for obj in objects:
            rows.append({"frame": t, **obj})
    table = pd.DataFrame(rows)
    if table.empty:
        return np.zeros(len(FEATURE_NAMES), dtype=float), table
    grouped = table.groupby("track_id")
    lifetimes = grouped["frame"].nunique()
    motion = grouped.apply(lambda g: np.hypot(np.diff(g["cy"]), np.diff(g["cx"])).mean() if len(g) > 1 else 0.0, include_groups=False)
    def slope(col: str) -> float:
        means = table.groupby("frame")[col].mean()
        return float(np.polyfit(means.index, means.values, 1)[0]) if len(means) > 1 else 0.0
    first = table[table.frame == table.frame.min()]
    last = table[table.frame == table.frame.max()]
    def ratio(col: str) -> float:
        a, b = first[col].mean(), last[col].mean()
        return float(b / max(a, 1e-6))
    features = np.array([
        len(last) / max(len(first), 1), ratio("area"), ratio("intensity"), ratio("elongation"),
        float(motion.mean()), float((lifetimes >= max(2, len(sequence.frames) // 2)).mean()),
        slope("frame") if False else (len(last) - len(first)) / max(len(sequence.frames) - 1, 1),
        slope("area"), slope("intensity"),
    ], dtype=float)
    return features, table


class TemporalReservoir:
    """Small fixed reservoir; only the readout is fitted."""
    def __init__(self, input_dim: int, hidden: int = 32, seed: int = 7, spectral: float = 0.82):
        rng = np.random.default_rng(seed)
        self.W_in = rng.normal(0, 0.45, (hidden, input_dim))
        w = rng.normal(0, 1, (hidden, hidden))
        radius = max(abs(np.linalg.eigvals(w)))
        self.W = (w / radius) * spectral
        self.hidden = hidden

    def transform(self, tables: Iterable[pd.DataFrame]) -> np.ndarray:
        states = []
        for table in tables:
            state = np.zeros(self.hidden)
            if table.empty:
                states.append(state)
                continue
            for _, frame in table.groupby("frame"):
                x = frame[["area", "intensity", "elongation"]].mean().to_numpy(float)
                x = np.pad(x, (0, self.W_in.shape[1] - len(x)))[: self.W_in.shape[1]]
                state = np.tanh(self.W @ state + self.W_in @ x)
            states.append(state)
        return np.asarray(states)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def run(out_dir: Path, seed: int = 42, n_samples: int = 180) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sequences = [generate_sequence(int(rng.integers(1_000_000)), float(rng.uniform(0, 1))) for _ in range(n_samples)]
    X, static_X, y, tables, doses = [], [], [], [], []
    for seq in sequences:
        feat, table = phenotype_features(seq)
        first_objects = table[table["frame"] == table["frame"].min()] if not table.empty else table
        static_X.append([
            float(len(first_objects)),
            float(first_objects["area"].mean()) if not first_objects.empty else 0.0,
            float(first_objects["intensity"].mean()) if not first_objects.empty else 0.0,
            float(first_objects["elongation"].mean()) if not first_objects.empty else 0.0,
        ])
        X.append(feat); y.append(seq.label); tables.append(table); doses.append(seq.dose)
    X, static_X, y, doses = np.asarray(X), np.asarray(static_X), np.asarray(y), np.asarray(doses)
    idx = np.arange(n_samples)
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=seed, stratify=y)
    baseline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed))])
    baseline.fit(static_X[train_idx], y[train_idx])
    p_base = baseline.predict_proba(static_X[test_idx])[:, 1]
    reservoir = TemporalReservoir(input_dim=3, seed=seed + 11)
    R = reservoir.transform(tables)
    fusion = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed))])
    fusion.fit(np.c_[X[train_idx], R[train_idx]], y[train_idx])
    p_temporal = fusion.predict_proba(np.c_[X[test_idx], R[test_idx]])[:, 1]
    result = {
        "seed": seed,
        "n_samples": n_samples,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "data_kind": "synthetic_organ_on_chip_proxy",
        "baseline": metrics(y[test_idx], p_base),
        "temporal_reservoir": metrics(y[test_idx], p_temporal),
        "delta_roc_auc": float(roc_auc_score(y[test_idx], p_temporal) - roc_auc_score(y[test_idx], p_base)),
        "feature_names": FEATURE_NAMES,
        "uncertainty_proxy": "distance from 0.5; not a clinical confidence interval",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pred = pd.DataFrame({"index": test_idx, "dose": doses[test_idx], "label": y[test_idx], "baseline_probability": p_base, "temporal_probability": p_temporal})
    pred.to_csv(out_dir / "predictions.csv", index=False)
    pd.concat([t.assign(sequence=i) for i, t in enumerate(tables)], ignore_index=True).to_csv(out_dir / "phenotype_table.csv", index=False)

    demo = sequences[int(np.argmax(doses))]
    demo_features, demo_table = phenotype_features(demo)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(demo.frames[0], cmap="magma"); axes[0].set_title("t=0 microscopy proxy"); axes[0].axis("off")
    axes[1].imshow(demo.frames[-1], cmap="magma"); axes[1].set_title("t=end, perturbed"); axes[1].axis("off")
    axes[2].hist(doses, bins=12, alpha=0.7, label="dose")
    axes[2].axvline(0.42 / 1.35, color="black", ls="--", label="decision region")
    axes[2].set_title("generated dose distribution"); axes[2].set_xlabel("dose"); axes[2].legend()
    fig.tight_layout(); fig.savefig(out_dir / "demo_overview.png", dpi=160); plt.close(fig)

    cm = confusion_matrix(y[test_idx], (p_temporal >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
    ax.set_title("Temporal model confusion matrix"); ax.set_xlabel("predicted"); ax.set_ylabel("observed")
    for (i, j), value in np.ndenumerate(cm): ax.text(j, i, int(value), ha="center", va="center")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=160); plt.close(fig)

    html = f"""<!doctype html><meta charset='utf-8'><title>NeuroChip Twin demo</title>
    <h1>NeuroChip Twin</h1><p>Self-contained synthetic organ-on-chip proxy benchmark; no clinical claim.</p>
    <p>Temporal ROC-AUC: <b>{result['temporal_reservoir']['roc_auc']:.3f}</b> · Static ROC-AUC: <b>{result['baseline']['roc_auc']:.3f}</b> · Δ={result['delta_roc_auc']:+.3f}</p>
    <img src='demo_overview.png' style='max-width:100%'><img src='confusion_matrix.png' style='max-width:420px'>
    <h2>Interpretation</h2><pre>{json.dumps(dict(zip(FEATURE_NAMES, demo_features.round(4))), indent=2)}</pre>
    <p>Uncertainty is a transparent distance-from-threshold proxy and must not be read as a clinical confidence interval.</p>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=180)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.seed, args.samples), indent=2))


if __name__ == "__main__":
    main()
