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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
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
PHYSICS_FEATURE_NAMES = [
    "dose",
    "flow_rate_uL_min",
    "wall_shear_proxy",
    "clearance_factor",
    "effective_dose",
    "compound_logp_proxy",
    "target_pathway_proxy",
    "compound_sensitivity_proxy",
]

COMPOUND_CONTEXT = np.array([
    [0.15, 0.10, 0.62],
    [0.35, 0.85, 0.78],
    [0.55, 0.25, 0.94],
    [0.75, 0.70, 0.58],
    [0.95, 0.40, 1.08],
    [1.20, 0.90, 0.72],
    [1.45, 0.15, 1.22],
    [1.70, 0.60, 0.88],
], dtype=float)


@dataclass(frozen=True)
class Sequence:
    frames: np.ndarray
    dose: float
    label: int
    seed: int
    flow_rate: float = 0.0
    toxicity_score: float = 0.0
    viability_target: float = 100.0
    ic50_target: float = -6.0
    compound_id: int = 0
    compound_logp: float = 0.0
    target_pathway: float = 0.0
    compound_sensitivity: float = 1.0


def _draw_gaussian(image: np.ndarray, cy: float, cx: float, sy: float, sx: float, amp: float) -> None:
    h, w = image.shape
    y0, y1 = max(0, int(cy - 4 * sy)), min(h, int(cy + 4 * sy + 1))
    x0, x1 = max(0, int(cx - 4 * sx)), min(w, int(cx + 4 * sx + 1))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    image[y0:y1, x0:x1] += amp * np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


def generate_sequence(
    seed: int,
    dose: float,
    *,
    flow_rate: float | None = None,
    frames: int = 12,
    shape: tuple[int, int] = (128, 160),
    scenario: str = "exposure_only",
) -> Sequence:
    """Generate an organ-on-chip-like cell movie with a known perturbation."""
    if scenario not in {"exposure_only", "compound_specific"}:
        raise ValueError(f"Unsupported scenario: {scenario}")
    rng = np.random.default_rng(seed)
    h, w = shape
    sampled_flow = float(rng.choice([0.0, 2.0, 10.0, 40.0]))
    if flow_rate is None:
        flow_rate = sampled_flow
    if scenario == "compound_specific":
        compound_id = int(rng.integers(0, len(COMPOUND_CONTEXT)))
        compound_logp, target_pathway, compound_sensitivity = COMPOUND_CONTEXT[compound_id]
    else:
        compound_id = 0
        compound_logp, target_pathway, compound_sensitivity = 0.0, 0.0, 1.0
    clearance_factor = 1.0 / (1.0 + 0.045 * flow_rate)
    shear_stress = 0.18 * flow_rate
    shear_penalty = max(0.0, flow_rate - 15.0) / 30.0
    n_cells = int(rng.integers(14, 24))
    y = rng.uniform(18, h - 18, n_cells)
    x = rng.uniform(18, w - 18, n_cells)
    sy = rng.uniform(2.0, 4.3, n_cells)
    sx = sy * rng.uniform(0.8, 1.8, n_cells)
    drift = rng.normal(0, 0.8, (n_cells, 2))
    # The response is driven by effective exposure plus a high-shear penalty.
    # This is a transparent synthetic physics proxy, not a biological law.
    effective_dose = dose * clearance_factor
    exposure_response = effective_dose * (1.15 + 0.85 * compound_sensitivity)
    pathway_response = target_pathway * effective_dose * 0.18
    if scenario == "compound_specific":
        # A latent, sequence-level susceptibility factor is only observable
        # through the resulting temporal phenotype.  It is intentionally not
        # included in the compound metadata or hydrodynamic feature block.
        latent_cell_state = float(rng.normal(0.0, 0.28))
        toxicity = np.clip(
            0.16
            + effective_dose * (0.48 + 0.34 * compound_sensitivity)
            + pathway_response * 0.55
            + shear_penalty * 0.22
            + latent_cell_state
            + rng.normal(0, 0.05),
            0,
            1,
        )
    else:
        toxicity = np.clip(exposure_response + pathway_response + shear_penalty * 0.32 + rng.normal(0, 0.08), 0, 1)
    label = int(toxicity > 0.42)
    out = np.zeros((frames, h, w), dtype=np.float32)
    for t in range(frames):
        image = rng.normal(0.025, 0.012, shape).astype(np.float32)
        vitality = np.clip(1.0 - toxicity * (t / max(frames - 1, 1)) * 1.20, 0.12, 1.0)
        if flow_rate > 15.0:
            vitality = np.clip(vitality - shear_penalty * (t / max(frames - 1, 1)) * 0.22, 0.08, 1.0)
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
    viability_target = float(np.clip(100.0 * (1.0 - toxicity), 0.0, 100.0))
    ic50_target = float(-7.2 + 3.6 * (1.0 - toxicity) + 0.015 * flow_rate)
    return Sequence(
        out,
        float(dose),
        label,
        seed,
        float(flow_rate),
        float(toxicity),
        viability_target,
        ic50_target,
        compound_id,
        float(compound_logp),
        float(target_pathway),
        float(compound_sensitivity),
    )


def segment(
    frame: np.ndarray,
    *,
    threshold_scale: float = 0.35,
    min_area: int = 8,
    max_area: int = 1200,
    opening_size: int = 2,
    closing_size: int = 3,
    method: str = "connected_components",
    min_distance: int = 5,
    min_peak_height: float = 2.0,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Segment bright cells and return a labelled image plus interpretable objects.

    The defaults preserve the synthetic benchmark.  Explicit parameters allow
    a calibration split to adapt the front-end to a new microscopy domain
    without changing the downstream feature or model code. The optional
    distance-watershed method splits touching foreground objects; it is an
    experimental instance-segmentation option, not the default OoC model.
    """
    if method not in {"connected_components", "distance_watershed"}:
        raise ValueError(f"Unsupported segmentation method: {method}")
    threshold = max(float(np.quantile(frame, 0.985) * threshold_scale), 0.10)
    mask = frame > threshold
    if opening_size > 0:
        mask = ndimage.binary_opening(mask, structure=np.ones((opening_size, opening_size)))
    if closing_size > 0:
        mask = ndimage.binary_closing(mask, structure=np.ones((closing_size, closing_size)))
    labels, _ = ndimage.label(mask)
    if method == "distance_watershed":
        for idx, bounds in enumerate(ndimage.find_objects(labels), start=1):
            if bounds is None:
                continue
            component = labels[bounds]
            area = int(np.count_nonzero(component == idx))
            if area < min_area or area > max_area:
                component[component == idx] = 0
        foreground = labels > 0
        if foreground.any():
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed

            distance = ndimage.distance_transform_edt(foreground)
            coordinates = peak_local_max(
                distance,
                min_distance=min_distance,
                threshold_abs=min_peak_height,
                labels=foreground.astype(np.uint8),
                exclude_border=False,
            )
            markers = np.zeros(foreground.shape, dtype=np.int32)
            for marker_id, (row, column) in enumerate(coordinates, start=1):
                markers[row, column] = marker_id
            components, _ = ndimage.label(foreground)
            next_marker = len(coordinates) + 1
            for component_id, bounds in enumerate(ndimage.find_objects(components), start=1):
                if bounds is None:
                    continue
                component = components[bounds] == component_id
                if np.any(markers[bounds][component]):
                    continue
                local_distance = np.where(component, distance[bounds], -1.0)
                local_row, local_column = np.unravel_index(np.argmax(local_distance), local_distance.shape)
                markers[bounds[0].start + local_row, bounds[1].start + local_column] = next_marker
                next_marker += 1
            labels = watershed(-distance, markers, mask=foreground).astype(np.int32, copy=False)
    objects: list[dict[str, float]] = []
    for idx, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        component = labels[bounds]
        local_y, local_x = np.nonzero(component == idx)
        ys = local_y + bounds[0].start
        xs = local_x + bounds[1].start
        area = len(xs)
        if area < min_area or area > max_area:
            component[component == idx] = 0
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


def physics_features(sequence: Sequence) -> np.ndarray:
    """Return transparent hydrodynamic and exposure covariates."""
    flow = float(sequence.flow_rate)
    clearance = 1.0 / (1.0 + 0.045 * flow)
    return np.array([
        sequence.dose,
        flow,
        0.18 * flow,
        clearance,
        sequence.dose * clearance,
        sequence.compound_logp,
        sequence.target_pathway,
        sequence.compound_sensitivity,
    ], dtype=float)


def cross_modal_features(temporal: np.ndarray, physical: np.ndarray) -> np.ndarray:
    """Fuse phenotype and physics with explicit low-order interactions.

    This is an interpretable local analogue of multimodal cross-attention: the
    readout can condition phenotype changes on exposure and shear without
    hiding the interaction in a large neural network.
    """
    interactions = np.einsum("ij,ik->ijk", temporal, physical[:, 2:]).reshape(len(temporal), -1)
    return np.c_[temporal, physical, interactions]


def fused_feature_names() -> list[str]:
    interaction_names = [f"{t}×{p}" for t in FEATURE_NAMES + [f"reservoir_{i}" for i in range(32)] for p in PHYSICS_FEATURE_NAMES[2:]]
    return FEATURE_NAMES + [f"reservoir_{i}" for i in range(32)] + PHYSICS_FEATURE_NAMES + interaction_names


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
    predicted_bins, observed_bins, counts = calibration_points(y, p)
    ece = float(np.sum((counts / max(len(y), 1)) * np.abs(observed_bins - predicted_bins)))
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": ece,
    }


def calibration_points(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predicted, observed and count values for non-empty bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    predicted, observed, counts = [], [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (p >= lower) & ((p < upper) if upper < 1.0 else (p <= upper))
        if np.any(in_bin):
            predicted.append(float(np.mean(p[in_bin])))
            observed.append(float(np.mean(y[in_bin])))
            counts.append(int(np.sum(in_bin)))
    return np.asarray(predicted), np.asarray(observed), np.asarray(counts, dtype=float)


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
    }


def _split_indices(
    y: np.ndarray,
    n_samples: int,
    seed: int,
    split_mode: str,
    compound_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Create either a stratified random split or a leakage-resistant group split."""
    idx = np.arange(n_samples)
    if split_mode == "stratified":
        train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=seed, stratify=y)
        return train_idx, test_idx, None
    if split_mode == "compound_holdout":
        if compound_ids is None:
            raise ValueError("compound_ids are required for compound_holdout")
        groups = np.asarray(compound_ids)
        if np.unique(groups).size < 4:
            raise ValueError("compound_holdout requires at least four compounds")
        splitter = GroupShuffleSplit(n_splits=64, test_size=0.25, random_state=seed)
        for train_idx, test_idx in splitter.split(idx, y, groups):
            if np.unique(y[train_idx]).size == 2 and np.unique(y[test_idx]).size == 2:
                return train_idx, test_idx, int(np.unique(groups).size)
        raise RuntimeError("Could not find a compound holdout containing both classes")
    if split_mode != "grouped":
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    # Synthetic acquisition batches stand in for independent chips/experiments.
    # A real-data run must replace these IDs with measured chip or experiment IDs.
    groups = np.arange(n_samples) // 8
    splitter = GroupShuffleSplit(n_splits=32, test_size=0.25, random_state=seed)
    for train_idx, test_idx in splitter.split(idx, y, groups):
        if np.unique(y[train_idx]).size == 2 and np.unique(y[test_idx]).size == 2:
            return train_idx, test_idx, int(np.unique(groups).size)
    raise RuntimeError("Could not find a grouped split containing both classes")


def run(
    out_dir: Path,
    seed: int = 42,
    n_samples: int = 180,
    split_mode: str = "stratified",
    scenario: str = "compound_specific",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sequences = [
        generate_sequence(
            int(rng.integers(1_000_000)),
            float(rng.uniform(0, 1)),
            scenario=scenario,
        )
        for _ in range(n_samples)
    ]
    X, static_X, physical_X, y, viability, ic50, tables, doses, compound_ids = [], [], [], [], [], [], [], [], []
    for seq in sequences:
        feat, table = phenotype_features(seq)
        first_objects = table[table["frame"] == table["frame"].min()] if not table.empty else table
        static_X.append([
            float(len(first_objects)),
            float(first_objects["area"].mean()) if not first_objects.empty else 0.0,
            float(first_objects["intensity"].mean()) if not first_objects.empty else 0.0,
            float(first_objects["elongation"].mean()) if not first_objects.empty else 0.0,
        ])
        X.append(feat); y.append(seq.label); tables.append(table); doses.append(seq.dose); compound_ids.append(seq.compound_id)
        physical_X.append(physics_features(seq)); viability.append(seq.viability_target); ic50.append(seq.ic50_target)
    X, static_X, physical_X = np.asarray(X), np.asarray(static_X), np.asarray(physical_X)
    y, viability, ic50, doses, compound_ids = np.asarray(y), np.asarray(viability), np.asarray(ic50), np.asarray(doses), np.asarray(compound_ids)
    train_idx, test_idx, group_count = _split_indices(y, n_samples, seed, split_mode, compound_ids)
    baseline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed))])
    baseline.fit(static_X[train_idx], y[train_idx])
    p_base = baseline.predict_proba(static_X[test_idx])[:, 1]
    reservoir = TemporalReservoir(input_dim=3, seed=seed + 11)
    R = reservoir.transform(tables)
    temporal_X = np.c_[X, R]
    temporal_model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed))])
    temporal_model.fit(temporal_X[train_idx], y[train_idx])
    p_temporal = temporal_model.predict_proba(temporal_X[test_idx])[:, 1]
    fused_X = cross_modal_features(temporal_X, physical_X)
    physics_model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed))])
    physics_model.fit(physical_X[train_idx], y[train_idx])
    p_physics = physics_model.predict_proba(physical_X[test_idx])[:, 1]
    no_interaction_model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed))])
    no_interaction_model.fit(np.c_[temporal_X[train_idx], physical_X[train_idx]], y[train_idx])
    p_no_interaction = no_interaction_model.predict_proba(np.c_[temporal_X[test_idx], physical_X[test_idx]])[:, 1]
    multimodal = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed, C=0.7))])
    multimodal.fit(fused_X[train_idx], y[train_idx])
    p_multimodal = multimodal.predict_proba(fused_X[test_idx])[:, 1]
    primary_model = "multimodal_no_interactions"
    # The interaction block is intentionally wide (p > n for small studies).
    # A stronger penalty plus LSQR avoids ill-conditioned normal-equation
    # solutions that can explode on otherwise valid random seeds.
    # Regression heads use the compact multimodal block rather than all
    # pairwise interactions: with p > n, the wide interaction matrix can make
    # otherwise valid seeds numerically explosive.  The classifier still uses
    # the full auditable interaction design above.
    regression_X = np.c_[temporal_X, physical_X]
    viability_model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0, solver="lsqr"))])
    ic50_model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0, solver="lsqr"))])
    viability_model.fit(regression_X[train_idx], viability[train_idx])
    ic50_model.fit(regression_X[train_idx], ic50[train_idx])
    viability_pred = np.clip(
        viability_model.predict(regression_X[test_idx]),
        viability[train_idx].min(),
        viability[train_idx].max(),
    )
    ic50_pred = np.clip(
        ic50_model.predict(regression_X[test_idx]),
        ic50[train_idx].min(),
        ic50[train_idx].max(),
    )
    demo = sequences[int(np.argmax([s.dose * (1.0 / (1.0 + 0.045 * s.flow_rate)) for s in sequences]))]
    counterfactual_rows = []
    for flow in [0.0, 2.0, 10.0, 20.0, 40.0]:
        cf = generate_sequence(demo.seed, demo.dose, flow_rate=flow, scenario=scenario)
        cf_feat, cf_table = phenotype_features(cf)
        cf_r = reservoir.transform([cf_table])[0]
        cf_physics = physics_features(cf)[None, :]
        cf_temporal = np.concatenate([cf_feat, cf_r])[None, :]
        cf_fused = cross_modal_features(cf_temporal, cf_physics)
        cf_regression = np.c_[cf_temporal, cf_physics]
        counterfactual_rows.append({
            "flow_rate_uL_min": flow,
            "wall_shear_proxy": 0.18 * flow,
            "predicted_toxicity_probability": float(no_interaction_model.predict_proba(np.c_[cf_temporal, cf_physics])[0, 1]),
            "interaction_ablation_probability": float(multimodal.predict_proba(cf_fused)[0, 1]),
            "predicted_viability": float(
                np.clip(
                    viability_model.predict(cf_regression)[0],
                    viability[train_idx].min(),
                    viability[train_idx].max(),
                )
            ),
        })
    counterfactual = pd.DataFrame(counterfactual_rows)
    counterfactual.to_csv(out_dir / "counterfactual_flow.csv", index=False)
    result = {
        "seed": seed,
        "n_samples": n_samples,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "split_mode": split_mode,
        "scenario": scenario,
        "primary_model": primary_model,
        "group_count": group_count,
        "data_kind": "synthetic_organ_on_chip_proxy",
        "baseline": metrics(y[test_idx], p_base),
        "physics_only": metrics(y[test_idx], p_physics),
        "temporal_reservoir": metrics(y[test_idx], p_temporal),
        "multimodal_no_interactions": metrics(y[test_idx], p_no_interaction),
        "multimodal_physics": metrics(y[test_idx], p_multimodal),
        "multimodal_vs_temporal_delta_roc_auc": float(roc_auc_score(y[test_idx], p_multimodal) - roc_auc_score(y[test_idx], p_temporal)),
        "multimodal_viability": regression_metrics(viability[test_idx], viability_pred),
        "multimodal_ic50": regression_metrics(ic50[test_idx], ic50_pred),
        "feature_names": FEATURE_NAMES,
        "physics_feature_names": PHYSICS_FEATURE_NAMES,
        "fusion_feature_count": int(fused_X.shape[1]),
        "fusion_design": "phenotype + fixed temporal reservoir + hydrodynamic/compound covariates + explicit cross-modal interactions",
        "uncertainty_proxy": "distance from 0.5; not a clinical confidence interval",
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pred = pd.DataFrame({
        "index": test_idx,
        "dose": doses[test_idx],
        "label": y[test_idx],
        "baseline_probability": p_base,
        "temporal_probability": p_temporal,
        "multimodal_no_interactions_probability": p_no_interaction,
        "multimodal_interaction_probability": p_multimodal,
    })
    pred.to_csv(out_dir / "predictions.csv", index=False)
    pd.concat([t.assign(sequence=i) for i, t in enumerate(tables)], ignore_index=True).to_csv(out_dir / "phenotype_table.csv", index=False)

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

    predicted_bins, observed_bins, bin_counts = calibration_points(y[test_idx], p_multimodal)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    sizes = 30 + 120 * bin_counts / max(float(bin_counts.max()), 1.0)
    ax.scatter(predicted_bins, observed_bins, s=sizes, color="navy", label="multimodal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Probability calibration (held-out test set)")
    ax.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(counterfactual["flow_rate_uL_min"], counterfactual["predicted_viability"], marker="o", label="predicted viability")
    ax2 = ax.twinx(); ax2.plot(counterfactual["flow_rate_uL_min"], counterfactual["predicted_toxicity_probability"], marker="s", color="crimson", label="toxicity probability")
    ax.set_xlabel("flow rate (uL/min)"); ax.set_ylabel("viability"); ax2.set_ylabel("toxicity probability"); ax.set_title("Physics counterfactual: flow modulation")
    fig.tight_layout(); fig.savefig(out_dir / "counterfactual_flow.png", dpi=160); plt.close(fig)

    html = f"""<!doctype html><meta charset='utf-8'><title>NeuroChip Twin demo</title>
    <h1>NeuroChip Twin</h1><p>Self-contained synthetic organ-on-chip proxy benchmark; no clinical claim.</p>
    <p>Static ROC-AUC: <b>{result['baseline']['roc_auc']:.3f}</b> · Temporal: <b>{result['temporal_reservoir']['roc_auc']:.3f}</b> · Primary additive multimodal: <b>{result[primary_model]['roc_auc']:.3f}</b> · Interaction ablation: <b>{result['multimodal_physics']['roc_auc']:.3f}</b></p>
    <img src='demo_overview.png' style='max-width:100%'><img src='confusion_matrix.png' style='max-width:420px'><img src='calibration_curve.png' style='max-width:420px'><img src='counterfactual_flow.png' style='max-width:620px'>
    <h2>Interpretation</h2><pre>{json.dumps(dict(zip(FEATURE_NAMES, demo_features.round(4))), indent=2)}</pre>
    <h2>Physics counterfactual</h2><pre>{counterfactual.to_string(index=False)}</pre>
    <p>Uncertainty is a transparent distance-from-threshold proxy and must not be read as a clinical confidence interval.</p>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--split-mode", choices=["stratified", "grouped", "compound_holdout"], default="stratified")
    parser.add_argument("--scenario", choices=["exposure_only", "compound_specific"], default="compound_specific")
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.seed, args.samples, args.split_mode, args.scenario), indent=2))


if __name__ == "__main__":
    main()
