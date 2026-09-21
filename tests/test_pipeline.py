from pathlib import Path

import numpy as np

from src.neurochip_twin import generate_sequence, phenotype_features, run, segment, track_objects
from src.external_validation import _pixel_metrics
from src.validation import run_validation


def test_generation_is_deterministic():
    a = generate_sequence(12, 0.7)
    b = generate_sequence(12, 0.7)
    assert np.array_equal(a.frames, b.frames)
    assert a.label == b.label


def test_segmentation_and_features_are_finite():
    seq = generate_sequence(3, 0.8)
    _, objects = segment(seq.frames[0])
    features, table = phenotype_features(seq)
    assert len(objects) > 0
    assert table["track_id"].nunique() > 0
    assert np.isfinite(features).all()


def test_tracking_assigns_ids():
    detections = [[{"cy": 1.0, "cx": 1.0}], [{"cy": 1.5, "cx": 1.5}]]
    tracked = track_objects(detections)
    assert tracked[0][0]["track_id"] == tracked[1][0]["track_id"]


def test_end_to_end_outputs(tmp_path: Path):
    result = run(tmp_path, seed=5, n_samples=48)
    assert result["test_size"] == 12
    assert 0.0 <= result["temporal_reservoir"]["roc_auc"] <= 1.0
    assert 0.0 <= result["multimodal_physics"]["roc_auc"] <= 1.0
    assert "r2" in result["multimodal_viability"]
    assert (tmp_path / "counterfactual_flow.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "index.html").exists()


def test_multi_seed_validation_writes_audit_report(tmp_path: Path):
    report = run_validation(tmp_path, seeds=[1, 2], n_samples=40)
    assert report["seeds"] == [1, 2]
    assert (tmp_path / "validation_per_seed.csv").exists()
    assert (tmp_path / "validation_summary.json").exists()
    assert report["aggregate"]["multimodal_physics_roc_auc"]["min"] >= 0.0
    assert report["aggregate"]["multimodal_viability_r2"]["min"] > -10.0


def test_grouped_split_is_supported(tmp_path: Path):
    result = run(tmp_path, seed=42, n_samples=80, split_mode="grouped")
    assert result["split_mode"] == "grouped"
    assert result["group_count"] is not None
    assert 0 < result["test_size"] < result["n_samples"]


def test_external_pixel_metrics_have_expected_bounds():
    metrics = _pixel_metrics(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
    assert metrics == {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0}
