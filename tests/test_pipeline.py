from pathlib import Path

import numpy as np

from src.neurochip_twin import generate_sequence, phenotype_features, run, segment, track_objects


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
