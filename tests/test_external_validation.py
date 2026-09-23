import numpy as np
import pytest
from PIL import Image

from src.external_validation import (
    _instance_metrics,
    run_external_validation,
)
from src.neurochip_twin import segment


def test_instance_metrics_are_perfect_for_matching_labels():
    labels = np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32)
    result = _instance_metrics(labels, labels.copy())

    assert result == {
        "instance_precision_iou50": 1.0,
        "instance_recall_iou50": 1.0,
        "instance_f1_iou50": 1.0,
        "instance_f1_mean_iou50_95": 1.0,
    }


def test_instance_metrics_penalize_splitting_an_object():
    truth = np.ones((2, 4), dtype=np.int32)
    prediction = np.array([[4, 4, 9, 9], [4, 4, 9, 9]], dtype=np.int32)

    result = _instance_metrics(prediction, truth)

    assert result["instance_precision_iou50"] == 0.5
    assert result["instance_recall_iou50"] == 1.0
    assert result["instance_f1_iou50"] == pytest.approx(2 / 3)
    assert result["instance_f1_mean_iou50_95"] == pytest.approx(1 / 15)


def test_instance_metrics_handle_missed_objects_and_empty_predictions():
    truth = np.array([[1, 1], [0, 2]], dtype=np.int32)
    prediction = np.zeros_like(truth)

    result = _instance_metrics(prediction, truth)

    assert result["instance_precision_iou50"] == 0.0
    assert result["instance_recall_iou50"] == 0.0
    assert result["instance_f1_iou50"] == 0.0
    assert result["instance_f1_mean_iou50_95"] == 0.0


def test_instance_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        _instance_metrics(np.zeros((2, 2), dtype=int), np.zeros((3, 2), dtype=int))


def test_external_validation_excludes_previously_used_case_ids(tmp_path):
    root = tmp_path / "imageset"
    case_ids = ["case-a", "case-b", "case-c"]
    for case_id in case_ids:
        image_dir = root / case_id / "images"
        mask_dir = root / case_id / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        image = np.zeros((16, 16), dtype=np.uint8)
        image[5:10, 5:10] = 220
        Image.fromarray(image).save(image_dir / f"{case_id}.png")
        Image.fromarray((image > 0).astype(np.uint8) * 255).save(mask_dir / "object.png")

    summary = run_external_validation(
        root,
        tmp_path / "out",
        samples=3,
        exclude_case_ids={"case-b"},
    )

    assert summary["dataset_case_count"] == 3
    assert summary["excluded_case_ids"] == ["case-b"]
    assert summary["sampled_cases"] == 2
    assert summary["evaluation_samples"] == 2


def test_external_validation_replays_an_explicit_calibration_and_evaluation_split(tmp_path):
    root = tmp_path / "imageset"
    case_ids = [f"case-{index:02d}" for index in range(10)]
    for case_id in case_ids:
        image_dir = root / case_id / "images"
        mask_dir = root / case_id / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        image = np.zeros((16, 16), dtype=np.uint8)
        image[5:10, 5:10] = 220
        Image.fromarray(image).save(image_dir / f"{case_id}.png")
        Image.fromarray((image > 0).astype(np.uint8) * 255).save(mask_dir / "object.png")

    calibration_ids = case_ids[:8]
    evaluation_ids = case_ids[8:]
    summary = run_external_validation(
        root,
        tmp_path / "out",
        calibrate=True,
        method="distance_watershed",
        case_split={
            "calibration_case_ids": calibration_ids,
            "evaluation_case_ids": evaluation_ids,
        },
    )

    assert summary["split_source"] == "explicit_case_ids"
    assert summary["calibration_case_ids"] == calibration_ids
    assert summary["evaluation_case_ids"] == evaluation_ids
    assert summary["calibrated"] is True
    assert summary["calibration_samples"] == 8
    assert summary["evaluation_samples"] == 2


def test_external_validation_rejects_overlapping_explicit_splits(tmp_path):
    root = tmp_path / "imageset"
    case_ids = ["case-a", "case-b"]
    for case_id in case_ids:
        image_dir = root / case_id / "images"
        mask_dir = root / case_id / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        image = np.zeros((8, 8), dtype=np.uint8)
        Image.fromarray(image).save(image_dir / f"{case_id}.png")
        Image.fromarray(image).save(mask_dir / "empty.png")

    with pytest.raises(ValueError, match="unique and disjoint"):
        run_external_validation(
            root,
            tmp_path / "out",
            case_split={
                "calibration_case_ids": ["case-a"],
                "evaluation_case_ids": ["case-a", "case-b"],
            },
        )


def test_distance_watershed_splits_touching_foreground_components():
    yy, xx = np.mgrid[:48, :64]
    foreground = ((yy - 24) ** 2 + (xx - 24) ** 2 <= 9**2) | (
        (yy - 24) ** 2 + (xx - 38) ** 2 <= 9**2
    )
    frame = foreground.astype(np.float32)
    baseline_labels, baseline_objects = segment(
        frame,
        threshold_scale=0.1,
        min_area=4,
        opening_size=0,
        closing_size=0,
    )

    split_labels, split_objects = segment(
        frame,
        threshold_scale=0.1,
        min_area=4,
        opening_size=0,
        closing_size=0,
        method="distance_watershed",
        min_distance=5,
        min_peak_height=2.0,
    )

    assert len(baseline_objects) == 1
    assert len(split_objects) == 2
    assert np.array_equal(split_labels > 0, baseline_labels > 0)
