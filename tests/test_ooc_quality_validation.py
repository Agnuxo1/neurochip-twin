from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from src.ooc_quality_validation import run_ooc_quality_audit
from src import ooc_quality_validation


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_ooc_quality_audit_uses_cell_type_held_out_folds(tmp_path):
    archive_path = tmp_path / "tiny-ooc.zip"
    metadata_rows = []
    with ZipFile(archive_path, "w") as archive:
        for cell_type in ("LineA", "LineB", "LineC"):
            for label, class_folder in ((0, "good"), (1, "bad")):
                for replicate in range(2):
                    image_id = f"{cell_type}_{class_folder}_{replicate}"
                    base = np.zeros((64, 64), dtype=np.uint8)
                    if label:
                        base[::2, :] = 220
                    else:
                        base[20:44, 20:44] = 180
                    member = f"train/{class_folder}/{cell_type}/0 days/{image_id}.png"
                    archive.writestr(member, _png_bytes(base))
                    metadata_rows.append(
                        {
                            "imageID": image_id,
                            "cell type": cell_type,
                            "Decision 1/2 (good/bad)": "2" if label else "1",
                            "seeding density, cells/ml": "1000000",
                            "time after seeding, h": "1",
                            "day": "0",
                            "flow rate": "2 uL/min",
                        }
                    )

    summary, predictions = run_ooc_quality_audit(metadata_rows, archive_path)

    assert summary["matched_labelled_images"] == 12
    assert summary["quality_counts"] == {"good": 6, "bad": 6}
    assert len(summary["folds"]) == 3
    assert summary["models"]["image_hog"]["pooled_out_of_fold"]["roc_auc"] is None
    assert summary["models"]["image_hog"]["cell_type_macro"]["roc_auc"] is not None
    assert len(predictions) == 12 * 3
    assert all(fold["train_cell_types"] == [name for name in ("LineA", "LineB", "LineC") if name != fold["held_out_cell_type"]] for fold in summary["folds"])
    assert all(row["held_out_cell_type"] == row["cell_type"] for row in predictions)


def test_ooc_quality_audit_detects_datasheet_folder_label_mismatch(tmp_path):
    archive_path = tmp_path / "mismatch.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("train/bad/LineA/1 days/image-1.png", _png_bytes(np.zeros((64, 64))))
    rows = [
        {
            "imageID": "image-1",
            "cell type": "LineA",
            "Decision 1/2 (good/bad)": "1",
        }
    ]

    with pytest.raises(ValueError, match="label mismatch"):
        run_ooc_quality_audit(rows, archive_path)


def test_ooc_quality_audit_rejects_duplicate_datasheet_image_ids(tmp_path):
    archive_path = tmp_path / "duplicate.zip"
    rows = []
    with ZipFile(archive_path, "w") as archive:
        for cell_type in ("LineA", "LineB", "LineC"):
            for label, class_folder in ((0, "good"), (1, "bad")):
                image_id = f"{cell_type}_{class_folder}"
                archive.writestr(
                    f"train/{class_folder}/{cell_type}/0 days/{image_id}.png",
                    _png_bytes(np.full((64, 64), 180 if label == 0 else 40)),
                )
                row = {
                    "imageID": image_id,
                    "cell type": cell_type,
                    "Decision 1/2 (good/bad)": "2" if label else "1",
                }
                rows.append(row)
        rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="Duplicate labelled image IDs"):
        run_ooc_quality_audit(rows, archive_path)


def test_literature_inspired_encoder_is_an_opt_in_loco_comparison(tmp_path, monkeypatch):
    archive_path = tmp_path / "encoder.zip"
    rows = []
    with ZipFile(archive_path, "w") as archive:
        for cell_type in ("LineA", "LineB", "LineC"):
            for label, class_folder in ((0, "good"), (1, "bad")):
                for replicate in range(2):
                    image_id = f"{cell_type}_{class_folder}_{replicate}"
                    pixels = np.full((64, 64), 180 if label == 0 else 40)
                    archive.writestr(
                        f"train/{class_folder}/{cell_type}/0 days/{image_id}.png",
                        _png_bytes(pixels),
                    )
                    rows.append(
                        {
                            "imageID": image_id,
                            "cell type": cell_type,
                            "Decision 1/2 (good/bad)": "2" if label else "1",
                        }
                    )

    monkeypatch.setattr(
        ooc_quality_validation,
        "_inception_features",
        lambda archive, members, batch_size: (
            np.asarray([[float(i), float(i % 3), 0.5, -0.5] for i in range(len(members))]),
            {"weights": "mock", "embedding_dimensions": 4, "batch_size": batch_size},
        ),
    )
    summary, predictions = run_ooc_quality_audit(
        rows, archive_path, include_inception=True, inception_batch_size=2
    )

    assert summary["literature_comparison_models"] == [
        "inception_v3_logistic",
        "inception_v3_plus_metadata",
    ]
    assert summary["inception_encoder"]["weights"] == "mock"
    assert len(predictions) == 12 * 5
    assert all(
        name in fold
        for fold in summary["folds"]
        for name in ("inception_v3_logistic", "inception_v3_plus_metadata")
    )


def test_inception_only_skips_hog_and_keeps_metadata_hybrid(tmp_path, monkeypatch):
    archive_path = tmp_path / "inception-only.zip"
    rows = []
    with ZipFile(archive_path, "w") as archive:
        for cell_type in ("LineA", "LineB", "LineC"):
            for label, class_folder in ((0, "good"), (1, "bad")):
                for replicate in range(2):
                    image_id = f"{cell_type}_{class_folder}_{replicate}"
                    archive.writestr(
                        f"train/{class_folder}/{cell_type}/0 days/{image_id}.png",
                        _png_bytes(np.full((64, 64), 180 if label == 0 else 40)),
                    )
                    rows.append(
                        {
                            "imageID": image_id,
                            "cell type": cell_type,
                            "Decision 1/2 (good/bad)": "2" if label else "1",
                            "seeding density, cells/ml": "1000000",
                            "time after seeding, h": "1",
                            "day": "0",
                            "flow rate": "2 uL/min",
                        }
                    )

    def no_hog(*args, **kwargs):
        raise AssertionError("inception_only must not compute HOG features")

    monkeypatch.setattr(ooc_quality_validation, "_image_features", no_hog)
    monkeypatch.setattr(
        ooc_quality_validation,
        "_inception_features",
        lambda archive, members, batch_size: (
            np.asarray([[float(i), float(i % 2), 0.25] for i in range(len(members))]),
            {"weights": "mock", "embedding_dimensions": 3, "batch_size": batch_size},
        ),
    )
    summary, predictions = run_ooc_quality_audit(
        rows, archive_path, include_inception=True, inception_only=True
    )

    assert summary["inception_only"] is True
    assert summary["hog_features_included"] is False
    assert set(summary["models"]) == {
        "metadata_only",
        "inception_v3_logistic",
        "inception_v3_plus_metadata",
    }
    assert len(predictions) == 12 * 3
