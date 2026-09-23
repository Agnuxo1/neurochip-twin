import csv
import json

from scripts.export_ooc_quality_summary import build_public_summary


def _write_audit(root, model_name, reported_auc=1.0):
    root.mkdir()
    summary = {
        "archive_md5_verified": True,
        "task": "expert-assessed good/bad OOC sample image-quality classification",
        "models": {model_name: {"cell_type_macro": {"roc_auc": reported_auc}}},
        "cell_type_counts": {"LineA": 4, "LineB": 4},
        "quality_counts": {"good": 4, "bad": 4},
        "archive_md5": "a" * 32,
        "metadata_sha256": "b" * 64,
        "source_record_url": "https://zenodo.org/records/example",
        "source_paper_url": "https://doi.org/10.0000/example",
        "license_notes": {"resolution": "unresolved"},
        "matched_labelled_images": 8,
        "dataset": "Organ-on-a-Chip (OOC) Image Dataset",
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = []
    for line in ("LineA", "LineB"):
        rows.extend([
            {"image_id": f"{line}-good-1", "cell_type": line, "held_out_cell_type": line, "quality_true": "good", "quality_score_bad": "0.1", "quality_predicted": "good", "model": model_name},
            {"image_id": f"{line}-good-2", "cell_type": line, "held_out_cell_type": line, "quality_true": "good", "quality_score_bad": "0.2", "quality_predicted": "good", "model": model_name},
            {"image_id": f"{line}-bad-1", "cell_type": line, "held_out_cell_type": line, "quality_true": "bad", "quality_score_bad": "0.8", "quality_predicted": "bad", "model": model_name},
            {"image_id": f"{line}-bad-2", "cell_type": line, "held_out_cell_type": line, "quality_true": "bad", "quality_score_bad": "0.9", "quality_predicted": "bad", "model": model_name},
        ])
    with (root / "per_image_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_export_contains_aggregate_metrics_not_image_rows(tmp_path):
    hog = tmp_path / "hog"
    inception = tmp_path / "inception"
    _write_audit(hog, "image_hog")
    _write_audit(inception, "inception_v3_logistic")

    result = build_public_summary(hog, inception)

    assert result["n_labelled_images"] == 8
    assert result["models"]["image_hog"]["cell_type_macro_roc_auc"] == 1.0
    assert result["models"]["inception_v3_logistic"]["per_cell_type_roc_auc"] == {
        "LineA": 1.0,
        "LineB": 1.0,
    }
    assert "image_id" not in json.dumps(result)
    assert result["license_and_redistribution"]["images_included"] is False


def test_export_rejects_audit_run_with_leaked_cell_type_fold(tmp_path):
    hog = tmp_path / "hog"
    inception = tmp_path / "inception"
    _write_audit(hog, "image_hog")
    _write_audit(inception, "inception_v3_logistic")
    path = inception / "per_image_predictions.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0])
    rows[0]["held_out_cell_type"] = "Other"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    try:
        build_public_summary(hog, inception)
    except ValueError as exc:
        assert "Fold assignment" in str(exc)
    else:
        raise AssertionError("Expected held-out-cell-type consistency check to fail")


def test_export_rejects_audit_rows_missing_from_summary_count(tmp_path):
    hog = tmp_path / "hog"
    inception = tmp_path / "inception"
    _write_audit(hog, "image_hog")
    _write_audit(inception, "inception_v3_logistic")
    path = hog / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["matched_labelled_images"] = 9
    path.write_text(json.dumps(summary), encoding="utf-8")

    try:
        build_public_summary(hog, inception)
    except ValueError as exc:
        assert "Scored row count" in str(exc)
    else:
        raise AssertionError("Expected incomplete audit predictions to be rejected")


def test_export_rejects_conflicting_duplicate_model_across_runs(tmp_path):
    hog = tmp_path / "hog"
    inception = tmp_path / "inception"
    _write_audit(hog, "metadata_only", reported_auc=1.0)
    _write_audit(inception, "metadata_only", reported_auc=0.0)

    predictions = inception / "per_image_predictions.csv"
    with predictions.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0])
    for row in rows:
        row["quality_score_bad"] = "0.9" if row["quality_true"] == "good" else "0.1"
    with predictions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    try:
        build_public_summary(hog, inception)
    except ValueError as exc:
        assert "Overlapping model results disagree" in str(exc)
    else:
        raise AssertionError("Expected conflicting duplicate model results to fail")
