"""Export aggregate-only OoC image-quality results without per-image rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "per_image_predictions.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("archive_md5_verified"):
        raise ValueError(f"Dataset archive checksum was not verified: {run_dir}")
    if summary.get("task") != "expert-assessed good/bad OOC sample image-quality classification":
        raise ValueError(f"Unexpected target task in {summary_path}")

    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    with predictions_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "cell_type", "held_out_cell_type", "quality_true", "quality_score_bad", "model"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Prediction file is missing columns: {predictions_path}")
        for row in reader:
            by_model[row["model"]].append(row)

    if set(by_model) != set(summary.get("models", {})):
        raise ValueError(f"Model list does not match the audit summary: {run_dir}")

    model_results: dict[str, Any] = {}
    expected_groups = set(summary["cell_type_counts"])
    expected_cell_counts = {key: int(value) for key, value in summary["cell_type_counts"].items()}
    expected_quality_counts = {key: int(value) for key, value in summary["quality_counts"].items()}
    expected_n = int(summary["matched_labelled_images"])
    for model_name, rows in sorted(by_model.items()):
        if len(rows) != expected_n:
            raise ValueError(f"Scored row count does not match the summary for {model_name}")
        ids = [row["image_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate image rows for {model_name} in {run_dir}")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        cell_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        for row in rows:
            if row["cell_type"] != row["held_out_cell_type"]:
                raise ValueError(f"Fold assignment does not hold out its cell type: {run_dir}")
            label = row["quality_true"].strip().lower()
            if label not in {"good", "bad"}:
                raise ValueError(f"Unknown quality label in {run_dir}")
            score = float(row["quality_score_bad"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite quality score in {run_dir}")
            grouped[row["cell_type"]].append(row)
            cell_counts[row["cell_type"]] += 1
            quality_counts[label] += 1
        if set(grouped) != expected_groups:
            raise ValueError(f"Cell-line groups do not match the summary: {run_dir}")
        if dict(cell_counts) != expected_cell_counts:
            raise ValueError(f"Cell-line row counts do not match the summary for {model_name}")
        if dict(quality_counts) != expected_quality_counts:
            raise ValueError(f"Quality-label counts do not match the summary for {model_name}")

        fold_auc: dict[str, float] = {}
        for cell_type, fold_rows in sorted(grouped.items()):
            labels = [{"good": 0, "bad": 1}.get(row["quality_true"].strip().lower()) for row in fold_rows]
            scores = [float(row["quality_score_bad"]) for row in fold_rows]
            fold_auc[cell_type] = float(roc_auc_score(labels, scores))
        macro_auc = sum(fold_auc.values()) / len(fold_auc)
        reported_auc = float(summary["models"][model_name]["cell_type_macro"]["roc_auc"])
        if not math.isclose(macro_auc, reported_auc, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Recomputed macro AUC does not match summary for {model_name}")
        model_results[model_name] = {
            "cell_type_macro_roc_auc": round(macro_auc, 9),
            "per_cell_type_roc_auc": {key: round(value, 6) for key, value in fold_auc.items()},
            "n_scored_images": len(rows),
        }

    provenance = {
        "archive_md5": summary["archive_md5"],
        "metadata_sha256": summary["metadata_sha256"],
        "source_record_url": summary["source_record_url"],
        "source_paper_url": summary["source_paper_url"],
        "license_notes": summary["license_notes"],
        "summary_sha256": _sha256(summary_path),
        "predictions_sha256": _sha256(predictions_path),
    }
    return summary, {"models": model_results, "provenance": provenance}


def build_public_summary(hog_dir: Path, inception_dir: Path) -> dict[str, Any]:
    hog, hog_result = _load_run(hog_dir)
    inception, inception_result = _load_run(inception_dir)
    invariant_fields = ("archive_md5", "metadata_sha256", "cell_type_counts", "quality_counts")
    for field in invariant_fields:
        if hog.get(field) != inception.get(field):
            raise ValueError(f"Audit runs disagree on dataset field {field!r}")

    models = dict(hog_result["models"])
    for model_name, result in inception_result["models"].items():
        if model_name in models and models[model_name] != result:
            raise ValueError(f"Overlapping model results disagree for {model_name!r}")
        models[model_name] = result
    provenance = {
        "dataset_archive_md5": hog["archive_md5"],
        "metadata_sha256": hog["metadata_sha256"],
        "source_record_url": hog["source_record_url"],
        "source_paper_url": hog["source_paper_url"],
        "license_notes": hog["license_notes"],
        "audit_summary_sha256": {
            "hog": hog_result["provenance"]["summary_sha256"],
            "frozen_inception_v3": inception_result["provenance"]["summary_sha256"],
        },
        "private_prediction_file_sha256": {
            "hog": hog_result["provenance"]["predictions_sha256"],
            "frozen_inception_v3": inception_result["provenance"]["predictions_sha256"],
        },
    }
    return {
        "schema_version": "ooc_quality_public_aggregate_v1",
        "dataset": hog["dataset"],
        "task": hog["task"],
        "n_labelled_images": int(hog["matched_labelled_images"]),
        "cell_type_counts": {key: int(value) for key, value in sorted(hog["cell_type_counts"].items())},
        "quality_counts": {key: int(value) for key, value in sorted(hog["quality_counts"].items())},
        "validation": {
            "protocol": "leave-one-cell-type-out; cell type is not a model feature",
            "primary_metric": "unweighted macro mean of six held-out-cell-type ROC-AUCs",
            "pooled_roc_auc": None,
            "interpretation": "Exploratory sample-quality triage only; not toxicity or treatment-response validation.",
        },
        "models": models,
        "license_and_redistribution": {
            "images_included": False,
            "per_image_predictions_included": False,
            "model_weights_included": False,
            "status": "Zenodo record and data paper state different licenses; discrepancy unresolved.",
        },
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hog-audit-dir", type=Path, required=True)
    parser.add_argument("--inception-audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_public_summary(args.hog_audit_dir, args.inception_audit_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
