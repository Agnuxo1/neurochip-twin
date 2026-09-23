"""Cell-line-held-out image-quality audit on the public OOC image dataset.

This is a sample-quality control experiment (expert ``good``/``bad`` labels),
not toxicity or treatment-response prediction. The Zenodo record says CC-BY-4.0
while its data paper says CC-BY-SA; this audit keeps the source external, does
not redistribute images/model weights, and records the discrepancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog

from .ooc_metadata_audit import METADATA_URL, SOURCE_RECORD_URL, read_xlsx_table

SOURCE_PAPER_URL = "https://doi.org/10.3390/data9020028"
EXPECTED_ARCHIVE_MD5 = "8f7e058996203d48eb03b2d86c0a2e4d"
ARCHIVE_URL = (
    "https://zenodo.org/records/10203721/files/"
    "OOC_image_dataset.zip?download=1"
)
QUALITY_COLUMN = "Decision 1/2 (good/bad)"
CELL_TYPE_COLUMN = "cell type"
IMAGE_ID_COLUMN = "imageID"
NUMERIC_COLUMNS = (
    "seeding density, cells/ml",
    "time after seeding, h",
    "day",
    "flow rate",
)
IMAGE_SIZE = (64, 64)
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _number(value: Any) -> float:
    """Parse the numeric portion of a datasheet value; missing becomes NaN."""

    text = str(value or "").strip().replace(",", "")
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        return float(match.group(0)) if match else float("nan")


def _label(value: Any) -> int | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "1.0", "good"}:
        return 0
    if normalized in {"2", "2.0", "bad"}:
        return 1
    return None


def _image_index(archive: ZipFile) -> dict[str, str]:
    result: dict[str, str] = {}
    for member in archive.namelist():
        path = PurePosixPath(member.replace("\\", "/"))
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        key = path.stem.strip().casefold()
        if key in result:
            raise ValueError(f"Duplicate image stem in dataset archive: {key!r}")
        result[key] = member
    return result


def _image_features(archive: ZipFile, member: str) -> np.ndarray:
    with Image.open(BytesIO(archive.read(member))) as opened:
        image = ImageOps.fit(
            opened.convert("L"),
            IMAGE_SIZE,
            method=Image.Resampling.BILINEAR,
        )
        gray = np.asarray(image, dtype=np.float32) / 255.0

    gy, gx = np.gradient(gray)
    laplacian = ndimage.laplace(gray)
    summary = np.asarray(
        [
            float(gray.mean()),
            float(gray.std()),
            float(np.quantile(gray, 0.01)),
            float(np.quantile(gray, 0.50)),
            float(np.quantile(gray, 0.99)),
            float(np.mean(gray <= 0.01)),
            float(np.mean(gray >= 0.99)),
            float(np.mean(np.hypot(gx, gy))),
            float(laplacian.var()),
        ],
        dtype=np.float32,
    )
    texture = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32, copy=False)
    return np.concatenate((texture, summary))


def _metadata_features(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([_number(row.get(column)) for column in NUMERIC_COLUMNS], dtype=float)


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    predicted = (scores >= 0).astype(np.int8)
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None
    return {
        "roc_auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1_bad": float(f1_score(y_true, predicted, zero_division=0)),
        "precision_bad": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_bad": float(recall_score(y_true, predicted, zero_division=0)),
    }


def _new_models(include_inception: bool = False, inception_only: bool = False) -> dict[str, Any]:
    metadata_model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=42),
    )
    models = {"metadata_only": metadata_model}
    if not inception_only:
        models["image_hog"] = make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=20_000,
                random_state=42,
            ),
        )
        models["image_plus_metadata"] = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=20_000,
                random_state=42,
            ),
        )
    if include_inception:
        models["inception_v3_logistic"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=42),
        )
        models["inception_v3_plus_metadata"] = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=42),
        )
    return models


def _inception_features(
    archive: ZipFile, members: Sequence[str], batch_size: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract frozen ImageNet Inception-v3 pool embeddings in small batches."""

    if batch_size < 1:
        raise ValueError("Inception batch size must be positive")
    try:
        import torch
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError as exc:
        raise RuntimeError(
            "The Inception comparison requires optional torch and torchvision dependencies"
        ) from exc

    weights = Inception_V3_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(weights=weights).to(device)
    model.fc = torch.nn.Identity()
    model.eval()

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(members), batch_size):
            tensors = []
            for member in members[start : start + batch_size]:
                with Image.open(BytesIO(archive.read(member))) as opened:
                    tensors.append(transform(opened.convert("RGB")))
            logits = model(torch.stack(tensors).to(device))
            embeddings = logits.logits if hasattr(logits, "logits") else logits
            batches.append(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
            completed = min(start + batch_size, len(members))
            if completed % 256 == 0 or completed == len(members):
                print(f"Extracted Inception embeddings for {completed}/{len(members)} images", flush=True)

    return np.vstack(batches), {
        "weights": "torchvision Inception_V3_Weights.IMAGENET1K_V1",
        "embedding_dimensions": 2048,
        "preprocessing": "torchvision weight preset: RGB, resize 342, center crop 299, ImageNet normalization",
        "device": str(device),
        "batch_size": batch_size,
    }


def run_ooc_quality_audit(
    metadata_rows: Sequence[Mapping[str, Any]],
    archive_path: str | Path,
    *,
    include_inception: bool = False,
    inception_only: bool = False,
    inception_batch_size: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate quality classification by leaving one cell line out at a time.

    Image ID, quality label, and cell line are read from the released datasheet.
    Cell line is used only to define the outer test fold, never as a predictor.
    Hyperparameters are fixed before the held-out folds are scored.
    """

    rows = [dict(row) for row in metadata_rows]
    if inception_only and not include_inception:
        raise ValueError("inception_only requires include_inception=True")
    with ZipFile(archive_path) as archive:
        image_index = _image_index(archive)
        samples: list[dict[str, Any]] = []
        missing_image_ids: list[str] = []
        for row in rows:
            image_id = str(row.get(IMAGE_ID_COLUMN, "")).strip()
            label = _label(row.get(QUALITY_COLUMN))
            cell_type = str(row.get(CELL_TYPE_COLUMN, "")).strip()
            if not image_id or label is None or not cell_type:
                continue
            member = image_index.get(image_id.casefold())
            if member is None:
                missing_image_ids.append(image_id)
                continue
            path_parts = {
                part.casefold()
                for part in PurePosixPath(member.replace("\\", "/")).parts
            }
            folder_label = "good" if "good" in path_parts else "bad" if "bad" in path_parts else None
            expected_folder_label = "bad" if label else "good"
            if folder_label is not None and folder_label != expected_folder_label:
                raise ValueError(
                    f"Datasheet/folder label mismatch for image {image_id!r}"
                )
            samples.append(
                {
                    "image_id": image_id,
                    "archive_member": member,
                    "cell_type": cell_type,
                    "label": label,
                    "image": None if inception_only else _image_features(archive, member),
                    "metadata": _metadata_features(row),
                }
            )
            if not inception_only and len(samples) % 256 == 0:
                print(f"Extracted fixed image features for {len(samples)} labelled samples", flush=True)

    if missing_image_ids:
        raise ValueError(
            f"{len(missing_image_ids)} labelled image IDs were absent from the archive; "
            f"examples: {missing_image_ids[:3]}"
        )
    if not samples:
        raise ValueError("No labelled images matched the datasheet and archive")
    image_ids = [sample["image_id"].casefold() for sample in samples]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate labelled image IDs in the datasheet")
    inception_matrix = None
    inception_info = None
    if include_inception:
        with ZipFile(archive_path) as archive:
            inception_matrix, inception_info = _inception_features(
                archive,
                [sample["archive_member"] for sample in samples],
                inception_batch_size,
            )

    labels = np.asarray([sample["label"] for sample in samples], dtype=np.int8)
    groups = np.asarray([sample["cell_type"] for sample in samples], dtype=object)
    metadata_matrix = np.vstack([sample["metadata"] for sample in samples])
    image_matrix = None if inception_only else np.vstack([sample["image"] for sample in samples])
    combined_matrix = None if inception_only else np.hstack((image_matrix, metadata_matrix))

    prediction_rows: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    models_template = _new_models(include_inception, inception_only)
    scores_by_model: dict[str, np.ndarray] = {
        name: np.full(len(samples), np.nan, dtype=float) for name in models_template
    }
    for held_out in sorted(set(groups)):
        print(f"Evaluating held-out cell type: {held_out}", flush=True)
        test = groups == held_out
        train = ~test
        if len(np.unique(labels[train])) != 2:
            raise ValueError(f"Training fold for {held_out!r} does not contain both labels")
        if len(np.unique(labels[test])) != 2:
            raise ValueError(f"Held-out cell line {held_out!r} does not contain both labels")
        fold_row: dict[str, Any] = {
            "held_out_cell_type": str(held_out),
            "train_samples": int(train.sum()),
            "test_samples": int(test.sum()),
            "test_good": int(np.count_nonzero(labels[test] == 0)),
            "test_bad": int(np.count_nonzero(labels[test] == 1)),
            "train_cell_types": sorted(set(groups[train])),
        }
        models = _new_models(include_inception, inception_only)
        matrices = {"metadata_only": metadata_matrix}
        if not inception_only:
            matrices["image_hog"] = image_matrix
            matrices["image_plus_metadata"] = combined_matrix
        if inception_matrix is not None:
            matrices["inception_v3_logistic"] = inception_matrix
            matrices["inception_v3_plus_metadata"] = np.hstack((inception_matrix, metadata_matrix))
        for name, model in models.items():
            matrix = matrices[name]
            model.fit(matrix[train], labels[train])
            scores = np.asarray(model.decision_function(matrix[test]), dtype=float)
            scores_by_model[name][test] = scores
            fold_row[name] = _metrics(labels[test], scores)
            test_indices = np.flatnonzero(test)
            for index, score in zip(test_indices, scores):
                prediction_rows.append(
                    {
                        "image_id": samples[index]["image_id"],
                        "cell_type": samples[index]["cell_type"],
                        "held_out_cell_type": str(held_out),
                        "quality_true": "bad" if labels[index] else "good",
                        "quality_score_bad": float(score),
                        "quality_predicted": "bad" if score >= 0 else "good",
                        "model": name,
                    }
                )
        fold_results.append(fold_row)

    summaries: dict[str, Any] = {}
    for name, scores in scores_by_model.items():
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Incomplete out-of-fold predictions for {name}")
        per_fold = [fold[name] for fold in fold_results]
        pooled = _metrics(labels, scores)
        # Each outer fold is a separately fitted classifier. Its decision
        # scores are not necessarily calibrated to the same scale, so ranking
        # examples across folds can make a pooled OOF AUC misleading.
        pooled["roc_auc"] = None
        summaries[name] = {
            "pooled_out_of_fold": pooled,
            "cell_type_macro": {
                metric: float(np.mean([row[metric] for row in per_fold if row[metric] is not None]))
                for metric in per_fold[0]
                if metric != "roc_auc" or any(row[metric] is not None for row in per_fold)
            },
        }

    primary_model_name = "inception_v3_logistic" if inception_only else "image_hog"
    summary = {
        "schema_version": "ooc_quality_audit_v1",
        "dataset": "Organ-on-a-Chip (OOC) Image Dataset",
        "source_record_url": SOURCE_RECORD_URL,
        "source_paper_url": SOURCE_PAPER_URL,
        "archive_url": ARCHIVE_URL,
        "archive_expected_md5": EXPECTED_ARCHIVE_MD5,
        "license_notes": {
            "zenodo_record_api": "CC-BY-4.0",
            "dataset_paper": "CC-BY-SA",
            "resolution": (
                "The two source statements differ. This audit does not redistribute "
                "images or model weights; cite both sources and do not re-host the archive."
            ),
        },
        "task": "expert-assessed good/bad OOC sample image-quality classification",
        "matched_labelled_images": len(samples),
        "quality_counts": {
            "good": int(np.count_nonzero(labels == 0)),
            "bad": int(np.count_nonzero(labels == 1)),
        },
        "cell_type_counts": dict(sorted(Counter(str(value) for value in groups).items())),
        "outer_validation": "leave-one-cell-type-out; cell type is not a model feature",
        "primary_model": (
            "inception_v3_logistic (frozen ImageNet Inception-v3 + balanced logistic regression)"
            if inception_only
            else "image_hog (fixed linear SVM, specified before evaluation)"
        ),
        "inception_only": inception_only,
        "hog_features_included": not inception_only,
        "primary_metric": "unweighted macro mean of the six held-out-cell-type ROC-AUC values",
        "pooled_roc_auc": (
            "Not reported: decision scores come from six separately fitted outer-fold models "
            "and are not assumed to share a calibrated scale."
        ),
        "ablation_models": [name for name in models_template if name != primary_model_name],
        "literature_comparison_models": (
            ["inception_v3_logistic", "inception_v3_plus_metadata"]
            if include_inception
            else []
        ),
        "inception_encoder": inception_info,
        "paper_comparison": (
            "George and Kenry, DOI 10.1021/cbe.5c00087: frozen Inception-v3 embeddings and supervised classifiers"
        ),
        "metadata_features": list(NUMERIC_COLUMNS),
        "image_features": (
            "Inception-v3 pooled 2048D embeddings"
            if inception_only
            else "64x64 grayscale HOG (9 orientations, 8x8 cells) plus nine fixed image-quality summary statistics"
        ),
        "models": summaries,
        "folds": fold_results,
        "limitations": [
            "The target is sample image quality, not toxicity or treatment response.",
            "Cell type is the only available grouping key; chip/experiment IDs are absent, so this is not experiment-held-out validation.",
            "Images are brightfield OOC samples from six non-neural cell lines; this does not validate neural OoC transfer.",
            "There are six outer groups; per-image confidence intervals would overstate independent sample size and are not reported.",
        ],
    }
    return summary, prediction_rows


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # Zenodo publishes an MD5 for transfer integrity.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--metadata-xlsx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--skip-archive-md5",
        action="store_true",
        help="Skip the slow transfer-integrity check (not recommended)",
    )
    parser.add_argument(
        "--include-inception",
        action="store_true",
        help="Also run the published-method-inspired frozen Inception-v3 comparisons (needs torch/torchvision and pretrained weights)",
    )
    parser.add_argument(
        "--inception-only",
        action="store_true",
        help="Run Inception, metadata and hybrid models without recomputing the slower HOG/SVM baselines",
    )
    parser.add_argument("--inception-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.inception_only and not args.include_inception:
        parser.error("--inception-only requires --include-inception")

    digest = None if args.skip_archive_md5 else _md5(args.images_zip)
    print("Transfer checksum verified" if digest is not None else "Transfer checksum skipped", flush=True)
    if digest is not None and digest.lower() != EXPECTED_ARCHIVE_MD5:
        raise ValueError(
            f"Dataset archive MD5 mismatch: got {digest}, expected {EXPECTED_ARCHIVE_MD5}"
        )
    rows = read_xlsx_table(args.metadata_xlsx)
    print(f"Loaded {len(rows)} datasheet rows; starting image audit", flush=True)
    summary, predictions = run_ooc_quality_audit(
        rows,
        args.images_zip,
        include_inception=args.include_inception,
        inception_only=args.inception_only,
        inception_batch_size=args.inception_batch_size,
    )
    summary["metadata_url"] = METADATA_URL
    summary["metadata_sha256"] = hashlib.sha256(args.metadata_xlsx.read_bytes()).hexdigest()
    summary["archive_md5_verified"] = digest is not None
    summary["archive_md5"] = digest
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_path = args.out / "per_image_predictions.csv"
    if predictions:
        import pandas as pd

        pd.DataFrame(predictions).to_csv(prediction_path, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
