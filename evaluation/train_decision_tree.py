"""Compare a decision tree with a Mamdani fuzzy system on tactile slip data."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATASET_URL = (
    "https://media.githubusercontent.com/media/"
    "ARQ-CRISP/slip_detection_dataset_2021/main/data/"
)
DATASETS = {
    "spool_solder": "slipDataset_SpoolSolder_tactile.h5",
    "brush": "slipDataset_brush_tactile.h5",
    "screwdriver": "slipDataset_screwDriver_tactile.h5",
}
LABELS = {
    "tactile_data_raw",
    "tactile_slips_label",
}
VALID_LABELS = np.array([0, 1, 4, 5])
SLIP_LABELS = np.array([1, 5])
FEATURE_NAMES = ("shear_normal_ratio", "relative_shear_change")


def download_dataset(dataset_name: str, data_dir: Path) -> Path:
    """Download one Git LFS-backed HDF5 dataset if it is not present."""
    destination = data_dir / DATASETS[dataset_name]
    data_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not h5py.is_hdf5(destination):
            raise ValueError(
                f"{destination} is not a valid HDF5 file. Remove it and run again."
            )
        return destination

    url = DATASET_URL + DATASETS[dataset_name]
    print(f"Downloading {dataset_name} dataset from the public GitHub repository...")
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=data_dir, suffix=".download", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)
        if not h5py.is_hdf5(temporary_path):
            raise ValueError(f"Downloaded dataset is not a valid HDF5 file: {url}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def features_from_experiment(raw_data: np.ndarray) -> np.ndarray:
    """Summarize 18 three-axis sensor readings into two interpretable features."""
    if raw_data.ndim != 3 or raw_data.shape[:2] != (18, 3):
        raise ValueError(
            "Expected tactile_data_raw to have shape (18 pins, 3 axes, samples); "
            f"got {raw_data.shape}."
        )

    shear = np.hypot(raw_data[:, 0, :], raw_data[:, 1, :])
    normal_force = np.abs(raw_data[:, 2, :])
    mean_shear = np.mean(shear, axis=0)
    mean_normal = np.mean(normal_force, axis=0)

    shear_normal_ratio = mean_shear / np.maximum(mean_normal, 1e-8)
    shear_change = np.zeros_like(mean_shear)
    if mean_shear.size > 1:
        shear_change[1:] = np.mean(np.abs(np.diff(shear, axis=1)), axis=0)
    relative_shear_change = shear_change / np.maximum(mean_normal, 1e-8)

    features = np.column_stack((shear_normal_ratio, relative_shear_change))
    if not np.isfinite(features).all():
        raise ValueError("Tactile data produced non-finite features.")
    return features


def load_experiments(dataset_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load annotated slip samples and keep experiment IDs for group splitting."""
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []

    for dataset_path in dataset_paths:
        if not h5py.is_hdf5(dataset_path):
            raise ValueError(f"Not an HDF5 dataset: {dataset_path}")
        with h5py.File(dataset_path, "r") as handle:
            experiments: list[tuple[str, h5py.Group]] = []

            def collect_experiment(name: str, item: Any) -> None:
                if isinstance(item, h5py.Group) and LABELS.issubset(item.keys()):
                    experiments.append((name, item))

            handle.visititems(collect_experiment)
            for experiment_name, experiment in experiments:
                raw_data = np.asarray(experiment["tactile_data_raw"])
                raw_labels = np.asarray(experiment["tactile_slips_label"])
                if raw_data.shape[-1] != raw_labels.size:
                    raise ValueError(
                        f"Sample count mismatch in {dataset_path}:{experiment_name}."
                    )

                valid = np.isin(raw_labels, VALID_LABELS)
                if not valid.any():
                    continue
                features = features_from_experiment(raw_data)[valid]
                binary_labels = np.isin(raw_labels[valid], SLIP_LABELS).astype(
                    np.int8
                )
                group_name = f"{dataset_path.stem}/{experiment_name}"
                feature_parts.append(features)
                label_parts.append(binary_labels)
                group_parts.append(
                    np.full(binary_labels.size, group_name, dtype=object)
                )

    if not feature_parts:
        raise ValueError("No labeled experiments were found in the selected dataset.")
    return (
        np.concatenate(feature_parts),
        np.concatenate(label_parts),
        np.concatenate(group_parts),
    )


def grouped_train_validation_test_split(
    labels: np.ndarray, groups: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by complete experiment so adjacent time samples cannot leak."""
    outer_split = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state
    )
    train_validation, test = next(
        outer_split.split(np.zeros(labels.size), labels, groups)
    )
    validation_split = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=random_state + 1
    )
    train_relative, validation_relative = next(
        validation_split.split(
            np.zeros(train_validation.size),
            labels[train_validation],
            groups[train_validation],
        )
    )
    train = train_validation[train_relative]
    validation = train_validation[validation_relative]

    for name, indices in (
        ("training", train),
        ("validation", validation),
        ("test", test),
    ):
        if np.unique(labels[indices]).size != 2:
            raise ValueError(
                f"The {name} split does not contain both slip and no-slip samples. "
                "Choose --dataset all or try a different --seed."
            )
    return train, validation, test


def low_membership(values: np.ndarray) -> np.ndarray:
    return np.clip((0.6 - values) / 0.35, 0.0, 1.0)


def high_membership(values: np.ndarray) -> np.ndarray:
    return np.clip((values - 0.25) / 0.4, 0.0, 1.0)


def triangle(universe: np.ndarray, left: float, peak: float, right: float) -> np.ndarray:
    rising = np.ones_like(universe) if left == peak else (universe - left) / (peak - left)
    falling = np.ones_like(universe) if peak == right else (right - universe) / (right - peak)
    return np.clip(np.minimum(rising, falling), 0.0, 1.0)


class MamdaniSlipDetector:
    """Two-input Mamdani inference system with a centroid crisp output."""

    def __init__(self) -> None:
        self.feature_min: np.ndarray | None = None
        self.feature_max: np.ndarray | None = None
        self.output_universe = np.linspace(0.0, 1.0, 101)
        self.output_memberships = {
            "low": triangle(self.output_universe, 0.0, 0.0, 0.5),
            "medium": triangle(self.output_universe, 0.25, 0.5, 0.75),
            "high": triangle(self.output_universe, 0.5, 1.0, 1.0),
        }

    def fit(self, features: np.ndarray) -> MamdaniSlipDetector:
        self.feature_min = np.quantile(features, 0.01, axis=0)
        self.feature_max = np.quantile(features, 0.99, axis=0)
        self.feature_max = np.maximum(self.feature_max, self.feature_min + 1e-8)
        return self

    def predict_score(self, features: np.ndarray) -> np.ndarray:
        if self.feature_min is None or self.feature_max is None:
            raise RuntimeError("Fit the fuzzy detector before requesting predictions.")

        normalized = np.clip(
            (features - self.feature_min) / (self.feature_max - self.feature_min),
            0.0,
            1.0,
        )
        scores: list[np.ndarray] = []
        rules = (
            ("low", "low", "low"),
            ("low", "high", "medium"),
            ("high", "low", "low"),
            ("high", "high", "high"),
        )
        for start in range(0, normalized.shape[0], 4096):
            batch = normalized[start : start + 4096]
            memberships = {
                "low": low_membership(batch),
                "high": high_membership(batch),
            }
            aggregated = np.zeros((batch.shape[0], self.output_universe.size))
            for first, second, output in rules:
                firing_strength = np.minimum(
                    memberships[first][:, 0], memberships[second][:, 1]
                )
                clipped_output = np.minimum(
                    firing_strength[:, None],
                    self.output_memberships[output][None, :],
                )
                aggregated = np.maximum(aggregated, clipped_output)

            area = aggregated.sum(axis=1)
            if np.any(area <= 1e-12):
                raise RuntimeError("Mamdani rules produced an empty output set.")
            scores.append(
                (aggregated @ self.output_universe) / area
            )
        return np.concatenate(scores) if scores else np.empty(0)


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def calculate_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = scores >= threshold
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="spool_solder",
        help="Object dataset to use; 'all' downloads and combines all three objects.",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    dataset_paths = [
        download_dataset(dataset_name, args.data_dir)
        for dataset_name in dataset_names
    ]
    features, labels, groups = load_experiments(dataset_paths)
    train, validation, test = grouped_train_validation_test_split(
        labels, groups, args.seed
    )

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=args.seed,
    )
    tree.fit(features[train], labels[train])
    tree_validation_scores = tree.predict_proba(features[validation])[:, 1]
    tree_threshold = best_f1_threshold(labels[validation], tree_validation_scores)
    tree_test_scores = tree.predict_proba(features[test])[:, 1]

    fuzzy = MamdaniSlipDetector().fit(features[train])
    fuzzy_validation_scores = fuzzy.predict_score(features[validation])
    fuzzy_threshold = best_f1_threshold(labels[validation], fuzzy_validation_scores)
    fuzzy_test_scores = fuzzy.predict_score(features[test])

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for name, scores, threshold in (
        ("decision_tree", tree_test_scores, tree_threshold),
        ("mamdani_fuzzy", fuzzy_test_scores, fuzzy_threshold),
    ):
        metrics = calculate_metrics(labels[test], scores, threshold)
        result_rows.append(
            {
                "model": name,
                "test_samples": int(test.size),
                "test_experiments": int(np.unique(groups[test]).size),
                **metrics,
            }
        )
        print(
            f"{name:14s} "
            f"F1={metrics['f1']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.3f} "
            f"PR-AUC={metrics['average_precision']:.3f} "
            f"threshold={threshold:.3f}"
        )

    for sample_index, label, tree_score, fuzzy_score in zip(
        test, labels[test], tree_test_scores, fuzzy_test_scores, strict=True
    ):
        prediction_rows.append(
            {
                "experiment": str(groups[sample_index]),
                "true_slip": int(label),
                "decision_tree_score": float(tree_score),
                "mamdani_fuzzy_score": float(fuzzy_score),
            }
        )
    for row in prediction_rows:
        row["decision_tree_prediction"] = int(
            row["decision_tree_score"] >= tree_threshold
        )
        row["mamdani_fuzzy_prediction"] = int(
            row["mamdani_fuzzy_score"] >= fuzzy_threshold
        )

    write_csv(args.output_dir / "metrics.csv", result_rows)
    write_csv(args.output_dir / "test_predictions.csv", prediction_rows)
    print(f"Loaded {features.shape[0]:,} valid samples from {np.unique(groups).size} experiments.")
    print(f"Metrics: {args.output_dir / 'metrics.csv'}")
    print(f"Test predictions: {args.output_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
