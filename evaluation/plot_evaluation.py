"""Create report-ready figures from the held-out model comparison results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


ROOT = Path(__file__).resolve().parent
MODEL_COLUMNS = {
    "Decision tree": ("decision_tree_score", "decision_tree_prediction", "#0072B2"),
    "Mamdani fuzzy": ("mamdani_fuzzy_score", "mamdani_fuzzy_prediction", "#D55E00"),
}
METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1 score"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("average_precision", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation file not found: {path}")
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"Evaluation file has no data rows: {path}")
    return rows


def validate_predictions(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    required = {"true_slip", *(column for cols in MODEL_COLUMNS.values() for column in cols[:2])}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Prediction file is missing columns: {', '.join(sorted(missing))}")

    labels = np.asarray([int(row["true_slip"]) for row in rows], dtype=np.int8)
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("The test predictions must include both binary classes 0 and 1.")

    model_data: dict[str, np.ndarray] = {}
    for model_name, (score_column, prediction_column, _) in MODEL_COLUMNS.items():
        scores = np.asarray([float(row[score_column]) for row in rows])
        predictions = np.asarray(
            [int(row[prediction_column]) for row in rows], dtype=np.int8
        )
        if not np.isfinite(scores).all() or not np.isin(predictions, [0, 1]).all():
            raise ValueError(f"Invalid scores or predictions found for {model_name}.")
        model_data[model_name] = np.column_stack((scores, predictions))
    return labels, model_data


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_precision_recall(
    labels: np.ndarray, model_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    prevalence = float(labels.mean())
    axis.axhline(
        prevalence,
        color="#777777",
        linestyle=":",
        linewidth=1.5,
        label=f"No-skill baseline ({prevalence:.3f})",
    )
    for model_name, data in model_data.items():
        scores = data[:, 0]
        precision, recall, _ = precision_recall_curve(labels, scores)
        average_precision = average_precision_score(labels, scores)
        color = MODEL_COLUMNS[model_name][2]
        axis.plot(
            recall,
            precision,
            color=color,
            linewidth=2.2,
            label=f"{model_name} (AP = {average_precision:.3f})",
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Recall (slip detection rate)",
        ylabel="Precision (fraction of alarms that are slips)",
        title="Precision–recall curves",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best", frameon=True)
    save_figure(figure, output_dir / "precision_recall")


def plot_roc(
    labels: np.ndarray, model_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    axis.plot(
        [0, 1],
        [0, 1],
        color="#777777",
        linestyle=":",
        linewidth=1.5,
        label="Chance baseline (AUC = 0.500)",
    )
    for model_name, data in model_data.items():
        scores = data[:, 0]
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        color = MODEL_COLUMNS[model_name][2]
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            linewidth=2.2,
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate (recall)",
        title="Receiver operating characteristic",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=True)
    save_figure(figure, output_dir / "roc_curve")


def plot_confusion_matrices(
    labels: np.ndarray, model_data: dict[str, np.ndarray], output_dir: Path
) -> None:
    figure, axes = plt.subplots(
        1, len(model_data), figsize=(10.5, 4.8), layout="constrained"
    )
    image = None
    for axis, (model_name, data) in zip(axes, model_data.items(), strict=True):
        predictions = data[:, 1].astype(np.int8)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        row_totals = matrix.sum(axis=1, keepdims=True)
        row_percentages = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(2):
            for column in range(2):
                count = matrix[row, column]
                percentage = row_percentages[row, column] * 100
                axis.text(
                    column,
                    row,
                    f"{count:,}\n({percentage:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if count > matrix.max() * 0.55 else "#222222",
                    fontsize=11,
                )
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No slip", "Slip"],
            yticklabels=["No slip", "Slip"],
            xlabel="Predicted label",
            ylabel="Actual label",
            title=model_name,
        )
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label="Number of samples")
    figure.suptitle("Held-out test confusion matrices (row-normalized % in parentheses)")
    save_figure(figure, output_dir / "confusion_matrices")


def plot_metric_comparison(metrics_path: Path, output_dir: Path) -> None:
    rows = load_csv(metrics_path)
    metrics_by_model = {row["model"]: row for row in rows}
    expected_models = {"decision_tree", "mamdani_fuzzy"}
    if not expected_models.issubset(metrics_by_model):
        raise ValueError("Metrics file must contain decision_tree and mamdani_fuzzy rows.")

    values = np.asarray(
        [
            [float(metrics_by_model["decision_tree"][key]) for key, _ in METRICS],
            [float(metrics_by_model["mamdani_fuzzy"][key]) for key, _ in METRICS],
        ]
    )
    figure, axis = plt.subplots(figsize=(10.2, 5.4), layout="constrained")
    x = np.arange(len(METRICS))
    width = 0.36
    for index, (model_name, color) in enumerate(
        (("Decision tree", MODEL_COLUMNS["Decision tree"][2]),
         ("Mamdani fuzzy", MODEL_COLUMNS["Mamdani fuzzy"][2]))
    ):
        bars = axis.bar(
            x + (index - 0.5) * width,
            values[index],
            width,
            color=color,
            label=model_name,
        )
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axis.set(
        xticks=x,
        xticklabels=[label for _, label in METRICS],
        ylim=(0, 1.12),
        ylabel="Score",
        title="Held-out test metrics",
    )
    axis.tick_params(axis="x", labelrotation=15)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", frameon=True)
    save_figure(figure, output_dir / "metric_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "results" / "test_predictions.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "results" / "metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "figures",
    )
    args = parser.parse_args()

    labels, model_data = validate_predictions(load_csv(args.predictions))
    plot_precision_recall(labels, model_data, args.output_dir)
    plot_roc(labels, model_data, args.output_dir)
    plot_confusion_matrices(labels, model_data, args.output_dir)
    plot_metric_comparison(args.metrics, args.output_dir)
    print(f"Saved report figures (PNG and PDF) to {args.output_dir}")


if __name__ == "__main__":
    main()
