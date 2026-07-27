"""Train and evaluate the fall-detection SVM using an 80/20 trial split."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from model import build_model
from preprocess import load_combined_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    X, y, metadata, feature_names = load_combined_dataset(args.data_dir)
    indices = np.arange(len(y))
    groups = np.asarray([item["group"] for item in metadata])
    # Search deterministic participant-level candidates for the closest 80/20 split
    # that also preserves the overall fall/no-fall ratio.
    candidates = GroupShuffleSplit(n_splits=250, test_size=0.20, random_state=42).split(X, y, groups)
    train_idx, test_idx = min(
        candidates,
        key=lambda split: abs(len(split[1]) / len(y) - 0.20) + abs(y[split[1]].mean() - y.mean()),
    )
    classifier = build_model()
    classifier.fit(X[train_idx], y[train_idx])
    predicted = classifier.predict(X[test_idx])
    probability = classifier.predict_proba(X[test_idx])[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y[test_idx], predicted, average="binary", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y[test_idx], predicted)),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y[test_idx], probability)),
        "train_recordings": int(len(train_idx)),
        "test_recordings": int(len(test_idx)),
        "train_falls": int(y[train_idx].sum()),
        "test_falls": int(y[test_idx].sum()),
        "sources": {source: int(sum(item["source"] == source for item in metadata)) for source in sorted({item["source"] for item in metadata})},
        "split_method": "participant-grouped 80/20",
    }

    joblib.dump({"model": classifier, "feature_names": feature_names}, output / "fall_svm.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    split = {"train": [metadata[i] for i in train_idx], "test": [metadata[i] for i in test_idx]}
    (output / "dataset_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    # Standalone, publication-ready figures. PDFs retain vector text and lines;
    # 600 dpi PNGs work well in word processors and IEEE submission systems.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    matrix = confusion_matrix(y[test_idx], predicted)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Number of recordings")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["No fall", "Fall"], yticklabels=["No fall", "Fall"],
        xlabel="Predicted class", ylabel="True class",
        title="SVM Test-Set Confusion Matrix",
    )
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:d}", ha="center", va="center",
                    color="white" if matrix[row, column] > threshold else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    scores = [metrics[k] for k in ("accuracy", "precision", "recall_sensitivity", "f1", "roc_auc")]
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    bars = ax.bar(metric_names, scores, width=0.68, color="#4C78A8", edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=2, fontsize=7.5)
    ax.set(ylabel="Score", ylim=(0, 1.08), title="SVM Fall-Detection Performance")
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "performance_metrics.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "performance_metrics.pdf", bbox_inches="tight")
    # Retain the originally requested filename for the primary metrics figure.
    fig.savefig(output / "model_performance.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y[test_idx], probability)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.plot(false_positive_rate, true_positive_rate, color="#1F77B4", label=f"SVM (AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=0.9, label="Random classifier")
    ax.set(
        xlabel="False-positive rate (1-specificity)",
        ylabel="True-positive rate (sensitivity)",
        title="Receiver Operating Characteristic",
        xlim=(0, 1), ylim=(0, 1.01),
    )
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.75)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "roc_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model and graph to {output.resolve()}")


if __name__ == "__main__":
    main()
