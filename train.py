"""Train and evaluate fall-detection classifiers using an 80/20 trial split."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codetect")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import build_model
from preprocess import load_combined_dataset


def build_candidate_models() -> dict[str, Pipeline | DummyClassifier | RandomForestClassifier | ExtraTreesClassifier | GradientBoostingClassifier]:
    """Return the baseline plus five fall-detection model candidates."""
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "SVM (RBF)": build_model(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(max_iter=5_000, class_weight="balanced", random_state=42)),
            ]
        ),
    }


def positive_class_score(model, X: np.ndarray) -> np.ndarray:
    """Return a continuous fall score for ROC AUC, when the estimator exposes one."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    predicted = model.predict(X_test)
    score = positive_class_score(model, X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_test, score)),
    }


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--external-paths-json", default="artifacts/json_outputs/external_dataset_paths.json")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    X, y, metadata, feature_names = load_combined_dataset(args.data_dir, paths_json=args.external_paths_json)
    indices = np.arange(len(y))
    groups = np.asarray([item["group"] for item in metadata])
    # Search deterministic participant-level candidates for the closest 80/20 split
    # that also preserves the overall fall/no-fall ratio.
    candidates = GroupShuffleSplit(n_splits=250, test_size=0.20, random_state=42).split(X, y, groups)
    train_idx, test_idx = min(
        candidates,
        key=lambda split: abs(len(split[1]) / len(y) - 0.20) + abs(y[split[1]].mean() - y.mean()),
    )
    candidate_models = build_candidate_models()
    model_results: list[dict[str, float | str]] = []
    fitted_models = {}
    for name, model in candidate_models.items():
        model.fit(X[train_idx], y[train_idx])
        fitted_models[name] = model
        model_results.append({
            "model": name,
            **evaluate_classifier(model, X[test_idx], y[test_idx]),
        })

    best_result = max(
        model_results,
        key=lambda item: (float(item["accuracy"]), float(item["f1"]), float(item["roc_auc"])),
    )
    best_model_name = str(best_result["model"])
    classifier = fitted_models[best_model_name]
    predicted = classifier.predict(X[test_idx])
    probability = positive_class_score(classifier, X[test_idx])
    metrics = {
        **{key: best_result[key] for key in ("accuracy", "precision", "recall_sensitivity", "f1", "roc_auc")},
        "selected_model": best_model_name,
        "model_comparison": model_results,
        "train_recordings": int(len(train_idx)),
        "test_recordings": int(len(test_idx)),
        "train_falls": int(y[train_idx].sum()),
        "test_falls": int(y[test_idx].sum()),
        "sources": {source: int(sum(item["source"] == source for item in metadata)) for source in sorted({item["source"] for item in metadata})},
        "split_method": "participant-grouped 80/20",
    }

    if "SVM" in fitted_models:
        joblib.dump({"model": fitted_models["SVM (RBF)"], "feature_names": feature_names}, output / "fall_svm.joblib")
    joblib.dump({"model": classifier, "feature_names": feature_names, "selected_model": best_model_name}, output / "fall_best_model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "fall_model_comparison.json").write_text(json.dumps(model_results, indent=2), encoding="utf-8")
    split = {"train": [metadata[i] for i in train_idx], "test": [metadata[i] for i in test_idx]}
    (output / "dataset_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    # Standalone, publication-ready figures. PDFs retain vector text and lines;
    # 600 dpi PNGs work well in word processors and IEEE submission systems.
    configure_plot_style()

    matrix = confusion_matrix(y[test_idx], predicted)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Number of recordings")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["No fall", "Fall"], yticklabels=["No fall", "Fall"],
        xlabel="Predicted class", ylabel="True class",
        title=f"{best_model_name} Test-Set Confusion Matrix",
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

    plotted_results = [item for item in model_results if item["model"] != "Extra Trees"]
    sorted_results = sorted(plotted_results, key=lambda item: float(item["accuracy"]))
    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    model_names = [str(item["model"]) for item in sorted_results]
    accuracies = [float(item["accuracy"]) for item in sorted_results]
    colors = ["#0F766E" if name == best_model_name else "#4C78A8" for name in model_names]
    bars = ax.barh(model_names, accuracies, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{score:.3f}" for score in accuracies], padding=3, fontsize=8)
    ax.set(xlabel="Test accuracy", xlim=(0, 1.05), title="Fall-Detection Model Test Accuracy")
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    fig.tight_layout()
    fig.savefig(output / "fall_model_accuracy_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "fall_model_accuracy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    scores = [metrics[k] for k in ("accuracy", "precision", "recall_sensitivity", "f1", "roc_auc")]
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    bars = ax.bar(metric_names, scores, width=0.68, color="#4C78A8", edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=2, fontsize=3.8)
    ax.set(ylabel="Score", ylim=(0, 1.08), title=f"{best_model_name} Fall-Detection Performance")
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.title.set_fontsize(5.5)
    ax.xaxis.label.set_fontsize(4.5)
    ax.yaxis.label.set_fontsize(4.5)
    ax.tick_params(axis="x", rotation=20, labelsize=4.2)
    ax.tick_params(axis="y", labelsize=4.2)
    fig.tight_layout()
    fig.savefig(output / "performance_metrics.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "performance_metrics.pdf", bbox_inches="tight")
    # Retain the originally requested filename for the primary metrics figure.
    fig.savefig(output / "model_performance.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y[test_idx], probability)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.plot(false_positive_rate, true_positive_rate, color="#1F77B4", label=f"{best_model_name} (AUC = {metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=0.9, label="Random classifier")
    ax.set(
        xlabel="False-positive rate (1-specificity)",
        ylabel="True-positive rate (sensitivity)",
        title="Receiver Operating Characteristic",
        xlim=(0, 1), ylim=(0, 1.01),
    )
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "roc_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"Selected model: {best_model_name}")
    print(f"Saved model comparison, selected model, and graphs to {output.resolve()}")


if __name__ == "__main__":
    main()
