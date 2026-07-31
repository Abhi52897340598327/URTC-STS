"""Generate weighted engineering/design decision matrices for CODetect models."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codetect")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("artifacts/json_outputs")


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    return sum(scores[name] * weights[name] for name in weights)


def render_matrix(
    rows: list[dict],
    criteria: list[tuple[str, str]],
    title: str,
    output_base: Path,
    selected_model: str,
) -> None:
    col_labels = ["Model", *[label for key, label in criteria], "Weighted Score"]
    cell_text = []
    weighted_col = len(col_labels) - 1

    for row in rows:
        cell_text.append([
            row["model"],
            *[f"{row[key]:.2f}" for key, label in criteria],
            f"{row['weighted_score']:.3f}",
        ])

    fig_width = 12.0
    fig_height = max(5.0, fig_width * (len(rows) + 1.6) / len(col_labels) * 0.58)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    col_widths = [0.24, *([0.12] * (len(col_labels) - 2)), 0.16]
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.16, 0.96, 0.70],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.35)

    selected_row_index = next(i for i, row in enumerate(rows, start=1) if row["model"] == selected_model)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(1.0)
        cell.PAD = 0.08
        if row_idx == 0:
            cell.set_facecolor("#111827")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(9.5)
            cell.get_text().set_wrap(True)
            cell.set_linewidth(1.4)
        elif row_idx == selected_row_index:
            cell.set_facecolor("#C7F9E5")
            cell.get_text().set_weight("bold")
        elif col_idx == weighted_col:
            cell.set_facecolor("#E0F2FE")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 == 0 else "white")

        if row_idx == selected_row_index and col_idx == weighted_col:
            cell.set_facecolor("#059669")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_linewidth(2.0)

        if row_idx > 0 and col_idx == 0:
            cell.get_text().set_ha("left")
            cell.get_text().set_weight("bold")

    ax.text(
        0.5,
        0.875,
        title,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.095,
        "Scores are normalized on a 1-5 scale; final score is the weighted sum.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#475569",
        transform=ax.transAxes,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fall_model_matrix() -> list[dict]:
    comparison = json.loads((OUTPUT_DIR / "fall_model_comparison.json").read_text(encoding="utf-8"))
    metrics = {item["model"]: item for item in comparison}

    selected_models = ["SVM (RBF)", "Random Forest", "Extra Trees", "Gradient Boosting"]
    weights = {
        "accuracy": 0.35,
        "recall_sensitivity": 0.25,
        "deployment_efficiency": 0.15,
        "mobile_deployability": 0.15,
        "interpretability": 0.10,
    }
    engineering_scores = {
        "SVM (RBF)": {
            "deployment_efficiency": 4.5,
            "mobile_deployability": 4.0,
            "interpretability": 3.0,
        },
        "Random Forest": {
            "deployment_efficiency": 3.0,
            "mobile_deployability": 3.0,
            "interpretability": 4.0,
        },
        "Extra Trees": {
            "deployment_efficiency": 3.0,
            "mobile_deployability": 3.0,
            "interpretability": 3.5,
        },
        "Gradient Boosting": {
            "deployment_efficiency": 4.0,
            "mobile_deployability": 4.0,
            "interpretability": 4.0,
        },
    }

    rows = []
    for model in selected_models:
        result = metrics[model]
        row_scores = {
            "accuracy": result["accuracy"] * 5,
            "recall_sensitivity": result["recall_sensitivity"] * 5,
            **engineering_scores[model],
        }
        rows.append({
            "model": model,
            **row_scores,
            "weighted_score": weighted_score(row_scores, weights),
        })
    return rows


def gru_architecture_matrix() -> list[dict]:
    weights = {
        "temporal_fit": 0.30,
        "on_device_latency": 0.25,
        "model_size": 0.15,
        "training_stability": 0.10,
        "implementation_maturity": 0.10,
        "interpretability": 0.10,
    }
    rows = [
        {
            "model": "Dense Baseline",
            "temporal_fit": 2.0,
            "on_device_latency": 5.0,
            "model_size": 5.0,
            "training_stability": 4.5,
            "implementation_maturity": 4.0,
            "interpretability": 4.0,
        },
        {
            "model": "1D CNN",
            "temporal_fit": 3.5,
            "on_device_latency": 4.5,
            "model_size": 4.0,
            "training_stability": 4.0,
            "implementation_maturity": 4.0,
            "interpretability": 3.0,
        },
        {
            "model": "Temporal CNN (TCN)",
            "temporal_fit": 4.0,
            "on_device_latency": 4.0,
            "model_size": 3.5,
            "training_stability": 4.0,
            "implementation_maturity": 3.5,
            "interpretability": 3.0,
        },
        {
            "model": "LSTM",
            "temporal_fit": 4.5,
            "on_device_latency": 3.0,
            "model_size": 3.0,
            "training_stability": 3.5,
            "implementation_maturity": 4.5,
            "interpretability": 2.5,
        },
        {
            "model": "GRU",
            "temporal_fit": 4.5,
            "on_device_latency": 4.0,
            "model_size": 4.0,
            "training_stability": 4.0,
            "implementation_maturity": 4.5,
            "interpretability": 2.5,
        },
    ]

    for row in rows:
        row["weighted_score"] = weighted_score(row, weights)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fall_rows = fall_model_matrix()
    fall_rows.sort(key=lambda item: item["weighted_score"], reverse=True)
    render_matrix(
        fall_rows,
        [
            ("accuracy", "Accuracy\n35%"),
            ("recall_sensitivity", "Recall\n25%"),
            ("deployment_efficiency", "Runtime\nEfficiency\n15%"),
            ("mobile_deployability", "Mobile\nDeploy.\n15%"),
            ("interpretability", "Interp.\n10%"),
        ],
        "Fall-Detection Engineering Decision Matrix",
        OUTPUT_DIR / "fall_engineering_decision_matrix",
        selected_model="Gradient Boosting",
    )

    gru_rows = gru_architecture_matrix()
    gru_rows.sort(key=lambda item: item["weighted_score"], reverse=True)
    render_matrix(
        gru_rows,
        [
            ("temporal_fit", "Temporal Fit\n30%"),
            ("on_device_latency", "On-Device\nLatency\n25%"),
            ("model_size", "Model Size\n15%"),
            ("training_stability", "Training\nStability\n10%"),
            ("implementation_maturity", "Impl.\nMaturity\n10%"),
            ("interpretability", "Interp.\n10%"),
        ],
        "CO Forecasting Architecture Decision Matrix",
        OUTPUT_DIR / "gru_engineering_decision_matrix",
        selected_model="GRU",
    )

    (OUTPUT_DIR / "engineering_decision_matrices.json").write_text(
        json.dumps(
            {
                "fall_detection": fall_rows,
                "co_forecasting": gru_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"fall_detection": fall_rows, "co_forecasting": gru_rows}, indent=2))


if __name__ == "__main__":
    main()
