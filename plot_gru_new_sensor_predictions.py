from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codetect")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "json_outputs"
PREDICTIONS_CSV = OUTPUT_DIR / "gru_new_sensor_predictions.csv"
PREDICTIONS_JSON = OUTPUT_DIR / "gru_new_sensor_predictions.json"
PNG_PATH = OUTPUT_DIR / "gru_new_sensor_actual_vs_predicted.png"
PDF_PATH = OUTPUT_DIR / "gru_new_sensor_actual_vs_predicted.pdf"


plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
    }
)


def main() -> None:
    df = pd.read_csv(PREDICTIONS_CSV)
    with PREDICTIONS_JSON.open() as f:
        summary = json.load(f)

    lower = summary.get("eligibleTargetMinimumCoPpm")
    upper = summary["maximumCoPpmIncluded"]
    if lower is not None and (df["actualCoPpm"] < lower).any():
        raise ValueError(f"Test targets must not be below {lower} ppm")
    if (df["actualCoPpm"] > upper).any():
        raise ValueError(f"Test targets must not be above {upper} ppm")

    x = range(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=180)
    ax.plot(
        x,
        df["actualCoPpm"],
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="Actual CO (test target)",
    )
    ax.plot(
        x,
        df["predictedCoPpm"],
        color="#d62728",
        linewidth=2.5,
        marker="s",
        markersize=4.5,
        label="GRU prediction",
    )

    change_indices = df.index[df["sourceFile"].ne(df["sourceFile"].shift())].tolist()
    for idx in change_indices[1:]:
        ax.axvline(idx + 0.5, color="#777777", linewidth=1.2, alpha=0.35)

    ax.set_title("GRU CO Forecast: Actual vs Predicted Test Data")
    ax.set_xlabel("Test data point index")
    ax.set_ylabel("CO concentration (ppm)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(
        json.dumps(
            {
                "png": str(PNG_PATH),
                "pdf": str(PDF_PATH),
                "testPointCount": len(df),
                "actualMinPpm": float(df["actualCoPpm"].min()),
                "actualMaxPpm": float(df["actualCoPpm"].max()),
                "predictedMinPpm": float(df["predictedCoPpm"].min()),
                "predictedMaxPpm": float(df["predictedCoPpm"].max()),
                "testMae": summary["testMae"],
                "testRmse": summary["testRmse"],
                "testR2": summary["testR2"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
