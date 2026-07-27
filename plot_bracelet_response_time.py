"""Generate an IEEE-style chart of end-to-end bracelet response time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRIALS = np.arange(1, 14)
TIMES_SECONDS = np.array([4.2, 6.8, 4.5, 7.3, 6.5, 3.4, 6.9, 3.7, 5.7, 3.5, 3.9, 4.6, 4.0])


def main() -> None:
    output = Path("artifacts")
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # IEEE double-column width keeps all 13 trial labels legible.
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    bars = ax.bar(
        TRIALS,
        TIMES_SECONDS,
        width=0.68,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in TIMES_SECONDS], padding=2, fontsize=7.5)
    ax.set(
        xlabel="Trial number",
        ylabel="End-to-end response time (s)",
        title="Bracelet System End-to-End Response Time",
        xlim=(0.35, 13.65),
        ylim=(0, 8.0),
    )
    ax.set_xticks(TRIALS)
    ax.set_yticks(np.arange(0, 8.1, 1))
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(output / "bracelet_response_time.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / "bracelet_response_time.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
