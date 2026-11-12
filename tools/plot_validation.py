"""Plot published OOS references alongside measured checkpoint reevaluations.

Usage:
    python tools/plot_validation.py --output docs/verification/oos_comparison.svg

The reference and measurement files remain separate. No metric is fitted,
smoothed, or replaced by a reference value.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def plot_comparison(output):
    """Render Figure 5 references and all five measured neighbor settings.

    Args:
        output (Path): SVG, PDF, or PNG destination supported by Matplotlib.

    Returns:
        Path: The written figure path.
    """
    targets = json.loads((ROOT / "config/paper_targets.json").read_text())["oos"]
    neighbors = targets["k_values"]
    positions = np.arange(len(neighbors))
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "ssgt-oos-validation"})
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for row, dataset in enumerate(("WUT", "KGBD")):
        path = ROOT / "docs/verification" / (dataset.lower() + "_oos_summary.json")
        measured = json.loads(path.read_text())["metrics"]
        for column, metric in enumerate(("AUROC", "FPR95")):
            ax = axes[row, column]
            mean_key, std_key = metric + "_mean", metric + "_sample_std"
            ax.errorbar(positions - 0.06, targets[dataset][mean_key],
                        yerr=targets[dataset][std_key], marker="o", markersize=4,
                        capsize=3, color="#52677f", label="Paper reference")
            ax.errorbar(positions + 0.06,
                        [measured[str(k)][mean_key] for k in neighbors],
                        yerr=[measured[str(k)][std_key] for k in neighbors],
                        marker="s", markersize=4, capsize=3, color="#c35d2b",
                        label="Archived weights, fresh evaluation")
            ax.set_title(dataset + " · " + metric)
            ax.set_ylabel(metric + " (%)")
            ax.set_xticks(positions)
            ax.set_xticklabels(neighbors)
            ax.grid(axis="y", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 1:
                ax.set_xlabel("Number of neighbors K")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.text(0.5, 0.015,
                "Mean ± sample standard deviation over five recovered identity splits; "
                "checkpoint inference, not retraining.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None} if output.suffix.lower() == ".svg" else None
    figure.savefig(output, dpi=180, metadata=metadata)
    plt.close(figure)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/verification/oos_comparison.svg")
    print(plot_comparison(parser.parse_args().output))
