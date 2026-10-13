"""Small reproducible G3 diagnostic plots from frozen prediction CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare import OUT


def load_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    folder = OUT / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    for name, source in (("development", OUT / "development_candidate_predictions.csv"),
                         ("confirmation", OUT / "confirmation_candidate_results.csv")):
        if not source.exists():
            continue
        rows = load_csv(source)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        all_values = np.asarray([float(row["force_next_max_N"]) for row in rows])
        maximum = max(float(np.max(all_values)), 1.0)
        for ax, model in zip(axes, ("M0_behavior", "M1_counterfactual")):
            subset = [row for row in rows if row["model"] == model]
            y = np.asarray([float(row["force_next_max_N"]) for row in subset])
            p = np.asarray([float(row["predicted_force_next_max_N"]) for row in subset])
            ax.scatter(y, p, s=4, alpha=.25)
            ax.plot([0, maximum], [0, maximum], color="black", linewidth=1)
            ax.set(xlabel="True next-interval force (nominal N)", ylabel="Predicted force (nominal N)", title=model)
        fig.suptitle(f"{name.title()} candidate-force predictions")
        fig.savefig(folder / f"{name}_candidate_scatter.png", dpi=160)
        plt.close(fig)
    for name, source in (("development", OUT / "development_candidate_metrics.csv"),
                         ("confirmation", OUT / "confirmation_state_metrics.csv")):
        if not source.exists():
            continue
        rows = load_csv(source)
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for model in ("M0_behavior", "M1_counterfactual"):
            value = [float(row["spearman"]) for row in rows if row["model"] == model and row["spearman"] not in ("", "None")]
            ax.hist(value, bins=np.linspace(-1, 1, 21), alpha=.5, label=model)
        ax.set(xlabel="Within-state Spearman", ylabel="States", title=f"{name.title()} candidate ranking")
        ax.legend()
        fig.savefig(folder / f"{name}_within_state_spearman.png", dpi=160)
        plt.close(fig)
    print(f"G3 plots saved in {folder}")


if __name__ == "__main__":
    main()
