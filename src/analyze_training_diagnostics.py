#!/usr/bin/env python3
"""Regenerate best-versus-final checkpoint diagnostics from a completed archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np


GROUPS = {
    "20M unweighted\n(seed 42)": ["capacity_20m_unweighted"],
    "83M unweighted\n(3 seeds)": [
        "main_83m_unweighted_s42", "main_83m_unweighted_s123", "main_83m_unweighted_s2026"
    ],
    "83M effective CE\n(3 seeds)": [
        "balanced_83m_effective_ce_s42", "balanced_83m_effective_ce_s123", "balanced_83m_effective_ce_s2026"
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_runs(archive_path):
    grouped = {key: [] for key in GROUPS}
    with ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if not n.startswith("__MACOSX/") and not PurePosixPath(n).name.startswith("._")]
        for label, jobs in GROUPS.items():
            for job in jobs:
                selected = sorted(
                    n for n in names if f"/{job}/fold_" in f"/{n}" and n.endswith(".json")
                )
                grouped[label].extend(json.loads(archive.read(name)) for name in selected)
    for label, runs in grouped.items():
        expected = 10 if label.startswith("20M") else 30
        if len(runs) != expected:
            raise RuntimeError(f"{label}: expected {expected} fold-runs, found {len(runs)}")
    return grouped


def summarize(grouped):
    summary = {}
    for label, runs in grouped.items():
        best = np.asarray([run["best_inner_validation"]["f1"] * 100 for run in runs])
        final = np.asarray([run["final_inner_validation"]["f1"] * 100 for run in runs])
        epoch = np.asarray([run["selected_epoch"] for run in runs])
        summary[label] = {
            "n_fold_runs": len(runs),
            "best_macro_f1_mean": float(best.mean()),
            "best_macro_f1_sd": float(best.std(ddof=0)),
            "final_macro_f1_mean": float(final.mean()),
            "final_macro_f1_sd": float(final.std(ddof=0)),
            "best_minus_final_pp_mean": float((best - final).mean()),
            "selected_epoch_median": float(np.median(epoch)),
            "selected_epoch_q1": float(np.percentile(epoch, 25)),
            "selected_epoch_q3": float(np.percentile(epoch, 75)),
        }
    return summary


def plot(grouped, summary, output):
    colors = ["#1f4e79", "#2e8b57", "#d98c10"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), constrained_layout=True)
    ax = axes[0]
    for (label, runs), color in zip(grouped.items(), colors):
        max_epoch = max(len(run["selection_history"]) for run in runs)
        matrix = np.full((len(runs), max_epoch), np.nan)
        for i, run in enumerate(runs):
            values = [history["val_macro_f1"] * 100 for history in run["selection_history"]]
            matrix[i, :len(values)] = values
            ax.plot(np.arange(1, len(values) + 1), values, color=color, alpha=0.10, linewidth=0.65)
        x = np.arange(1, max_epoch + 1)
        mean, sd, active = np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0), np.sum(~np.isnan(matrix), axis=0)
        keep = active >= max(3, int(np.ceil(len(runs) * 0.25)))
        ax.plot(x[keep], mean[keep], color=color, linewidth=2.1, label=label.replace("\n", " "))
        ax.fill_between(x[keep], (mean - sd)[keep], (mean + sd)[keep], color=color, alpha=0.12)
    ax.set_title("(a) Inner-validation histories"); ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1 (%)")
    ax.grid(alpha=0.22); ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axes[1]
    for j, ((label, runs), color) in enumerate(zip(grouped.items(), colors)):
        best = np.asarray([run["best_inner_validation"]["f1"] * 100 for run in runs])
        final = np.asarray([run["final_inner_validation"]["f1"] * 100 for run in runs])
        jitter = np.random.default_rng(100 + j).normal(0, 0.018, len(runs))
        for best_value, final_value, offset in zip(best, final, jitter):
            ax.plot([2 * j + offset, 2 * j + 0.72 + offset], [best_value, final_value], color=color, alpha=0.22, linewidth=0.7)
        ax.scatter(np.full(len(best), 2 * j) + jitter, best, s=12, color=color, alpha=0.58)
        ax.scatter(np.full(len(final), 2 * j + 0.72) + jitter, final, s=12, facecolor="white", edgecolor=color, alpha=0.75)
        ax.plot([2 * j, 2 * j + 0.72], [best.mean(), final.mean()], color=color, linewidth=3)
    ax.set_xticks([0.36, 2.36, 4.36], ["20M\nunweighted", "83M\nunweighted", "83M\neffective CE"])
    ax.set_ylabel("Inner-validation macro-F1 (%)"); ax.set_title("(b) Best checkpoint vs final epoch")
    ax.grid(axis="y", alpha=0.22); ax.text(0.02, 0.02, "Filled = best; open = final", transform=ax.transAxes, fontsize=7)

    ax = axes[2]
    epochs = [[run["selected_epoch"] for run in runs] for runs in grouped.values()]
    boxplot = ax.boxplot(epochs, patch_artist=True, widths=0.55, showfliers=True)
    for box, color in zip(boxplot["boxes"], colors):
        box.set_facecolor(color); box.set_alpha(0.6)
    ax.set_xticks([1, 2, 3], ["20M\nunweighted", "83M\nunweighted", "83M\neffective CE"])
    ax.set_ylabel("Selected epoch"); ax.set_title("(c) Checkpoint-selection epochs"); ax.grid(axis="y", alpha=0.22)
    for i, label in enumerate(GROUPS, 1):
        ax.text(i, ax.get_ylim()[1] * 0.96, f"Δ={summary[label]['best_minus_final_pp_mean']:.2f} pp", ha="center", va="top", fontsize=7, color=colors[i - 1])
    fig.savefig(output / "figure_s1_training_diagnostics.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    grouped = load_runs(Path(args.archive))
    summary = summarize(grouped)
    (output / "training_diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(grouped, summary, output)
    print(f"Generated checkpoint diagnostics from {sum(map(len, grouped.values()))} fold-runs in {output}")


if __name__ == "__main__":
    main()
