#!/usr/bin/env python3
"""Regenerate manuscript Figures 4 and 5 from the completed result archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_JOB = "main_83m_unweighted_s42"
CLASS_CODES = ["0", "1", "2", "3"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job", default=DEFAULT_JOB)
    return parser.parse_args()


def load_folds(archive_path: Path, job: str):
    with ZipFile(archive_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if f"/{job}/fold_" in f"/{name}"
            and name.endswith(".json")
            and not name.startswith("__MACOSX/")
            and not PurePosixPath(name).name.startswith("._")
        ]
        folds = [json.loads(archive.read(name)) for name in sorted(names)]
    if len(folds) != 10:
        raise RuntimeError(f"Expected 10 fold records for {job}; found {len(folds)}")
    return folds


def annotate(axis, matrix, fontsize):
    totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    threshold = matrix.max() * 0.52
    for row in range(4):
        for column in range(4):
            colour = "white" if matrix[row, column] >= threshold else "#10263d"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n({percentages[row, column] * 100:.1f}%)",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=colour,
                fontweight="semibold",
            )


def style_axes(axis):
    axis.set_xticks(range(4), CLASS_CODES)
    axis.set_yticks(range(4), CLASS_CODES)
    axis.set_xlabel("Predicted class", fontweight="semibold")
    axis.set_ylabel("Actual class", fontweight="semibold")


def make_figures(folds, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = [np.asarray(fold["outer_test"]["confusion_matrix"], dtype=int) for fold in folds]

    aggregate = np.sum(matrices, axis=0)
    fig, axis = plt.subplots(figsize=(7.3, 6.0))
    image = axis.imshow(aggregate, cmap="Blues", aspect="equal")
    annotate(axis, aggregate, 11)
    style_axes(axis)
    colourbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colourbar.set_label("Trial count")
    fig.tight_layout()
    fig.savefig(output_dir / "figure4_reference_83m_aggregate_confusion.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(9.2, 18.0), constrained_layout=True)
    maximum = max(matrix.max() for matrix in matrices)
    for index, (axis, matrix) in enumerate(zip(axes.flat, matrices), start=1):
        image = axis.imshow(matrix, cmap="Blues", aspect="equal", vmin=0, vmax=maximum)
        annotate(axis, matrix, 7.8)
        style_axes(axis)
        axis.set_title(f"Fold {index}", fontsize=11, fontweight="semibold")
    colourbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02)
    colourbar.set_label("Trial count")
    fig.savefig(output_dir / "figure5_reference_83m_per_fold_confusions.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    folds = load_folds(Path(args.archive), args.job)
    make_figures(folds, Path(args.output_dir))
    print(f"Generated manuscript Figures 4 and 5 from {len(folds)} outer folds")


if __name__ == "__main__":
    main()
