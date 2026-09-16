#!/usr/bin/env python3
"""End-to-end synthetic check for reviewer result aggregation."""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def make_summary(error_offset=0):
    labels = np.tile(np.arange(4), 8)
    subjects = np.repeat(np.arange(1, 9), 4)
    predictions = labels.copy()
    if error_offset:
        predictions[::error_offset] = (predictions[::error_offset] + 1) % 4
    folds = []
    for fold_number, selection in enumerate((np.arange(16), np.arange(16, 32)), start=1):
        truth = labels[selection]
        pred = predictions[selection]
        accuracy = float(np.mean(truth == pred))
        train_subjects = [5, 6, 7] if fold_number == 1 else [1, 2, 3]
        validation_subjects = [8] if fold_number == 1 else [4]
        folds.append({
            "outer_fold": fold_number,
            "inner_train_subjects": train_subjects,
            "inner_validation_subjects": validation_subjects,
            "outer_test_subjects": sorted(np.unique(subjects[selection]).tolist()),
            "outer_test": {
                "accuracy": accuracy,
                "balanced_accuracy": accuracy,
                "f1": accuracy,
                "per_class_recall": [accuracy] * 4,
            },
            "outer_test_predictions": {
                "sample_indices": selection.tolist(),
                "subject_ids": subjects[selection].tolist(),
                "y_true": truth.tolist(),
                "y_pred": pred.tolist(),
            },
        })
    values = [fold["outer_test"]["accuracy"] for fold in folds]
    metric = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": 2}
    return {
        "args": {"model": "synthetic", "kernels": [7, 9, 11, 13], "tmin": 0.0, "tmax": 4.0},
        "class_names": ["rest", "left", "right", "feet"],
        "kernels": [7, 9, 11, 13],
        "epoch": {"tmin_seconds": 0.0, "tmax_seconds": 4.0},
        "compute": {"parameters": 1, "flops_per_sample_2x_macs": 2, "latency_ms_batch1_mean": 0.1},
        "aggregate": {
            "accuracy": metric,
            "balanced_accuracy": metric,
            "f1": metric,
            "per_class_recall": {name: {"mean": metric["mean"], "std": metric["std"]} for name in ("rest", "left", "right", "feet")},
        },
        "folds": folds,
    }


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "experiments"
        for name, offset in (("kernel_k7_9_11_13", 0), ("baseline_eegnet", 3)):
            output = root / name
            output.mkdir(parents=True)
            (output / "summary.json").write_text(json.dumps(make_summary(offset)))
        analysis = Path(directory) / "analysis"
        subprocess.run([
            sys.executable,
            str(ROOT / "src" / "analyze_reviewer_results.py"),
            "--results-root", str(root),
            "--output-dir", str(analysis),
            "--bootstrap-resamples", "100",
        ], check=True)
        required = (
            "all_subject_metrics.csv",
            "subject_distribution_summary.csv",
            "subject_bootstrap_95ci.csv",
            "expanded_wilcoxon_effect_sizes.csv",
            "fold_assignments.csv",
            "fold_level_predictions.csv",
            "fold_confusion_matrices.csv",
            "aggregate_confusion_matrices.csv",
        )
        assert all((analysis / name).exists() for name in required)
        with (analysis / "fold_level_predictions.csv").open() as handle:
            prediction_rows = list(csv.DictReader(handle))
        assert len(prediction_rows) == 64
        assert {row["run"] for row in prediction_rows} == {
            "baseline_eegnet",
            "kernel_k7_9_11_13",
        }
        with (analysis / "fold_confusion_matrices.csv").open() as handle:
            confusion_rows = list(csv.DictReader(handle))
        assert len(confusion_rows) == 64
        assert sum(int(row["count"]) for row in confusion_rows) == 64
        with (analysis / "expanded_wilcoxon_effect_sizes.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert all("holm_adjusted_p" in row for row in rows)
    print("Synthetic reviewer-analysis smoke check passed.")


if __name__ == "__main__":
    main()
