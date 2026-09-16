#!/usr/bin/env python3
"""Create subject-level, bootstrap, ablation, and paired reviewer tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import confusion_matrix


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
CLASSIFIER_BASELINES = ("baseline_eegnet", "baseline_shallowconvnet", "baseline_deepconvnet1d", "baseline_resnet1d", "baseline_csp_lda")


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and not fieldnames:
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def metrics_from_confusions(cms):
    cms = np.asarray(cms, dtype=float)
    diagonal = np.diagonal(cms, axis1=-2, axis2=-1)
    row_sum = cms.sum(axis=-1)
    col_sum = cms.sum(axis=-2)
    accuracy = safe_divide(diagonal.sum(axis=-1), cms.sum(axis=(-2, -1)))
    balanced = safe_divide(diagonal, row_sum).mean(axis=-1)
    macro_f1 = safe_divide(2 * diagonal, 2 * diagonal + (col_sum - diagonal) + (row_sum - diagonal)).mean(axis=-1)
    return {"accuracy": accuracy, "balanced_accuracy": balanced, "macro_f1": macro_f1}


def predictions_by_subject(summary):
    by_subject = defaultdict(lambda: {"y_true": [], "y_pred": [], "sample_indices": []})
    for fold in summary["folds"]:
        predictions = fold.get("outer_test_predictions")
        if not predictions:
            raise ValueError("Fold predictions are missing; rerun this experiment with the reviewer package.")
        lengths = {len(predictions[key]) for key in ("subject_ids", "y_true", "y_pred", "sample_indices")}
        if len(lengths) != 1:
            raise ValueError("Prediction fields have inconsistent lengths.")
        for subject, truth, pred, sample in zip(
            predictions["subject_ids"], predictions["y_true"], predictions["y_pred"], predictions["sample_indices"]
        ):
            item = by_subject[int(subject)]
            item["y_true"].append(int(truth))
            item["y_pred"].append(int(pred))
            item["sample_indices"].append(int(sample))
    return dict(sorted(by_subject.items()))


def validate_subject_disjoint_summary(run_name, summary):
    test_fold_by_subject = {}
    all_sample_indices = []
    for fold in summary["folds"]:
        test = set(fold.get("outer_test_subjects", []))
        train = set(fold.get("inner_train_subjects", fold.get("outer_train_subjects", [])))
        validation = set(fold.get("inner_validation_subjects", []))
        if test & train or test & validation or train & validation:
            raise ValueError(f"{run_name}: subject overlap in outer fold {fold['outer_fold']}")
        for subject in test:
            if subject in test_fold_by_subject:
                raise ValueError(f"{run_name}: subject {subject} appears in multiple outer-test folds")
            test_fold_by_subject[subject] = fold["outer_fold"]
        predictions = fold.get("outer_test_predictions", {})
        all_sample_indices.extend(predictions.get("sample_indices", []))
    if len(all_sample_indices) != len(set(all_sample_indices)):
        raise ValueError(f"{run_name}: an outer-test sample index appears more than once")
    expected = summary.get("subjects")
    if expected is not None and len(test_fold_by_subject) != int(expected):
        raise ValueError(
            f"{run_name}: outer-test coverage contains {len(test_fold_by_subject)} subjects; expected {expected}"
        )


def subject_rows_and_confusions(run_name, summary):
    by_subject = predictions_by_subject(summary)
    n_classes = len(summary["class_names"])
    rows, cms = [], []
    for subject_id, values in by_subject.items():
        cm = confusion_matrix(values["y_true"], values["y_pred"], labels=np.arange(n_classes))
        calculated = metrics_from_confusions(cm[None, ...])
        rows.append({
            "run": run_name,
            "subject_id": subject_id,
            "n_trials": len(values["y_true"]),
            "held_out_sample_signature": hashlib.sha256(
                json.dumps(sorted(zip(values["sample_indices"], values["y_true"]))).encode("utf-8")
            ).hexdigest(),
            "accuracy": float(calculated["accuracy"][0]),
            "balanced_accuracy": float(calculated["balanced_accuracy"][0]),
            "macro_f1": float(calculated["macro_f1"][0]),
        })
        cms.append(cm)
    return rows, np.asarray(cms), np.asarray(sorted(by_subject), dtype=int)


def subject_block_bootstrap(cms, n_boot, seed):
    rng = np.random.default_rng(seed)
    n_subjects = len(cms)
    point = metrics_from_confusions(cms.sum(axis=0, keepdims=True))
    distributions = {name: [] for name in METRICS}
    for start in range(0, n_boot, 500):
        count = min(500, n_boot - start)
        sampled = rng.integers(0, n_subjects, size=(count, n_subjects))
        combined = cms[sampled].sum(axis=1)
        values = metrics_from_confusions(combined)
        for name in METRICS:
            distributions[name].append(values[name])
    rows = []
    for name in METRICS:
        values = np.concatenate(distributions[name])
        low, high = np.percentile(values, [2.5, 97.5])
        rows.append({
            "metric": name,
            "estimate": float(point[name][0]),
            "ci_2_5": float(low),
            "ci_97_5": float(high),
            "n_subjects": n_subjects,
            "bootstrap_resamples": n_boot,
            "seed": seed,
            "method": "subject-block percentile bootstrap of pooled out-of-fold predictions",
        })
    return rows


def distribution_rows(run_name, subject_rows):
    output = []
    for metric in METRICS:
        values = np.asarray([row[metric] for row in subject_rows], dtype=float)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        output.append({
            "run": run_name,
            "metric": metric,
            "n_subjects": len(values),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        })
    return output


def plot_subjects(path, run_name, subject_rows):
    values = [[100 * row[name] for row in subject_rows] for name in METRICS]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    labels = ["Accuracy", "Balanced\naccuracy", "Macro-F1"]
    try:
        box = axis.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True)
    except TypeError:  # Matplotlib 3.8 used the older keyword.
        box = axis.boxplot(values, labels=labels, patch_artist=True, showfliers=True)
    for patch, color in zip(box["boxes"], ("#4776a8", "#d28b18", "#2e7d5a")):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_ylabel("Held-out subject performance (%)")
    axis.set_title(f"Subject-level distribution: {run_name}")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def rank_biserial(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isclose(diff, 0.0)]
    if not len(diff):
        return 0.0
    ranks = rankdata(np.abs(diff))
    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm_adjust(rows):
    for metric in METRICS:
        family = [row for row in rows if row["metric"] == metric]
        order = sorted(range(len(family)), key=lambda index: family[index]["raw_p"])
        running = 0.0
        for rank, index in enumerate(order):
            adjusted = min(1.0, (len(family) - rank) * family[index]["raw_p"])
            running = max(running, adjusted)
            family[index]["holm_adjusted_p"] = running


def expanded_comparisons(subject_tables, reference_name, comparators, n_boot, seed):
    reference = {row["subject_id"]: row for row in subject_tables[reference_name]}
    rng = np.random.default_rng(seed)
    rows = []
    for comparator_name in comparators:
        comparator = {row["subject_id"]: row for row in subject_tables[comparator_name]}
        common = sorted(set(reference) & set(comparator))
        if not common:
            continue
        mismatched = [
            subject for subject in common
            if reference[subject]["held_out_sample_signature"] != comparator[subject]["held_out_sample_signature"]
        ]
        # Temporal-window experiments can legitimately contain slightly
        # different boundary-trial sets. Pair those comparisons by held-out
        # subject, which is the primary evaluation unit, and record whether the
        # underlying trial sets were identical. Kernel/baseline comparisons on
        # the same epoch retain exact trial-level matching.
        samples_identical = not mismatched
        for metric in METRICS:
            difference = np.asarray([reference[s][metric] - comparator[s][metric] for s in common], dtype=float)
            sampled = rng.integers(0, len(difference), size=(n_boot, len(difference)))
            boot_mean = difference[sampled].mean(axis=1)
            low, high = np.percentile(boot_mean, [2.5, 97.5])
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append({
                "reference": reference_name,
                "comparator": comparator_name,
                "metric": metric,
                "pairing_unit": "held-out subject",
                "held_out_samples_identical": samples_identical,
                "n_paired_subjects": len(common),
                "reference_mean": float(np.mean([reference[s][metric] for s in common])),
                "comparator_mean": float(np.mean([comparator[s][metric] for s in common])),
                "mean_difference": float(difference.mean()),
                "median_difference": float(np.median(difference)),
                "mean_difference_ci_2_5": float(low),
                "mean_difference_ci_97_5": float(high),
                "rank_biserial_correlation": rank_biserial(difference),
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
            })
    holm_adjust(rows)
    return rows


def expanded_fold_comparisons(summaries, reference_name, comparators, n_boot, seed):
    """Secondary paired-fold summary; subject-level inference remains primary."""
    rng = np.random.default_rng(seed)
    reference = {int(fold["outer_fold"]): fold for fold in summaries[reference_name]["folds"]}
    rows = []
    metric_keys = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1",
    }
    for comparator_name in comparators:
        comparator = {int(fold["outer_fold"]): fold for fold in summaries[comparator_name]["folds"]}
        common = sorted(set(reference) & set(comparator))
        if not common:
            continue
        subject_sets_identical = all(
            set(reference[index].get("outer_test_subjects", []))
            == set(comparator[index].get("outer_test_subjects", []))
            for index in common
        )
        if not subject_sets_identical:
            # Different epoch boundaries can change per-subject trial counts,
            # which in turn can change GroupKFold's fold composition. Such
            # runs remain valid for primary subject-level pairing but are not
            # falsely presented as paired-fold comparisons.
            continue
        for metric, key in metric_keys.items():
            reference_values = np.asarray([reference[index]["outer_test"][key] for index in common], dtype=float)
            comparator_values = np.asarray([comparator[index]["outer_test"][key] for index in common], dtype=float)
            difference = reference_values - comparator_values
            sampled = rng.integers(0, len(difference), size=(n_boot, len(difference)))
            boot_mean = difference[sampled].mean(axis=1)
            low, high = np.percentile(boot_mean, [2.5, 97.5])
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            rows.append({
                "reference": reference_name,
                "comparator": comparator_name,
                "metric": metric,
                "pairing_unit": "outer subject-wise fold",
                "n_paired_folds": len(common),
                "reference_mean": float(reference_values.mean()),
                "comparator_mean": float(comparator_values.mean()),
                "mean_difference": float(difference.mean()),
                "median_difference": float(np.median(difference)),
                "mean_difference_ci_2_5": float(low),
                "mean_difference_ci_97_5": float(high),
                "rank_biserial_correlation": rank_biserial(difference),
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
                "interpretation_note": "Secondary evaluation summary; n=10 yields discrete Wilcoxon p-values. Emphasize effect size and CI; subject-level results are primary.",
            })
    holm_adjust(rows)
    return rows


def experiment_row(run_name, summary):
    args = summary.get("args", {})
    compute = summary.get("compute", {})
    aggregate = summary["aggregate"]
    return {
        "run": run_name,
        "model": args.get("model", "csp_lda"),
        "kernels": "+".join(map(str, summary.get("kernels", args.get("kernels", [])))),
        "tmin_seconds": summary.get("epoch", {}).get("tmin_seconds", args.get("tmin")),
        "tmax_seconds": summary.get("epoch", {}).get("tmax_seconds", args.get("tmax")),
        "parameters": compute.get("parameters"),
        "flops_per_sample": compute.get("flops_per_sample_2x_macs"),
        "latency_ms_batch1": compute.get("latency_ms_batch1_mean"),
        "accuracy_fold_mean": aggregate["accuracy"]["mean"],
        "accuracy_fold_sd": aggregate["accuracy"]["std"],
        "balanced_accuracy_fold_mean": aggregate["balanced_accuracy"]["mean"],
        "balanced_accuracy_fold_sd": aggregate["balanced_accuracy"]["std"],
        "macro_f1_fold_mean": aggregate["f1"]["mean"],
        "macro_f1_fold_sd": aggregate["f1"]["std"],
    }


def fold_assignment_rows(run_name, summary):
    rows = []
    for fold in summary["folds"]:
        test_subjects = fold.get("outer_test_subjects", [])
        has_inner_split = "inner_train_subjects" in fold
        inner_train = fold.get("inner_train_subjects", fold.get("outer_train_subjects", []))
        inner_validation = fold.get("inner_validation_subjects", [])
        for role, subjects in (
            ("inner_train" if has_inner_split else "outer_train", inner_train),
            ("inner_validation", inner_validation),
            ("outer_test", test_subjects),
        ):
            for subject_id in subjects:
                rows.append({"run": run_name, "outer_fold": fold["outer_fold"], "role": role, "subject_id": subject_id})
    return rows


def fold_prediction_rows(run_name, summary):
    """Flatten untouched outer-test predictions for independent reanalysis."""
    rows = []
    for fold in summary["folds"]:
        predictions = fold.get("outer_test_predictions", {})
        required = ("sample_indices", "subject_ids", "y_true", "y_pred")
        if not all(key in predictions for key in required):
            continue
        lengths = {len(predictions[key]) for key in required}
        if len(lengths) != 1:
            raise ValueError(f"{run_name}: inconsistent prediction lengths in fold {fold['outer_fold']}")
        for sample_index, subject_id, truth, prediction in zip(
            predictions["sample_indices"],
            predictions["subject_ids"],
            predictions["y_true"],
            predictions["y_pred"],
        ):
            rows.append({
                "run": run_name,
                "outer_fold": int(fold["outer_fold"]),
                "sample_index": int(sample_index),
                "subject_id": int(subject_id),
                "y_true": int(truth),
                "y_pred": int(prediction),
            })
    return rows


def confusion_rows(run_name, summary):
    """Return long-form per-fold and aggregate outer-test confusion matrices."""
    rows = []
    matrices = []
    for fold in summary["folds"]:
        recorded = fold.get("outer_test", {}).get("confusion_matrix")
        if recorded is None:
            predictions = fold.get("outer_test_predictions", {})
            matrix = confusion_matrix(
                predictions["y_true"],
                predictions["y_pred"],
                labels=np.arange(len(summary["class_names"])),
            )
        else:
            matrix = np.asarray(recorded, dtype=int)
        matrices.append(matrix)
        for actual in range(matrix.shape[0]):
            for predicted in range(matrix.shape[1]):
                rows.append({
                    "run": run_name,
                    "outer_fold": int(fold["outer_fold"]),
                    "actual_class": int(actual),
                    "predicted_class": int(predicted),
                    "count": int(matrix[actual, predicted]),
                })
    aggregate = np.sum(matrices, axis=0)
    aggregate_rows = []
    for actual in range(aggregate.shape[0]):
        for predicted in range(aggregate.shape[1]):
            aggregate_rows.append({
                "run": run_name,
                "actual_class": int(actual),
                "predicted_class": int(predicted),
                "count": int(aggregate[actual, predicted]),
            })
    return rows, aggregate_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--reference", default="kernel_k7_9_11_13")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir) if args.output_dir else root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summaries = {path.parent.name: json.loads(path.read_text()) for path in sorted(root.glob("*/summary.json"))}
    if not summaries:
        raise SystemExit(f"No completed */summary.json experiments found under {root}")

    subject_tables = {}
    all_subject_rows, all_distributions, all_bootstrap = [], [], []
    for run_name, summary in summaries.items():
        try:
            validate_subject_disjoint_summary(run_name, summary)
            subject_rows, cms, _ = subject_rows_and_confusions(run_name, summary)
        except ValueError as exc:
            print(f"SKIP subject statistics for {run_name}: {exc}")
            continue
        subject_tables[run_name] = subject_rows
        all_subject_rows.extend(subject_rows)
        all_distributions.extend(distribution_rows(run_name, subject_rows))
        for row in subject_block_bootstrap(cms, args.bootstrap_resamples, args.seed):
            all_bootstrap.append({"run": run_name, **row})
        plot_subjects(output / f"subject_distribution_{run_name}.png", run_name, subject_rows)

    write_csv(output / "all_subject_metrics.csv", all_subject_rows)
    write_csv(output / "subject_distribution_summary.csv", all_distributions)
    write_csv(output / "subject_bootstrap_95ci.csv", all_bootstrap)
    write_csv(output / "experiment_performance_summary.csv", [experiment_row(name, summary) for name, summary in summaries.items()])
    fold_rows = []
    prediction_rows = []
    fold_confusion_rows = []
    aggregate_confusion_rows = []
    for name, summary in summaries.items():
        fold_rows.extend(fold_assignment_rows(name, summary))
        prediction_rows.extend(fold_prediction_rows(name, summary))
        per_fold_confusions, aggregate_confusions = confusion_rows(name, summary)
        fold_confusion_rows.extend(per_fold_confusions)
        aggregate_confusion_rows.extend(aggregate_confusions)
    write_csv(output / "fold_assignments.csv", fold_rows)
    write_csv(output / "fold_level_predictions.csv", prediction_rows)
    write_csv(output / "fold_confusion_matrices.csv", fold_confusion_rows)
    write_csv(output / "aggregate_confusion_matrices.csv", aggregate_confusion_rows)

    comparison_rows = []
    # Compare the primary full multi-scale model against every other completed
    # subject-disjoint experiment. This covers kernel and window ablations as
    # well as neural/classical baselines, while Holm correction is applied over
    # the complete family of reported comparisons.
    comparators = sorted(name for name in subject_tables if name != args.reference)
    if args.reference in subject_tables and comparators:
        comparison_rows = expanded_comparisons(
            subject_tables, args.reference, comparators, args.bootstrap_resamples, args.seed
        )
        write_csv(output / "expanded_wilcoxon_effect_sizes.csv", comparison_rows)
        fold_comparison_rows = expanded_fold_comparisons(
            summaries, args.reference, comparators, args.bootstrap_resamples, args.seed
        )
        write_csv(output / "fold_level_paired_comparisons.csv", fold_comparison_rows)

    manifest = {
        "completed_experiments": sorted(summaries),
        "subject_statistics_experiments": sorted(subject_tables),
        "comparison_reference": args.reference,
        "comparison_targets_completed": comparators,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "statistical_unit_note": "Subject-disjoint out-of-fold predictions are primary. Wilcoxon comparisons pair subjects and are accompanied by effect sizes and subject-bootstrap confidence intervals.",
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
