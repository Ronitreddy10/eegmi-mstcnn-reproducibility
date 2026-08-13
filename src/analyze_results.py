#!/usr/bin/env python3
"""Validate a completed reviewer-safe archive and regenerate manuscript outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


METRICS = ("accuracy", "balanced_accuracy", "f1")
LABELS = {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "f1": "Macro-F1"}
CAPACITIES = {
    "0.5 M": "capacity_0_5m_unweighted",
    "5 M": "capacity_5m_unweighted",
    "20 M": "capacity_20m_unweighted",
    "83 M": "main_83m_unweighted_s42",
}
# Exact 641-sample FLOP counts (FLOPs = 2 x MACs) from the public profiler.
# The stored performance archive predates the clarified sample-count reporting;
# its fold metrics are unchanged, while these deterministic compute counts use
# the actual MNE tensor length.
FLOPS_641 = {
    "capacity_0_5m_unweighted": 396_159_616,
    "capacity_5m_unweighted": 2_039_066_640,
    "capacity_20m_unweighted": 4_221_060_880,
    "main_83m_unweighted_s42": 7_078_550_544,
    "balanced_83m_effective_ce_s42": 7_078_550_544,
    "baseline_eegnet_effective_ce": 43_556_352,
    "baseline_shallowconvnet_effective_ce": 205_349_440,
    "baseline_deepconvnet1d_effective_ce": 398_696_448,
    "baseline_resnet1d_effective_ce": 585_852_928,
}
BASELINES = {
    "EEGNet": "baseline_eegnet_effective_ce",
    "ShallowConvNet": "baseline_shallowconvnet_effective_ce",
    "DeepConvNet1D": "baseline_deepconvnet1d_effective_ce",
    "ResNet1D": "baseline_resnet1d_effective_ce",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="Completed reviewer_safe_13_results_export ZIP")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def load_archive(path: Path):
    with ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.startswith("__MACOSX/") and not PurePosixPath(n).name.startswith("._")]
        summary_names = [n for n in names if n.endswith("/summary.json")]
        summaries = {PurePosixPath(name).parent.name: json.loads(archive.read(name)) for name in summary_names}
        folds = {}
        for job in summaries:
            selected = sorted(
                n for n in names
                if f"/{job}/fold_" in f"/{n}" and n.endswith(".json")
            )
            folds[job] = [json.loads(archive.read(name)) for name in selected]
    return summaries, folds


def validate(summaries, folds):
    expected = set(CAPACITIES.values()) | set(BASELINES.values()) | {
        f"main_83m_unweighted_s{seed}" for seed in (42, 123, 2026)
    } | {
        f"balanced_83m_effective_ce_s{seed}" for seed in (42, 123, 2026)
    }
    if set(summaries) != expected:
        raise RuntimeError(f"Archive run set differs from reviewer-safe grid: {set(summaries) ^ expected}")
    for job, records in folds.items():
        if len(records) != 10:
            raise RuntimeError(f"{job}: expected 10 fold records, found {len(records)}")
        outer_subjects = []
        for record in records:
            train = set(record["inner_train_subjects"])
            validation = set(record["inner_validation_subjects"])
            test = set(record["outer_test_subjects"])
            if train & validation or train & test or validation & test:
                raise RuntimeError(f"Subject overlap detected in {job}, fold {record['outer_fold']}")
            if set(record["subjects_used_to_fit_reported_checkpoint"]) != train:
                raise RuntimeError(f"Reported checkpoint-fit subjects differ in {job}, fold {record['outer_fold']}")
            outer_subjects.extend(test)
        unique, counts = np.unique(outer_subjects, return_counts=True)
        if len(unique) != 103 or set(counts.tolist()) != {1}:
            raise RuntimeError(f"Outer-test coverage is not exactly once for all 103 subjects in {job}")
        if summaries[job]["args"].get("refit_mode") != "none":
            raise RuntimeError(f"Unexpected refit mode in {job}")
        for metric in METRICS:
            measured = np.mean([r["outer_test"][metric] for r in records])
            claimed = summaries[job]["aggregate"][metric]["mean"]
            if not math.isclose(measured, claimed, abs_tol=1e-12):
                raise RuntimeError(f"Aggregate mismatch: {job}, {metric}")


def fold_vector(folds, job, metric):
    return np.asarray([record["outer_test"][metric] for record in folds[job]], dtype=float)


def make_statistics(folds):
    statistics = {"capacity": {}, "baseline": {}, "loss": {}, "twenty_vs_83": {}}
    for metric in METRICS:
        vectors = [fold_vector(folds, job, metric) for job in CAPACITIES.values()]
        omnibus = friedmanchisquare(*vectors)
        raw = {
            name: float(wilcoxon(vectors[2], vector, alternative="two-sided", method="auto").pvalue)
            for name, vector in zip(CAPACITIES, vectors) if name != "20 M"
        }
        statistics["capacity"][metric] = {
            "friedman_chi2": float(omnibus.statistic),
            "friedman_p": float(omnibus.pvalue),
            "20m_pairwise_raw": raw,
            "20m_pairwise_holm": holm(raw),
        }
        target = fold_vector(folds, CAPACITIES["20 M"], metric)
        raw_baseline = {
            name: float(wilcoxon(target, fold_vector(folds, job, metric), alternative="greater", method="auto").pvalue)
            for name, job in BASELINES.items()
        }
        statistics["baseline"][metric] = {"raw": raw_baseline, "holm": holm(raw_baseline)}
        reference = fold_vector(folds, CAPACITIES["83 M"], metric)
        statistics["twenty_vs_83"][metric] = {
            "p_two_sided": float(wilcoxon(target, reference, alternative="two-sided", method="auto").pvalue),
            "mean_difference_pp": float((target.mean() - reference.mean()) * 100),
        }
        unweighted = np.vstack([
            fold_vector(folds, f"main_83m_unweighted_s{seed}", metric) for seed in (42, 123, 2026)
        ]).mean(axis=0)
        weighted = np.vstack([
            fold_vector(folds, f"balanced_83m_effective_ce_s{seed}", metric) for seed in (42, 123, 2026)
        ]).mean(axis=0)
        statistics["loss"][metric] = {
            "p_two_sided": float(wilcoxon(weighted, unweighted, alternative="two-sided", method="auto").pvalue),
            "weighted_minus_unweighted_pp": float((weighted.mean() - unweighted.mean()) * 100),
        }
    return statistics


def seed_summary(summaries, folds):
    result = {}
    for mode, prefix in (
        ("Unweighted CE", "main_83m_unweighted_s"),
        ("Effective-number CE", "balanced_83m_effective_ce_s"),
    ):
        jobs = [f"{prefix}{seed}" for seed in (42, 123, 2026)]
        row = {}
        for metric in METRICS:
            seed_means = [summaries[job]["aggregate"][metric]["mean"] * 100 for job in jobs]
            row[metric] = dict(zip(("mean", "seed_sd"), mean_sd(seed_means)))
        recalls = np.asarray([
            np.mean([r["outer_test"]["per_class_recall"] for r in folds[job]], axis=0) * 100
            for job in jobs
        ])
        row["recall"] = {
            name: {"mean": float(recalls[:, i].mean()), "seed_sd": float(recalls[:, i].std(ddof=0))}
            for i, name in enumerate(("Rest", "Left-fist", "Right-fist", "Both-feet"))
        }
        result[mode] = row
    return result


def performance_rows(summaries):
    order = [
        ("MST-CNN 0.5 M, unweighted", "capacity_0_5m_unweighted"),
        ("MST-CNN 5 M, unweighted", "capacity_5m_unweighted"),
        ("MST-CNN 20 M, unweighted", "capacity_20m_unweighted"),
        ("MST-CNN 83 M, unweighted (seed 42)", "main_83m_unweighted_s42"),
        ("MST-CNN 83 M, effective CE (seed 42)", "balanced_83m_effective_ce_s42"),
        ("EEGNet, effective CE", "baseline_eegnet_effective_ce"),
        ("ShallowConvNet, effective CE", "baseline_shallowconvnet_effective_ce"),
        ("DeepConvNet1D, effective CE", "baseline_deepconvnet1d_effective_ce"),
        ("ResNet1D, effective CE", "baseline_resnet1d_effective_ce"),
    ]
    rows = []
    for label, job in order:
        summary = summaries[job]
        rows.append({
            "model": label,
            "parameters": summary["compute"]["parameters"],
            "gflops": FLOPS_641[job] / 1e9,
            "latency_ms": summary["compute"]["latency_ms_batch1_mean"],
            **{
                metric: (summary["aggregate"][metric]["mean"] * 100, summary["aggregate"][metric]["std"] * 100)
                for metric in METRICS
            },
        })
    return rows


def write_outputs(output, statistics, seed, rows):
    output.mkdir(parents=True, exist_ok=True)
    payload = {"statistics": statistics, "seed_summary": seed, "performance_rows": rows}
    (output / "manuscript_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "performance_rows.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Model", "Params", "GFLOPs", "Latency ms", "Accuracy mean", "Accuracy SD", "Balanced mean", "Balanced SD", "Macro-F1 mean", "Macro-F1 SD"])
        for row in rows:
            writer.writerow([row["model"], row["parameters"], row["gflops"], row["latency_ms"], *row["accuracy"], *row["balanced_accuracy"], *row["f1"]])


def make_figures(output, rows, seed):
    cap = rows[:4]
    params = np.asarray([row["parameters"] for row in cap]) / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    colors = {"accuracy": "#1f4e79", "balanced_accuracy": "#d28b18", "f1": "#2e7d5a"}
    for metric in METRICS:
        axes[0].plot(params, [row[metric][0] for row in cap], marker="o", lw=2.2, ms=6, label=LABELS[metric], color=colors[metric])
    axes[0].set_xscale("log"); axes[0].set_xticks(params, ["0.5", "5", "20", "83"])
    axes[0].set_xlabel("Trainable parameters (million; log scale)"); axes[0].set_ylabel("Performance (%)")
    axes[0].grid(axis="y", alpha=0.25); axes[0].legend(frameon=False, fontsize=9); axes[0].set_title("(a) Capacity ablation")
    axes[1].plot(params, [row["gflops"] for row in cap], marker="s", lw=2.2, color="#7a3e9d", label="GFLOPs/sample")
    right = axes[1].twinx(); right.plot(params, [row["latency_ms"] for row in cap], marker="^", lw=2.2, color="#b33a3a", label="Latency")
    axes[1].set_xscale("log"); axes[1].set_xticks(params, ["0.5", "5", "20", "83"])
    axes[1].set_xlabel("Trainable parameters (million; log scale)"); axes[1].set_ylabel("GFLOPs per sample", color="#7a3e9d")
    right.set_ylabel("Batch-1 latency (ms)", color="#b33a3a"); axes[1].grid(axis="y", alpha=0.25); axes[1].set_title("(b) Computational cost")
    handles = axes[1].lines + right.lines; axes[1].legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(output / "supplementary_figure_s2_capacity_cost.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    classes, x, width = ["0 Rest", "1 Left", "2 Right", "3 Feet"], np.arange(4), 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for offset, (mode, color) in zip((-width / 2, width / 2), (("Unweighted CE", "#1f4e79"), ("Effective-number CE", "#d28b18"))):
        means = [seed[mode]["recall"][c]["mean"] for c in ("Rest", "Left-fist", "Right-fist", "Both-feet")]
        sds = [seed[mode]["recall"][c]["seed_sd"] for c in ("Rest", "Left-fist", "Right-fist", "Both-feet")]
        ax.bar(x + offset, means, width, yerr=sds, capsize=4, label=mode, color=color, edgecolor="white", linewidth=0.7)
    ax.set_xticks(x, classes); ax.set_ylabel("Recall (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper center"); ax.set_title("Three-seed class-recall comparison for the 83 M MST-CNN")
    fig.tight_layout(); fig.savefig(output / "class_recall_diagnostic.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    args = parse_args()
    summaries, folds = load_archive(Path(args.archive))
    validate(summaries, folds)
    statistics = make_statistics(folds)
    seeds = seed_summary(summaries, folds)
    rows = performance_rows(summaries)
    output = Path(args.output_dir)
    write_outputs(output, statistics, seeds, rows)
    make_figures(output, rows, seeds)
    print(f"Validated {len(summaries)} experiments and {sum(map(len, folds.values()))} outer folds; outputs written to {output}")


if __name__ == "__main__":
    main()
