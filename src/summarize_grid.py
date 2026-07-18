#!/usr/bin/env python3
"""Create manuscript-ready CSV summaries from completed robustness runs."""

import argparse
import csv
import json
from pathlib import Path


def mean_std(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)
    runs = []
    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text())
        runs.append((path.parent.name, data))
    if not runs:
        raise SystemExit(f"No */summary.json files found under {root}")

    summary_rows = []
    recall_rows = []
    for run_name, data in runs:
        a = data["args"]
        c = data["compute"]
        metrics = data["aggregate"]
        summary_rows.append(
            {
                "run": run_name,
                "model": a["model"],
                "imbalance": a["imbalance"],
                "seed": a["seed"],
                "parameters": c["parameters"],
                "macs_per_sample": c["macs_per_sample"],
                "flops_per_sample": c["flops_per_sample_2x_macs"],
                "latency_ms_batch1": c["latency_ms_batch1_mean"],
                "accuracy_mean": metrics["accuracy"]["mean"],
                "accuracy_sd": metrics["accuracy"]["std"],
                "balanced_accuracy_mean": metrics["balanced_accuracy"]["mean"],
                "balanced_accuracy_sd": metrics["balanced_accuracy"]["std"],
                "macro_f1_mean": metrics["f1"]["mean"],
                "macro_f1_sd": metrics["f1"]["std"],
            }
        )
        for class_name, values in metrics["per_class_recall"].items():
            recall_rows.append(
                {
                    "run": run_name,
                    "model": a["model"],
                    "imbalance": a["imbalance"],
                    "seed": a["seed"],
                    "class": class_name,
                    "recall_mean": values["mean"],
                    "recall_sd": values["std"],
                }
            )

    for filename, rows in [("experiment_summary.csv", summary_rows), ("per_class_recall.csv", recall_rows)]:
        with (root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    grouped = {}
    for row in summary_rows:
        key = (row["model"], row["imbalance"])
        grouped.setdefault(key, []).append(row)
    multiseed = []
    for (model, imbalance), rows in sorted(grouped.items()):
        item = {"model": model, "imbalance": imbalance, "seeds_completed": len(rows)}
        for metric in ["accuracy_mean", "balanced_accuracy_mean", "macro_f1_mean", "latency_ms_batch1"]:
            mean, sd = mean_std([r[metric] for r in rows])
            item[f"{metric}_across_seeds"] = mean
            item[f"{metric}_seed_sd"] = sd
        item["parameters"] = rows[0]["parameters"]
        item["flops_per_sample"] = rows[0]["flops_per_sample"]
        multiseed.append(item)
    with (root / "multiseed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(multiseed[0]))
        writer.writeheader()
        writer.writerows(multiseed)
    print(f"Summarised {len(runs)} completed runs in {root}")


if __name__ == "__main__":
    main()
