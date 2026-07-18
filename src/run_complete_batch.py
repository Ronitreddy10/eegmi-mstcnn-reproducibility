#!/usr/bin/env python3
"""One-click, resumable runner for the complete multi-seed experiment grid."""

import argparse
import json
import shutil
import time
import traceback
from argparse import Namespace
from pathlib import Path

import torch

import run_robustness_study as study


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--data-dir", default="/kaggle/working/eeg_data")
    p.add_argument("--output-root", default="/kaggle/working/robustness_results")
    p.add_argument("--checkpoint-archive", default="/kaggle/working/robustness_results_export")
    p.add_argument("--max-wall-hours", type=float, default=11.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--refit-mode", choices=["none", "selected_epoch"], default="none")
    return p.parse_args()


def expand_jobs(path):
    jobs = json.loads(Path(path).read_text())
    expanded = []
    for job in jobs:
        for seed in job.get("seeds", [job.get("seed", 42)]):
            run_name = f'{job["name"]}_s{seed}' if "seeds" in job else job["name"]
            expanded.append({**job, "seed": int(seed), "run_name": run_name})
    return expanded


def experiment_args(cli, job, output):
    return Namespace(
        data_dir=cli.data_dir,
        output_dir=str(output),
        model=job["model"],
        imbalance=job["imbalance"],
        seed=job["seed"],
        outer_folds=2 if cli.smoke else 10,
        inner_val_fraction=0.15,
        epochs=2 if cli.smoke else cli.epochs,
        patience=2 if cli.smoke else cli.patience,
        batch_size=cli.batch_size,
        lr=1e-4,
        weight_decay=1e-4,
        dropout=0.5,
        effective_beta=0.9999,
        focal_gamma=2.0,
        num_workers=4,
        device="auto",
        max_subjects=20 if cli.smoke else None,
        max_folds=1 if cli.smoke else None,
        checkpoint_archive=Path(cli.checkpoint_archive),
        refit_mode=cli.refit_mode,
        deadline_unix=getattr(cli, "deadline_unix", None),
    )


def main():
    cli = parse_args()
    root = Path(cli.output_root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = expand_jobs(cli.grid)
    start_time = time.time()
    cli.deadline_unix = start_time + cli.max_wall_hours * 3600.0
    dataset = study.load_verified_dataset(cli.data_dir, max_subjects=20 if cli.smoke else None)
    manifest = {
        "protocol": "complete resumable multi-seed batch",
        "jobs": jobs,
        "gpu_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_wall_hours": cli.max_wall_hours,
    }
    (root / "complete_batch_manifest.json").write_text(json.dumps(manifest, indent=2))

    completed = 0
    for index, job in enumerate(jobs):
        output = root / job["run_name"]
        if (output / "summary.json").exists():
            print(f"SKIP completed job {index + 1}/{len(jobs)}: {job['run_name']}", flush=True)
            completed += 1
            continue
        elapsed_hours = (time.time() - start_time) / 3600.0
        if elapsed_hours >= cli.max_wall_hours:
            print(f"SAFE STOP after {elapsed_hours:.2f} h; rerun the same notebook with this archive attached to resume.", flush=True)
            break
        print(f"START job {index + 1}/{len(jobs)}: {job['run_name']}", flush=True)
        try:
            study.run_experiment(experiment_args(cli, job, output), dataset=dataset)
        except Exception as exc:
            error = {
                "job_index": index + 1,
                "run_name": job["run_name"],
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            (root / "BATCH_ERROR.json").write_text(json.dumps(error, indent=2))
            shutil.make_archive(str(cli.checkpoint_archive), "zip", root_dir=root)
            raise
        completed += 1
        shutil.make_archive(str(cli.checkpoint_archive), "zip", root_dir=root)

    status = {
        "completed_jobs": sum((root / j["run_name"] / "summary.json").exists() for j in jobs),
        "total_jobs": len(jobs),
        "elapsed_hours": (time.time() - start_time) / 3600.0,
        "complete": all((root / j["run_name"] / "summary.json").exists() for j in jobs),
    }
    (root / "BATCH_STATUS.json").write_text(json.dumps(status, indent=2))
    shutil.make_archive(str(cli.checkpoint_archive), "zip", root_dir=root)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
