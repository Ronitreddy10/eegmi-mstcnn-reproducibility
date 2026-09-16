#!/usr/bin/env python3
"""Single resumable entrypoint for all EEG MST-CNN reviewer experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

import torch


# Set this one value in a Kaggle cell or pass --data-dir. The directory may be
# the Kaggle dataset root; nested folders are searched for S###R##.edf files.
DATA_DIR = os.environ.get("DATA_DIR", "")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import eeg_journal_analysis as physiology
import run_csp_lda
import run_robustness_study as study


CORE_STAGES = {"kernels", "windows", "baselines"}
PACKAGE_NAMES = ("mne", "numpy", "scikit-learn", "scipy", "torch", "matplotlib")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-root", default="/kaggle/working/eeg_mstcnn_reviewer_results")
    parser.add_argument(
        "--stage",
        choices=[
            "all", "kernels", "windows", "baselines", "classical",
            "physiology", "remaining-cpu", "analysis", "optional-kernels",
        ],
        default="all",
    )
    parser.add_argument("--resume-archive", help="Optional prior checkpoint ZIP attached to this Kaggle session.")
    parser.add_argument("--max-wall-hours", type=float, default=11.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Use 20 subjects, two folds, one completed fold, and two epochs.")
    return parser.parse_args()


def discover_data_dir(value):
    if value:
        root = Path(value).expanduser().resolve()
        if root.exists():
            return root
        raise FileNotFoundError(f"Configured DATA_DIR does not exist: {root}")
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("S001R04.edf"):
            relative = candidate.relative_to(kaggle_input)
            return kaggle_input / relative.parts[0]
        for candidate in kaggle_input.rglob("S001R04.EDF"):
            relative = candidate.relative_to(kaggle_input)
            return kaggle_input / relative.parts[0]
    raise FileNotFoundError(
        "EEGMMIDB was not found. Pass --data-dir /kaggle/input/<dataset-folder> or set DATA_DIR at the top of kaggle_entrypoint.py."
    )


def safe_extract(archive_path, output_root):
    archive_path = Path(archive_path)
    with ZipFile(archive_path) as archive:
        root = output_root.resolve()
        for name in archive.namelist():
            destination = (output_root / name).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Unsafe ZIP member: {name}")
        archive.extractall(output_root)


def checkpoint(output_root):
    target = output_root.parent / "eeg_mstcnn_reviewer_results_checkpoint"
    return Path(shutil.make_archive(str(target), "zip", root_dir=output_root))


def load_jobs(stage):
    config = ROOT / "configs" / (
        "reviewer_optional_alternative_kernels.json" if stage == "optional-kernels" else "reviewer_experiments.json"
    )
    jobs = json.loads(config.read_text())
    if stage == "all":
        return jobs
    if stage in {"windows", "baselines"}:
        reference = [job for job in jobs if job["name"] == "kernel_k7_9_11_13"]
        return reference + [job for job in jobs if job["stage"] == stage]
    return [job for job in jobs if job["stage"] == stage]


def experiment_args(cli, job, output, deadline):
    return Namespace(
        data_dir=str(cli.data_dir),
        output_dir=str(output),
        model=job["model"],
        imbalance=job["imbalance"],
        seed=int(job["seed"]),
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
        num_workers=cli.num_workers,
        device="auto",
        max_subjects=20 if cli.smoke else None,
        max_folds=1 if cli.smoke else None,
        refit_mode="none",
        checkpoint_archive=None,
        deadline_unix=deadline,
        kernels=job.get("kernels", [7, 9, 11, 13]),
        tmin=float(job.get("tmin", 0.0)),
        tmax=float(job.get("tmax", 4.0)),
    )


def environment_manifest(cli, data_dir, jobs):
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_dir": str(data_dir) if data_dir else None,
        "stage": cli.stage,
        "jobs": jobs,
        "fixed_training_protocol": {
            "outer_cv": "10-fold subject-wise GroupKFold",
            "inner_validation": "grouped 15% of outer-training subjects",
            "checkpoint_selection": "maximum inner-validation macro-F1",
            "refit": False,
            "epochs_max": cli.epochs,
            "early_stopping_patience": cli.patience,
            "batch_size": cli.batch_size,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "dropout": 0.5,
        },
    }


def run_analysis(output_root):
    analysis = output_root / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(SRC / "analyze_reviewer_results.py"),
            "--results-root",
            str(output_root / "experiments"),
            "--output-dir",
            str(analysis),
            "--reference",
            "kernel_k7_9_11_13",
            "--bootstrap-resamples",
            "10000",
            "--seed",
            "42",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(SRC / "report_receptive_field.py"), "--output-dir", str(analysis)],
        check=True,
    )


def run_physiology(cli, data_dir, output_root):
    args = Namespace(
        data_dir=str(data_dir),
        output_dir=str(output_root / "physiology"),
        synthetic_smoke=cli.smoke,
        max_subjects=12 if cli.smoke else None,
        tmin=0.0,
        tmax=4.0,
        reject_uv=300.0,
        sample_channel="C3",
        sample_start=0.0,
        sample_seconds=12.0,
        seed=42,
        attached_edf_index=physiology.index_attached_edfs(),
    )
    output = physiology.ensure_dir(args.output_dir)
    signal_summary = physiology.run_signal_analysis(args, output)
    (output / "journal_analysis_manifest.json").write_text(json.dumps({"classes": physiology.CLASS_NAMES, "signal_analysis": signal_summary}, indent=2))


def main():
    cli = parse_args()
    output_root = Path(cli.output_root).expanduser().resolve()
    experiments_root = output_root / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)
    if cli.resume_archive:
        safe_extract(cli.resume_archive, output_root)

    needs_data = cli.stage != "analysis"
    data_dir = discover_data_dir(cli.data_dir) if needs_data else None
    if data_dir:
        os.environ["EEGMMIDB_SOURCE_DIR"] = str(data_dir)
        cli.data_dir = str(data_dir)
    jobs = load_jobs(cli.stage) if cli.stage not in {"classical", "physiology", "remaining-cpu", "analysis"} else []
    (output_root / "run_environment.json").write_text(json.dumps(environment_manifest(cli, data_dir, jobs), indent=2))
    start = time.time()
    deadline = start + cli.max_wall_hours * 3600.0
    current_dataset = None
    current_window = None
    stopped_for_time = False

    try:
        for position, job in enumerate(jobs, start=1):
            if time.time() >= deadline:
                stopped_for_time = True
                break
            output = experiments_root / job["name"]
            if (output / "summary.json").exists():
                print(f"SKIP completed {position}/{len(jobs)}: {job['name']}", flush=True)
                continue
            window = (float(job.get("tmin", 0.0)), float(job.get("tmax", 4.0)))
            if current_dataset is None or current_window != window:
                del current_dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_dataset = study.load_verified_dataset(
                    str(data_dir),
                    max_subjects=20 if cli.smoke else None,
                    tmin=window[0],
                    tmax=window[1],
                )
                current_window = window
            print(f"START {position}/{len(jobs)}: {job['name']}", flush=True)
            result = study.run_experiment(
                experiment_args(cli, job, output, deadline),
                dataset=current_dataset,
            )
            checkpoint(output_root)
            if result.get("status") == "incomplete":
                stopped_for_time = True
                break

        if cli.stage in {"all", "baselines", "classical", "remaining-cpu"} and not stopped_for_time:
            csp_output = experiments_root / "baseline_csp_lda"
            if not (csp_output / "summary.json").exists():
                if current_dataset is None or current_window != (0.0, 4.0):
                    current_dataset = study.load_verified_dataset(
                        str(data_dir), max_subjects=20 if cli.smoke else None, tmin=0.0, tmax=4.0
                    )
                csp_args = Namespace(
                    data_dir=str(data_dir), output_dir=str(csp_output), tmin=0.0, tmax=4.0,
                    outer_folds=2 if cli.smoke else 10, components=8, seed=42,
                    max_subjects=20 if cli.smoke else None, max_folds=1 if cli.smoke else None,
                    deadline_unix=deadline,
                )
                csp_result = run_csp_lda.run(csp_args, dataset=current_dataset)
                checkpoint(output_root)
                if csp_result.get("status") == "incomplete":
                    stopped_for_time = True

        if cli.stage in {"all", "physiology", "remaining-cpu"} and not stopped_for_time and time.time() < deadline:
            run_physiology(cli, data_dir, output_root)
            checkpoint(output_root)
        elif cli.stage in {"all", "physiology", "remaining-cpu"} and time.time() >= deadline:
            stopped_for_time = True

        if cli.stage in {
            "all", "kernels", "windows", "baselines", "classical",
            "remaining-cpu", "optional-kernels", "analysis",
        }:
            if any(experiments_root.glob("*/summary.json")):
                run_analysis(output_root)
    except Exception as exc:
        (output_root / "RUN_ERROR.json").write_text(json.dumps({
            "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()
        }, indent=2))
        checkpoint(output_root)
        raise

    completed = sorted(path.parent.name for path in experiments_root.glob("*/summary.json"))
    status = {
        "completed_experiments": completed,
        "requested_experiments": [job["name"] for job in jobs],
        "stopped_safely_for_wall_time": stopped_for_time,
        "elapsed_hours": (time.time() - start) / 3600.0,
        "checkpoint_zip": str(checkpoint(output_root)),
        "rerun_same_command_to_resume": stopped_for_time,
    }
    (output_root / "RUN_STATUS.json").write_text(json.dumps(status, indent=2))
    checkpoint(output_root)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
