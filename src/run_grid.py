#!/usr/bin/env python3
"""Run a resumable experiment grid and archive result JSON after each job."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--data-dir", default="/kaggle/working/eeg_data")
    p.add_argument("--output-root", default="/kaggle/working/robustness_results")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    jobs = json.loads(Path(args.grid).read_text())
    expanded = []
    for job in jobs:
        for seed in job.get("seeds", [job.get("seed", 42)]):
            expanded.append({**job, "seed": seed, "run_name": f'{job["name"]}_s{seed}' if "seeds" in job else job["name"]})
    selected = expanded[args.start : args.stop]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_robustness_study.py")
    manifest = {"total_expanded_jobs": len(expanded), "start": args.start, "stop": args.stop, "selected": selected}
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    for position, job in enumerate(selected, start=args.start):
        output = root / job["run_name"]
        if (output / "summary.json").exists():
            print(f"SKIP completed job {position}: {job['run_name']}")
            continue
        command = [
            sys.executable,
            str(runner),
            "--data-dir", args.data_dir,
            "--output-dir", str(output),
            "--model", job["model"],
            "--imbalance", job["imbalance"],
            "--seed", str(job["seed"]),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", str(args.batch_size),
        ]
        if args.smoke:
            command += ["--max-subjects", "20", "--outer-folds", "2", "--max-folds", "1", "--epochs", "2", "--patience", "2"]
        print(f"RUN job {position}/{len(expanded)-1}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
        shutil.make_archive(str(root / "robustness_results_latest"), "zip", root_dir=root)


if __name__ == "__main__":
    main()
