#!/usr/bin/env python3
"""Matched CSP+LDA baseline using the manuscript's subject-disjoint outer folds."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold

import run_robustness_study as study


def metrics(y_true, y_pred, n_classes):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": recall_score(
            y_true,
            y_pred,
            labels=np.arange(n_classes),
            average=None,
            zero_division=0,
        ).astype(float).tolist(),
    }


def aggregate(records, class_names):
    output = {}
    for name in ("accuracy", "balanced_accuracy", "f1"):
        values = np.asarray([record["outer_test"][name] for record in records], dtype=float)
        output[name] = {"mean": float(values.mean()), "std": float(values.std()), "n": int(len(values))}
    recall = np.asarray([record["outer_test"]["per_class_recall"] for record in records], dtype=float)
    output["per_class_recall"] = {
        name: {"mean": float(recall[:, index].mean()), "std": float(recall[:, index].std())}
        for index, name in enumerate(class_names)
    }
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/kaggle/working/eeg_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-folds", type=int)
    return parser.parse_args()


def run(args, dataset=None):
    mne.set_log_level("WARNING")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if dataset is None:
        dataset = study.load_verified_dataset(args.data_dir, args.max_subjects, args.tmin, args.tmax)
    if not np.isclose(dataset.tmin, args.tmin) or not np.isclose(dataset.tmax, args.tmax):
        raise ValueError("The reused dataset epoch window does not match CSP+LDA arguments.")

    x = torch_stack_numpy(dataset.samples)
    y = np.asarray(dataset.labels, dtype=int)
    subjects = np.asarray(dataset.subject_ids, dtype=int)
    indices = np.arange(len(dataset))
    folds = GroupKFold(n_splits=args.outer_folds).split(indices, y, groups=subjects)
    records = []
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        if args.max_folds and fold_number > args.max_folds:
            break
        fold_path = output / f"fold_{fold_number:02d}.json"
        if fold_path.exists():
            records.append(json.loads(fold_path.read_text()))
            continue
        deadline_unix = getattr(args, "deadline_unix", None)
        if deadline_unix and time.time() >= deadline_unix:
            print("SAFE STOP for CSP+LDA at an outer-fold boundary.", flush=True)
            break
        start = time.perf_counter()
        csp = CSP(
            n_components=args.components,
            reg=None,
            log=True,
            norm_trace=False,
            component_order="mutual_info",
        )
        lda = LinearDiscriminantAnalysis()
        train_features = csp.fit_transform(x[train_idx], y[train_idx])
        test_features = csp.transform(x[test_idx])
        lda.fit(train_features, y[train_idx])
        y_pred = lda.predict(test_features).astype(int)
        y_true = y[test_idx]
        record = {
            "outer_fold": fold_number,
            "seed": args.seed,
            "model": "csp_lda",
            "outer_train_subjects": sorted(np.unique(subjects[train_idx]).astype(int).tolist()),
            "outer_test_subjects": sorted(np.unique(subjects[test_idx]).astype(int).tolist()),
            "outer_test": metrics(y_true, y_pred, len(dataset.class_names)),
            "outer_test_predictions": {
                "sample_indices": test_idx.astype(int).tolist(),
                "subject_ids": subjects[test_idx].astype(int).tolist(),
                "y_true": y_true.astype(int).tolist(),
                "y_pred": y_pred.astype(int).tolist(),
            },
            "fit_and_predict_seconds": float(time.perf_counter() - start),
        }
        fold_path.write_text(json.dumps(record, indent=2))
        records.append(record)

    required = args.max_folds or args.outer_folds
    if len(records) < required:
        partial = {"status": "incomplete", "completed_outer_folds": len(records), "required_outer_folds": required}
        (output / "partial_progress.json").write_text(json.dumps(partial, indent=2))
        return partial
    final = {
        "protocol": "Outer subject-wise GroupKFold; CSP and LDA are fitted only on outer-training subjects; outer-test subjects are evaluated once.",
        "args": vars(args),
        "class_names": dataset.class_names,
        "samples": len(dataset),
        "subjects": int(len(np.unique(subjects))),
        "class_counts": dataset.class_counts(),
        "epoch": {
            "tmin_seconds": float(args.tmin),
            "tmax_seconds": float(args.tmax),
            "samples": int(x.shape[-1]),
            "sampling_frequency_hz": float(dataset.sampling_frequency_hz),
        },
        "preprocessing": dataset.preprocessing_metadata,
        "csp": {"n_components": args.components, "reg": None, "log": True, "norm_trace": False},
        "lda": {"implementation": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis", "defaults": True},
        "compute": {
            "parameters": None,
            "macs_per_sample": None,
            "flops_per_sample_2x_macs": None,
            "latency_ms_batch1_mean": None,
            "note": "Neural-network parameter/FLOP profiling is not applicable to CSP+LDA.",
        },
        "aggregate": aggregate(records, dataset.class_names),
        "folds": records,
    }
    (output / "summary.json").write_text(json.dumps(final, indent=2))
    return final


def torch_stack_numpy(samples):
    import torch

    return torch.stack(samples).numpy()


if __name__ == "__main__":
    run(parse_args())
