#!/usr/bin/env python3
"""Leakage-resistant robustness experiments for the PhysioNet EEGMMIDB study.

This runner deliberately separates three subject sets in every outer fold:

1. inner-training subjects: optimise model weights;
2. inner-validation subjects: select the epoch only;
3. outer-test subjects: evaluate exactly once using the inner-selected checkpoint.

It reuses the verified dataset/event/preprocessing implementation in
``eeg_stage_ablation.py`` and adds capacity, imbalance, seed, cost, and baseline
experiments requested during manuscript review.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import eeg_stage_ablation as base


CAPACITY_CONFIGS = {
    # Widths were selected by an explicit parameter search. Exact counts are
    # always measured and written to the output; target labels are approximate.
    "mstcnn_0_5m": dict(max_channels=48, stream_out=24, pool_size=12, hidden=128),
    "mstcnn_5m": dict(max_channels=128, stream_out=48, pool_size=32, hidden=450),
    "mstcnn_20m": dict(max_channels=192, stream_out=64, pool_size=48, hidden=1250),
    "mstcnn_83m": dict(max_channels=256, stream_out=64, pool_size=48, hidden=6146),
}

# MNE includes both endpoints when epochs are created with tmin=0 and tmax=4.
# At 160 Hz, the implemented 0--4 s window therefore contains 641 samples.
N_EPOCH_SAMPLES = 641


class ScalableMSTCNN(nn.Module):
    """The manuscript architecture with explicit, reproducible width controls."""

    def __init__(
        self,
        n_classes: int,
        max_channels: int,
        stream_out: int,
        pool_size: int,
        hidden: int,
        in_channels: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.streams = nn.ModuleList()
        for kernel in (7, 9, 11, 13):
            second_mid = int(np.geomspace(max_channels, stream_out, 3)[1])
            self.streams.append(
                nn.Sequential(
                    base.ConvBlock(kernel, in_channels, max_channels, max_channels, use_pooling=True),
                    base.ConvBlock(kernel, max_channels, second_mid, stream_out, use_pooling=False),
                )
            )
        fused = 4 * stream_out * pool_size
        self.pool = nn.AdaptiveMaxPool1d(pool_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        streams = [self.pool(stream(x)).flatten(1) for stream in self.streams]
        return self.classifier(torch.cat(streams, dim=1))


class DeepConvNet1D(nn.Module):
    """Mid-sized temporal CNN baseline with no parallel multi-scale fusion."""

    def __init__(self, n_classes: int, in_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        layers = []
        specs = [(64, 25), (128, 15), (256, 11), (256, 7)]
        current = in_channels
        for width, kernel in specs:
            layers.extend(
                [
                    nn.Conv1d(current, width, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(width),
                    nn.ELU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            current = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.classifier = nn.Linear(256 * 16, n_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class ResNet1DBaseline(nn.Module):
    """A second mid-sized baseline with residual temporal feature extraction."""

    def __init__(self, n_classes: int, in_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 128, 2),
            ResidualBlock1D(128, 256, 2),
            ResidualBlock1D(256, 256, 2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        index = torch.arange(targets.numel(), device=targets.device)
        log_pt = log_probs[index, targets]
        pt = probs[index, targets]
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, n_classes: int, dropout: float, device: torch.device):
    if name in CAPACITY_CONFIGS:
        model = ScalableMSTCNN(n_classes=n_classes, dropout=dropout, **CAPACITY_CONFIGS[name])
    elif name == "eegnet":
        model = base.EEGNetBaseline(64, N_EPOCH_SAMPLES, n_classes, dropout)
    elif name == "shallowconvnet":
        model = base.ShallowConvNetBaseline(64, N_EPOCH_SAMPLES, n_classes, dropout)
    elif name == "deepconvnet1d":
        model = DeepConvNet1D(n_classes, dropout=dropout)
    elif name == "resnet1d":
        model = ResNet1DBaseline(n_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)


def class_counts(labels, indices, n_classes):
    return np.bincount(np.asarray(labels)[indices], minlength=n_classes).astype(np.float64)


def effective_number_weights(counts, beta: float):
    counts = np.maximum(counts, 1.0)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return weights / weights.mean()


def make_criterion(mode, labels, indices, n_classes, device, beta, focal_gamma):
    counts = class_counts(labels, indices, n_classes)
    weight = None
    if mode in {"effective_ce", "effective_focal"}:
        weight = torch.tensor(effective_number_weights(counts, beta), dtype=torch.float32, device=device)
    if mode == "none" or mode == "oversample":
        return nn.CrossEntropyLoss(), None
    if mode == "effective_ce":
        return nn.CrossEntropyLoss(weight=weight), weight.detach().cpu().tolist()
    if mode == "focal":
        return FocalLoss(gamma=focal_gamma), None
    if mode == "effective_focal":
        return FocalLoss(gamma=focal_gamma, weight=weight), weight.detach().cpu().tolist()
    raise ValueError(mode)


def make_sampler(mode, labels, indices, n_classes):
    if mode != "oversample":
        return None
    counts = np.maximum(class_counts(labels, indices, n_classes), 1.0)
    sample_weights = 1.0 / counts[np.asarray(labels)[indices]]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True,
    )


def loaders(dataset, train_idx, eval_idx, batch_size, workers, device, imbalance):
    sampler = make_sampler(imbalance, dataset.labels, train_idx, len(dataset.class_names))
    train = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    evaluate = DataLoader(
        Subset(dataset, eval_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    return train, evaluate


def add_recall(metrics, y_true, y_pred, n_classes):
    out = base.metrics_for_json(metrics)
    out["per_class_recall"] = recall_score(
        y_true, y_pred, labels=np.arange(n_classes), average=None, zero_division=0
    ).astype(float).tolist()
    return out


def train_select_epoch(args, dataset, train_idx, val_idx, device, seed):
    seed_everything(seed)
    model = build_model(args.model, len(dataset.class_names), args.dropout, device)
    criterion, weights = make_criterion(
        args.imbalance, dataset.labels, train_idx, len(dataset.class_names), device, args.effective_beta, args.focal_gamma
    )
    train_loader, val_loader = loaders(
        dataset, train_idx, val_idx, args.batch_size, args.num_workers, device, args.imbalance
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_epoch, best_f1, patience = 1, -1.0, 0
    history = []
    best_val = None
    final_val = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = base.train_epoch(model, train_loader, criterion, optimizer, device)
        metrics, y_true, y_pred = base.evaluate(model, val_loader, criterion, device)
        final_val = add_recall(metrics, y_true, y_pred, len(dataset.class_names))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(metrics["loss"]),
                "val_accuracy": float(metrics["accuracy"]),
                "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "val_macro_f1": float(metrics["f1"]),
            }
        )
        if metrics["f1"] > best_f1:
            best_f1 = float(metrics["f1"])
            best_epoch = epoch
            best_val = final_val
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_val, final_val, history, weights


def evaluate_selected_model(args, model, dataset, train_idx, test_idx, device):
    """Evaluate the inner-selected checkpoint once on untouched outer subjects."""
    criterion, weights = make_criterion(
        args.imbalance,
        dataset.labels,
        train_idx,
        len(dataset.class_names),
        device,
        args.effective_beta,
        args.focal_gamma,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics, y_true, y_pred = base.evaluate(model, test_loader, criterion, device)
    return add_recall(metrics, y_true, y_pred, len(dataset.class_names)), weights


def refit_and_test(args, dataset, train_idx, test_idx, selected_epoch, device, seed):
    seed_everything(seed)
    model = build_model(args.model, len(dataset.class_names), args.dropout, device)
    criterion, weights = make_criterion(
        args.imbalance, dataset.labels, train_idx, len(dataset.class_names), device, args.effective_beta, args.focal_gamma
    )
    train_loader, test_loader = loaders(
        dataset, train_idx, test_idx, args.batch_size, args.num_workers, device, args.imbalance
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_curve = []
    for epoch in range(1, selected_epoch + 1):
        train_curve.append(float(base.train_epoch(model, train_loader, criterion, optimizer, device)))
    metrics, y_true, y_pred = base.evaluate(model, test_loader, criterion, device)
    return model, add_recall(metrics, y_true, y_pred, len(dataset.class_names)), train_curve, weights


def profile_macs(model, sample):
    """Count Conv/Linear multiply-accumulates for one forward pass."""
    macs = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv1d):
            batch, out_channels, out_length = output.shape
            kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
            macs += batch * out_channels * out_length * kernel_ops
        elif isinstance(module, nn.Conv2d):
            batch, out_channels, out_h, out_w = output.shape
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
            macs += batch * out_channels * out_h * out_w * kernel_ops
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(sample)
    for item in hooks:
        item.remove()
    return int(macs)


def benchmark(model, device, warmup=30, repeats=100):
    sample = torch.zeros(1, 64, N_EPOCH_SAMPLES, device=device)
    model.eval()
    macs = profile_macs(model, sample)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "parameters": parameter_count(model),
        "macs_per_sample": macs,
        "flops_per_sample_2x_macs": 2 * macs,
        "latency_ms_batch1_mean": float(np.mean(timings)),
        "latency_ms_batch1_std": float(np.std(timings)),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup": warmup,
        "repeats": repeats,
    }


def subject_list(dataset, indices):
    return sorted(np.unique(np.asarray(dataset.subject_ids)[indices]).astype(int).tolist())


def aggregate(results, class_names):
    metric_names = ["accuracy", "balanced_accuracy", "f1"]
    summary = {}
    for name in metric_names:
        values = np.asarray([r["outer_test"][name] for r in results], dtype=float)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std()), "n": int(values.size)}
    recall = np.asarray([r["outer_test"]["per_class_recall"] for r in results], dtype=float)
    summary["per_class_recall"] = {
        name: {"mean": float(recall[:, i].mean()), "std": float(recall[:, i].std())}
        for i, name in enumerate(class_names)
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/kaggle/working/eeg_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=sorted(list(CAPACITY_CONFIGS) + ["eegnet", "shallowconvnet", "deepconvnet1d", "resnet1d"]), required=True)
    parser.add_argument("--imbalance", choices=["none", "effective_ce", "focal", "effective_focal", "oversample"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--effective-beta", type=float, default=0.9999)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-folds", type=int, help="Smoke-test only; omit for the manuscript run.")
    parser.add_argument(
        "--refit-mode",
        choices=["none", "selected_epoch"],
        default="none",
        help="Use none for inner-checkpoint selection followed by one untouched outer-test evaluation. selected_epoch reinitializes and refits on all outer-training subjects.",
    )
    return parser.parse_args()


def load_verified_dataset(data_dir, max_subjects=None):
    """Load the verified four-class dataset once for one or many experiments."""
    return base.PhysioNetStageDataset(
        data_dir=data_dir,
        class_mode="mi4_rest",
        preprocess="bandpass_zscore",
        feature_mode="raw",
        channel_set="all",
        tmin=0.0,
        tmax=4.0,
        max_subjects=max_subjects,
        balance_classes=False,
    )


def run_experiment(args, dataset=None):
    """Run one experiment, optionally reusing an already-loaded dataset."""
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed)
    if dataset is None:
        dataset = load_verified_dataset(args.data_dir, args.max_subjects)
    all_indices = np.arange(len(dataset))
    outer = GroupKFold(n_splits=args.outer_folds)
    folds = outer.split(all_indices, dataset.labels, groups=dataset.subject_ids)
    profile_model = build_model(args.model, len(dataset.class_names), args.dropout, device)
    compute = benchmark(profile_model, device)
    del profile_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []
    for outer_fold, (outer_train, outer_test) in enumerate(folds, start=1):
        if args.max_folds and outer_fold > args.max_folds:
            break
        fold_path = output / f"fold_{outer_fold:02d}.json"
        if fold_path.exists():
            print(f"RESUME: using completed {fold_path}", flush=True)
            results.append(json.loads(fold_path.read_text()))
            continue
        deadline_unix = getattr(args, "deadline_unix", None)
        if deadline_unix and time.time() >= deadline_unix:
            print("SAFE STOP at an outer-fold boundary; attach the export and rerun to resume.", flush=True)
            break
        inner = GroupShuffleSplit(n_splits=1, test_size=args.inner_val_fraction, random_state=args.seed + outer_fold)
        inner_train_rel, inner_val_rel = next(
            inner.split(outer_train, np.asarray(dataset.labels)[outer_train], groups=np.asarray(dataset.subject_ids)[outer_train])
        )
        inner_train = outer_train[inner_train_rel]
        inner_val = outer_train[inner_val_rel]
        assert not (set(subject_list(dataset, inner_train)) & set(subject_list(dataset, inner_val)))
        assert not (set(subject_list(dataset, outer_train)) & set(subject_list(dataset, outer_test)))

        fold_seed = args.seed + outer_fold * 1000
        selected_model, selected_epoch, best_inner, final_inner, history, selection_weights = train_select_epoch(
            args, dataset, inner_train, inner_val, device, fold_seed
        )
        if getattr(args, "refit_mode", "none") == "selected_epoch":
            del selected_model
            model, outer_metrics, refit_curve, refit_weights = refit_and_test(
                args, dataset, outer_train, outer_test, selected_epoch, device, fold_seed
            )
            outer_training_subjects = subject_list(dataset, outer_train)
        else:
            model = selected_model
            outer_metrics, refit_weights = evaluate_selected_model(
                args, model, dataset, inner_train, outer_test, device
            )
            refit_curve = []
            outer_training_subjects = subject_list(dataset, inner_train)
        record = {
            "outer_fold": outer_fold,
            "seed": args.seed,
            "fold_seed": fold_seed,
            "selected_epoch": selected_epoch,
            "inner_train_subjects": subject_list(dataset, inner_train),
            "inner_validation_subjects": subject_list(dataset, inner_val),
            "outer_test_subjects": subject_list(dataset, outer_test),
            "subjects_used_to_fit_reported_checkpoint": outer_training_subjects,
            "refit_mode": getattr(args, "refit_mode", "none"),
            "best_inner_validation": best_inner,
            "final_inner_validation": final_inner,
            "selection_history": history,
            "refit_train_loss": refit_curve,
            "outer_test": outer_metrics,
            "selection_class_weights": selection_weights,
            "refit_class_weights": refit_weights,
        }
        fold_path.write_text(json.dumps(record, indent=2))
        results.append(record)
        partial = {
            "status": "incomplete",
            "completed_outer_folds": len(results),
            "required_outer_folds": args.max_folds or args.outer_folds,
            "model": args.model,
            "imbalance": args.imbalance,
            "seed": args.seed,
        }
        (output / "partial_progress.json").write_text(json.dumps(partial, indent=2))
        checkpoint_archive = getattr(args, "checkpoint_archive", None)
        if checkpoint_archive:
            shutil.make_archive(str(checkpoint_archive), "zip", root_dir=output.parent)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if deadline_unix and time.time() >= deadline_unix:
            print("SAFE STOP after preserving the completed outer fold.", flush=True)
            break

    required_folds = args.max_folds or args.outer_folds
    if len(results) < required_folds:
        partial = {
            "status": "incomplete",
            "completed_outer_folds": len(results),
            "required_outer_folds": required_folds,
            "model": args.model,
            "imbalance": args.imbalance,
            "seed": args.seed,
        }
        (output / "partial_progress.json").write_text(json.dumps(partial, indent=2))
        return partial

    final = {
        "protocol": (
            "10-fold outer subject-wise GroupKFold; grouped inner validation selects the checkpoint; "
            + ("model is reinitialized and refit for the selected epoch; " if getattr(args, "refit_mode", "none") == "selected_epoch" else "selected checkpoint is retained without refitting; ")
            + "outer test subjects are evaluated once"
        ),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "class_names": dataset.class_names,
        "samples": len(dataset),
        "subjects": len(set(dataset.subject_ids.tolist())),
        "class_counts": dataset.class_counts(),
        "compute": compute,
        "aggregate": aggregate(results, dataset.class_names),
        "folds": results,
    }
    (output / "summary.json").write_text(json.dumps(final, indent=2))
    print(json.dumps({"output": str(output), "compute": compute, "aggregate": final["aggregate"]}, indent=2))
    return final


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
