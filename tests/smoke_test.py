#!/usr/bin/env python3
"""Fast synthetic checks; this does not produce manuscript performance results."""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_robustness_study as study
import analyze_reviewer_results as reviewer_analysis
import report_receptive_field


class SyntheticEEG(Dataset):
    def __init__(self):
        self.class_names = ["rest", "left", "right", "feet"]
        self.labels = np.tile(np.arange(4), 8)
        self.subject_ids = np.repeat(np.arange(8), 4)
        self.samples = torch.randn(len(self.labels), 64, study.N_EPOCH_SAMPLES)
        self.tmin = 0.0
        self.tmax = 4.0
        self.sampling_frequency_hz = 160.0
        self.preprocessing_metadata = {"synthetic": True}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], torch.tensor(int(self.labels[index]))

    def class_counts(self):
        return {name: int(np.sum(self.labels == index)) for index, name in enumerate(self.class_names)}


def main():
    device = torch.device("cpu")
    expected = {
        "mstcnn_0_5m": 551_144,
        "mstcnn_5m": 4_956_614,
        "mstcnn_20m": 19_936_294,
        "mstcnn_83m": 83_080_458,
        "eegnet": 3_412,
        "shallowconvnet": 109_404,
        "deepconvnet1d": 1_062_276,
        "resnet1d": 1_949_316,
    }
    for name, count in expected.items():
        model = study.build_model(name, 4, 0.5, device)
        assert study.parameter_count(model) == count
        with torch.no_grad():
            assert model(torch.zeros(1, 64, study.N_EPOCH_SAMPLES)).shape == (1, 4)
        del model
    for kernels in ([7], [7, 9], [7, 9, 11], [7, 9, 11, 13]):
        model = study.build_model("mstcnn_0_5m", 4, 0.5, device, kernels=kernels, epoch_samples=321)
        with torch.no_grad():
            assert model(torch.zeros(1, 64, 321)).shape == (1, 4)
        assert model.kernels == tuple(kernels)
    assert report_receptive_field.receptive_field(7)[0] == 43
    assert report_receptive_field.receptive_field(13)[0] == 85
    data = SyntheticEEG()
    train_idx = np.arange(24)
    val_idx = np.arange(24, 32)
    for mode in ["none", "effective_ce", "focal", "effective_focal", "oversample"]:
        criterion, _ = study.make_criterion(mode, data.labels, train_idx, 4, device, 0.9999, 2.0)
        logits = torch.randn(8, 4, requires_grad=True)
        loss = criterion(logits, torch.tensor(data.labels[val_idx]))
        loss.backward()
        assert torch.isfinite(loss)
    evaluation_args = Namespace(
        imbalance="none",
        effective_beta=0.9999,
        focal_gamma=2.0,
        batch_size=8,
        num_workers=0,
    )
    evaluation_model = study.build_model("mstcnn_0_5m", 4, 0.5, device, kernels=[7, 9])
    metrics, weights, y_true, y_pred = study.evaluate_selected_model(
        evaluation_args, evaluation_model, data, train_idx, val_idx, device
    )
    assert weights is None and len(y_true) == len(y_pred) == len(val_idx)
    assert "per_class_recall" in metrics
    synthetic_summary = {
        "class_names": data.class_names,
        "folds": [
            {
                "outer_fold": 1,
                "outer_test_predictions": {
                    "sample_indices": list(range(16)),
                    "subject_ids": data.subject_ids[:16].tolist(),
                    "y_true": data.labels[:16].tolist(),
                    "y_pred": data.labels[:16].tolist(),
                },
            },
            {
                "outer_fold": 2,
                "outer_test_predictions": {
                    "sample_indices": list(range(16, 32)),
                    "subject_ids": data.subject_ids[16:].tolist(),
                    "y_true": data.labels[16:].tolist(),
                    "y_pred": data.labels[16:].tolist(),
                },
            },
        ],
    }
    rows, cms, subjects = reviewer_analysis.subject_rows_and_confusions("synthetic", synthetic_summary)
    assert len(rows) == len(subjects) == 8
    ci = reviewer_analysis.subject_block_bootstrap(cms, n_boot=100, seed=42)
    assert all(item["estimate"] == 1.0 for item in ci)
    print("Synthetic smoke checks passed; no EEG performance result was generated.")


if __name__ == "__main__":
    main()
