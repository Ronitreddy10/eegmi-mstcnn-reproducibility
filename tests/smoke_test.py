#!/usr/bin/env python3
"""Fast synthetic checks; this does not produce manuscript performance results."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_robustness_study as study


class SyntheticEEG(Dataset):
    def __init__(self):
        self.class_names = ["rest", "left", "right", "feet"]
        self.labels = np.tile(np.arange(4), 8)
        self.subject_ids = np.repeat(np.arange(8), 4)
        self.samples = torch.randn(len(self.labels), 64, study.N_EPOCH_SAMPLES)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], torch.tensor(int(self.labels[index]))


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
    data = SyntheticEEG()
    train_idx = np.arange(24)
    val_idx = np.arange(24, 32)
    for mode in ["none", "effective_ce", "focal", "effective_focal", "oversample"]:
        criterion, _ = study.make_criterion(mode, data.labels, train_idx, 4, device, 0.9999, 2.0)
        logits = torch.randn(8, 4, requires_grad=True)
        loss = criterion(logits, torch.tensor(data.labels[val_idx]))
        loss.backward()
        assert torch.isfinite(loss)
    print("Synthetic smoke checks passed; no EEG performance result was generated.")


if __name__ == "__main__":
    main()
