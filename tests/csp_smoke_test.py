#!/usr/bin/env python3
"""Small synthetic execution check for the subject-wise CSP+LDA runner."""

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_csp_lda


class SyntheticCSPDataset:
    def __init__(self):
        rng = np.random.default_rng(42)
        self.class_names = ["rest", "left", "right", "feet"]
        self.labels = np.tile(np.arange(4), 8)
        self.subject_ids = np.repeat(np.arange(1, 9), 4)
        samples = rng.normal(size=(32, 8, 96)).astype(np.float32)
        for index, label in enumerate(self.labels):
            samples[index, label, :] += np.sin(np.linspace(0, 6 * np.pi, 96))
        self.samples = [torch.from_numpy(sample) for sample in samples]
        self.tmin = 0.0
        self.tmax = 4.0
        self.sampling_frequency_hz = 160.0
        self.preprocessing_metadata = {"synthetic": True}

    def __len__(self):
        return len(self.labels)

    def class_counts(self):
        return {name: int(np.sum(self.labels == index)) for index, name in enumerate(self.class_names)}


def main():
    with tempfile.TemporaryDirectory() as directory:
        args = Namespace(
            data_dir=directory,
            output_dir=str(Path(directory) / "csp"),
            tmin=0.0,
            tmax=4.0,
            outer_folds=2,
            components=4,
            seed=42,
            max_subjects=8,
            max_folds=1,
        )
        result = run_csp_lda.run(args, dataset=SyntheticCSPDataset())
        assert len(result["folds"]) == 1
        predictions = result["folds"][0]["outer_test_predictions"]
        assert len(predictions["y_true"]) == len(predictions["subject_ids"])
        assert (Path(args.output_dir) / "summary.json").exists()
    print("Synthetic CSP+LDA smoke check passed.")


if __name__ == "__main__":
    main()
