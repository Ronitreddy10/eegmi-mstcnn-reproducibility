#!/usr/bin/env python3
"""
Stage-wise ablation runner for the Avola-style PhysioNet EEG MI pipeline.

Use this script to prove each proposed block before claiming it helps:
  1. dataset/class mapping
  2. preprocessing
  3. 4-stream CNN baseline
  4. optional input channel attention
  5. optional 4-class task

Examples:
  python eeg_stage_ablation.py --class_mode binary_lr --preprocess none --attention none --epochs 30
  python eeg_stage_ablation.py --class_mode binary_lr --preprocess bandpass_zscore --attention input_se --epochs 30
  python eeg_stage_ablation.py --class_mode mi4 --preprocess bandpass_zscore --attention input_se --epochs 30
  python eeg_stage_ablation.py --analyze_only --class_mode mi4 --preprocess bandpass_zscore
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import copy
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import torch.nn as nn
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eeg_stage_ablation")


def index_attached_edfs():
    """Index an optional read-only EEGMMIDB copy attached to a Kaggle notebook."""
    source_dir = os.environ.get("EEGMMIDB_SOURCE_DIR", "").strip()
    if not source_dir:
        return {}
    root = Path(source_dir)
    if not root.exists():
        logger.warning("EEGMMIDB_SOURCE_DIR does not exist: %s", root)
        return {}
    pattern = re.compile(r"^S(\d{3})R(\d{2})\.edf$", re.IGNORECASE)
    index = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            index[(int(match.group(1)), int(match.group(2)))] = str(path)
    logger.info("Indexed %d attached EEGMMIDB EDF files under %s", len(index), root)
    return index


def resolve_edf_paths(subject_id, runs, data_dir, attached_index=None):
    """Prefer attached EDF files; download through MNE only when any run is absent."""
    attached_index = attached_index or {}
    local = [attached_index.get((int(subject_id), int(run))) for run in runs]
    if all(local):
        return local
    return mne.datasets.eegbci.load_data(
        subject_id, runs, path=data_dir, update_path=False
    )


FILTER_BANKS = [
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
    (32.0, 36.0),
    (36.0, 40.0),
]

MOTOR_CHANNELS = [
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
]


CLASS_CONFIGS = {
    # Paper-style motor-imagery left/right only.
    "binary_lr": {
        "runs": [4, 8, 12],
        "class_names": ["left_fist_imagery", "right_fist_imagery"],
    },
    # Standard four motor-imagery classes, no rest.
    "mi4": {
        "runs": [4, 6, 8, 10, 12, 14],
        "class_names": [
            "left_fist_imagery",
            "right_fist_imagery",
            "both_fists_imagery",
            "both_feet_imagery",
        ],
    },
    # Your current four-class extension: rest replaces both-fists.
    "mi4_rest": {
        "runs": [4, 6, 8, 10, 12, 14],
        "class_names": [
            "rest",
            "left_fist_imagery",
            "right_fist_imagery",
            "both_feet_imagery",
        ],
    },
}


def map_physionet_event(class_mode, run, event_name):
    if event_name == "T0":
        return 0 if class_mode == "mi4_rest" else None

    lr_imagery = run in {4, 8, 12}
    fists_feet_imagery = run in {6, 10, 14}

    if class_mode == "binary_lr":
        if lr_imagery and event_name == "T1":
            return 0
        if lr_imagery and event_name == "T2":
            return 1
        return None

    if class_mode == "mi4":
        if lr_imagery and event_name == "T1":
            return 0
        if lr_imagery and event_name == "T2":
            return 1
        if fists_feet_imagery and event_name == "T1":
            return 2
        if fists_feet_imagery and event_name == "T2":
            return 3
        return None

    if class_mode == "mi4_rest":
        if lr_imagery and event_name == "T1":
            return 1
        if lr_imagery and event_name == "T2":
            return 2
        if fists_feet_imagery and event_name == "T2":
            return 3
        return None

    raise ValueError(f"Unknown class_mode: {class_mode}")


class PhysioNetStageDataset(Dataset):
    EXCLUDED_SUBJECTS = {43, 88, 89, 92, 100, 104}

    def __init__(
        self,
        data_dir="./eeg_data",
        class_mode="binary_lr",
        preprocess="none",
        feature_mode="raw",
        channel_set="all",
        fb_windows=4,
        subjects=None,
        tmin=-0.5,
        tmax=4.1,
        max_subjects=None,
        balance_classes=False,
    ):
        self.data_dir = data_dir
        self.class_mode = class_mode
        self.preprocess = preprocess
        self.feature_mode = feature_mode
        self.channel_set = channel_set
        self.fb_windows = fb_windows
        self.tmin = tmin
        self.tmax = tmax
        self.balance_classes = balance_classes
        self.class_names = CLASS_CONFIGS[class_mode]["class_names"]
        self.samples = []
        self.labels = []
        self.subject_ids = []
        self.channel_names = None
        self.attached_edf_index = index_attached_edfs()

        os.makedirs(data_dir, exist_ok=True)
        if subjects is None:
            subjects = [s for s in range(1, 110) if s not in self.EXCLUDED_SUBJECTS]
        if max_subjects:
            subjects = subjects[:max_subjects]

        logger.info(
            "Loading class_mode=%s preprocess=%s feature_mode=%s channel_set=%s subjects=%d runs=%s",
            class_mode,
            preprocess,
            feature_mode,
            channel_set,
            len(subjects),
            CLASS_CONFIGS[class_mode]["runs"],
        )
        for i, subject_id in enumerate(subjects, start=1):
            try:
                self._load_subject(subject_id)
            except Exception as exc:
                logger.warning("Skipping S%03d: %s", subject_id, exc)
            if i % 20 == 0 or i == len(subjects):
                logger.info("Loaded %d/%d subjects; samples=%d", i, len(subjects), len(self.samples))

        if not self.samples:
            raise RuntimeError("No samples were loaded. Check data_dir/class_mode.")
        self.labels = np.asarray(self.labels)
        self.subject_ids = np.asarray(self.subject_ids)
        if self.feature_mode == "fb_logvar":
            self._standardize_features()
        if self.balance_classes:
            self._balance_classes()
        logger.info("Class counts: %s", self.class_counts())

    def _standardize_features(self):
        x = torch.stack(self.samples)
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.samples = [((sample - mean.squeeze(0)) / std.squeeze(0)).float() for sample in self.samples]

    def _balance_classes(self):
        rng = np.random.default_rng(42)
        labels = np.asarray(self.labels)
        by_class = {class_idx: np.where(labels == class_idx)[0] for class_idx in sorted(set(labels.tolist()))}
        min_count = min(len(idx) for idx in by_class.values())
        keep = []
        for idx in by_class.values():
            keep.extend(rng.choice(idx, size=min_count, replace=False).tolist())
        keep = sorted(keep)
        self.samples = [self.samples[i] for i in keep]
        self.labels = np.asarray([int(labels[i]) for i in keep])
        self.subject_ids = np.asarray([int(self.subject_ids[i]) for i in keep])
        logger.info("Balanced classes by undersampling to %d samples/class", min_count)

    def _load_subject(self, subject_id):
        runs = CLASS_CONFIGS[self.class_mode]["runs"]
        edf_paths = resolve_edf_paths(
            subject_id, runs, self.data_dir, self.attached_edf_index
        )
        raws = []
        run_numbers = []
        for run, path in zip(runs, edf_paths):
            raw = mne.io.read_raw_edf(path, preload=True, stim_channel="auto", verbose=False)
            mne.datasets.eegbci.standardize(raw)
            if self.channel_set == "motor":
                available = [ch for ch in MOTOR_CHANNELS if ch in raw.ch_names]
                raw.pick_channels(available, ordered=True)
            if self.preprocess in {"bandpass", "bandpass_zscore"} and self.feature_mode == "raw":
                raw.filter(l_freq=4.0, h_freq=40.0, fir_design="firwin", verbose=False)
            raws.append(raw)
            run_numbers.append(run)
            if self.channel_names is None:
                self.channel_names = list(raw.ch_names)

        for run, raw in zip(run_numbers, raws):
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
            inv_event_id = {code: name for name, code in event_id.items()}

            if self.feature_mode == "fb_logvar":
                data, labels = self._filterbank_logvar(raw, events, event_id, picks)
            else:
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_id,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    proj=False,
                    picks=picks,
                    baseline=None,
                    preload=True,
                    verbose=False,
                )
                data = epochs.get_data()
                labels = epochs.events[:, 2]

                if self.preprocess in {"zscore", "bandpass_zscore"}:
                    mean = data.mean(axis=-1, keepdims=True)
                    std = data.std(axis=-1, keepdims=True)
                    data = (data - mean) / (std + 1e-6)

            for x_np, code in zip(data, labels):
                event_name = inv_event_id.get(int(code))
                mapped = self._map_event(run, event_name)
                if mapped is None:
                    continue
                self.samples.append(torch.from_numpy(x_np).float())
                self.labels.append(mapped)
                self.subject_ids.append(subject_id)

    def _filterbank_logvar(self, raw, events, event_id, picks):
        band_features = []
        labels = None
        for l_freq, h_freq in FILTER_BANKS:
            band_raw = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
            epochs = mne.Epochs(
                band_raw,
                events,
                event_id=event_id,
                tmin=self.tmin,
                tmax=self.tmax,
                proj=False,
                picks=picks,
                baseline=None,
                preload=True,
                verbose=False,
            )
            data = epochs.get_data()
            if labels is None:
                labels = epochs.events[:, 2]
            chunks = np.array_split(data, self.fb_windows, axis=-1)
            logvars = [np.log(np.var(chunk, axis=-1) + 1e-8) for chunk in chunks]
            band_features.append(np.stack(logvars, axis=-1))
        return np.stack(band_features, axis=1), labels

    def _map_event(self, run, event_name):
        return map_physionet_event(self.class_mode, run, event_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], torch.tensor(int(self.labels[idx]), dtype=torch.long)

    def class_counts(self):
        counts = Counter(self.labels.tolist())
        return {self.class_names[k]: counts.get(k, 0) for k in range(len(self.class_names))}


class LabelMappedSubset(Dataset):
    """
    Subset wrapper that remaps labels for hierarchical training.

    Example:
      original labels: 0=rest, 1=left, 2=right, 3=feet
      stage 1 labels: 0=rest, 1=motor
      stage 2 labels: 0=left, 1=right, 2=feet
    """

    def __init__(self, dataset, indices, label_fn, class_names):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.label_fn = label_fn
        self.class_names = list(class_names)
        self.labels = np.asarray(
            [int(label_fn(int(dataset.labels[i]))) for i in self.indices],
            dtype=np.int64,
        )
        self.subject_ids = np.asarray(
            [int(dataset.subject_ids[i]) for i in self.indices],
            dtype=np.int64,
        )
        self.channel_names = dataset.channel_names

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x, _ = self.dataset[original_idx]
        return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)

    def class_counts(self):
        counts = Counter(self.labels.tolist())
        return {self.class_names[k]: counts.get(k, 0) for k in range(len(self.class_names))}


class WindowedSubset(Dataset):
    """
    Train/eval wrapper for cropped EEG windows.

    Cropped training is common in EEG CNN work because one 4-second trial contains
    several informative motor-imagery subwindows. At evaluation time we keep it
    deterministic; test-time averaging is handled in evaluate().
    """

    def __init__(
        self,
        dataset,
        indices,
        crop_samples=0,
        random_crop=False,
        time_mask_prob=0.0,
        channel_dropout_prob=0.0,
    ):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.crop_samples = int(crop_samples or 0)
        self.random_crop = random_crop
        self.time_mask_prob = float(time_mask_prob)
        self.channel_dropout_prob = float(channel_dropout_prob)
        self.labels = np.asarray([int(dataset.labels[i]) for i in self.indices], dtype=np.int64)
        self.subject_ids = np.asarray([int(dataset.subject_ids[i]) for i in self.indices], dtype=np.int64)
        self.class_names = dataset.class_names
        self.channel_names = dataset.channel_names

    def __len__(self):
        return len(self.indices)

    def _crop(self, x):
        if self.crop_samples <= 0 or x.ndim != 2 or x.shape[-1] <= self.crop_samples:
            return x
        max_start = x.shape[-1] - self.crop_samples
        if self.random_crop:
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start = max_start // 2
        return x[:, start : start + self.crop_samples]

    def _augment(self, x):
        if self.channel_dropout_prob > 0 and torch.rand(()) < self.channel_dropout_prob:
            n_drop = max(1, int(round(0.05 * x.shape[0])))
            channels = torch.randperm(x.shape[0])[:n_drop]
            x = x.clone()
            x[channels] = 0
        if self.time_mask_prob > 0 and torch.rand(()) < self.time_mask_prob:
            mask_len = max(8, int(round(0.08 * x.shape[-1])))
            if x.shape[-1] > mask_len:
                start = int(torch.randint(0, x.shape[-1] - mask_len + 1, (1,)).item())
                x = x.clone()
                x[:, start : start + mask_len] = 0
        return x

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x, y = self.dataset[original_idx]
        x = self._crop(x)
        if self.random_crop:
            x = self._augment(x)
        return x, y


class ChannelAttention(nn.Module):
    def __init__(self, in_channels=64, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def weights(self, x):
        pooled = x.mean(dim=-1)
        return self.sigmoid(self.fc2(self.relu(self.fc1(pooled))))

    def forward(self, x):
        return x * self.weights(x).unsqueeze(-1)


class StatsChannelAttention(nn.Module):
    """
    Input electrode attention that still has signal after per-channel z-score.

    A plain squeeze-and-excitation block using x.mean(time) fails after z-score
    because every channel has near-zero temporal mean. This block uses several
    per-channel descriptors: mean, mean absolute amplitude, and max absolute
    amplitude. The latter two still vary after z-score and can drive attention.
    """

    def __init__(self, in_channels=64, reduction=8):
        super().__init__()
        hidden = max(in_channels // reduction, 4)
        self.fc1 = nn.Linear(in_channels * 3, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, in_channels)
        self.sigmoid = nn.Sigmoid()

    def weights(self, x):
        mean = x.mean(dim=-1)
        abs_mean = x.abs().mean(dim=-1)
        max_abs = x.abs().amax(dim=-1)
        pooled = torch.cat([mean, abs_mean, max_abs], dim=1)
        return self.sigmoid(self.fc2(self.relu(self.fc1(pooled))))

    def forward(self, x):
        return x * self.weights(x).unsqueeze(-1)


class ECAFeatureAttention(nn.Module):
    """
    Efficient Channel Attention over learned feature maps.

    This follows the idea used in efficient channel attention MI models: compute
    a global descriptor for each learned feature channel, then use a small 1D
    convolution to model local channel interactions before reweighting features.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        return x * self.sigmoid(y)


class ResidualTCNBlock(nn.Module):
    """Small dilated temporal block for learned feature maps."""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.ELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class TemporalAttentionPool(nn.Module):
    """Attention pooling over time for class-relevant temporal segments."""

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1)


class TemporalMultiPool(nn.Module):
    """Combine max, mean, and attention-pooled temporal summaries."""

    def __init__(self, channels):
        super().__init__()
        self.attn = TemporalAttentionPool(channels)

    def forward(self, x):
        max_pool = x.amax(dim=-1)
        mean_pool = x.mean(dim=-1)
        attn_pool = self.attn(x)
        return torch.cat([max_pool, mean_pool, attn_pool], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, mid_channels, out_channels, use_pooling=True):
        super().__init__()
        layers = OrderedDict()
        layers["conv_1"] = nn.Conv1d(in_channels, mid_channels, kernel_size, padding=kernel_size // 2)
        layers["activation_1"] = nn.ReLU()
        layers["conv_2"] = nn.Conv1d(mid_channels, out_channels, kernel_size, padding=kernel_size // 2)
        layers["activation_2"] = nn.ReLU()
        if use_pooling:
            layers["pooling"] = nn.Conv1d(
                out_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=2
            )
        self.block = nn.Sequential(layers)

    def forward(self, x):
        return self.block(x)


class MultiStream1DCNN(nn.Module):
    def __init__(
        self,
        in_channels=64,
        n_classes=2,
        n_streams=4,
        starting_kernel_size=7,
        max_channels=256,
        adaptive_pool_size=48,
        n_stream_blocks=2,
        dropout_p=0.5,
        attention="none",
        temporal_model="none",
        tcn_layers=3,
        tcn_dropout=0.3,
        classifier_hidden=1024,
    ):
        super().__init__()
        self.attention_mode = attention
        self.temporal_model = temporal_model
        if attention in {"input_se", "input_se_stream_eca"}:
            self.attention = ChannelAttention(in_channels)
        elif attention in {"input_stats_se", "input_stats_stream_eca"}:
            self.attention = StatsChannelAttention(in_channels)
        else:
            self.attention = nn.Identity()

        use_stream_eca = attention in {"stream_eca", "input_se_stream_eca", "input_stats_stream_eca"}
        self.streams = nn.ModuleList()
        self.stream_attentions = nn.ModuleList()
        for i in range(n_streams):
            kernel_size = starting_kernel_size + i * 2
            channels_up = np.geomspace(in_channels, max_channels, n_stream_blocks, endpoint=True, dtype=int)
            channels_down = np.geomspace(max_channels, 64, n_stream_blocks + 1, endpoint=True, dtype=int)
            channels = np.concatenate([channels_up, channels_down])
            blocks = []
            for b in range(n_stream_blocks):
                blocks.append(
                    ConvBlock(
                        kernel_size,
                        int(channels[b * 2]),
                        int(channels[b * 2 + 1]),
                        int(channels[b * 2 + 2]),
                        use_pooling=(b < n_stream_blocks - 1),
                    )
                )
            self.streams.append(nn.Sequential(*blocks))
            self.stream_attentions.append(ECAFeatureAttention(64) if use_stream_eca else nn.Identity())

        fused_channels = 64 * n_streams
        if temporal_model in {"tcn", "tcn_attn", "tcn_pool8", "tcn_multipool"}:
            self.temporal_encoder = nn.Sequential(
                *[
                    ResidualTCNBlock(
                        fused_channels,
                        kernel_size=3,
                        dilation=2**i,
                        dropout=tcn_dropout,
                    )
                    for i in range(tcn_layers)
                ]
            )
        else:
            self.temporal_encoder = nn.Identity()

        if temporal_model == "tcn_attn":
            self.temporal_pool = TemporalAttentionPool(fused_channels)
            classifier_in = fused_channels
        elif temporal_model == "tcn_pool8":
            self.temporal_pool = nn.Sequential(nn.AdaptiveMaxPool1d(8), nn.Flatten(start_dim=1))
            classifier_in = fused_channels * 8
        elif temporal_model == "tcn_multipool":
            self.temporal_pool = TemporalMultiPool(fused_channels)
            classifier_in = fused_channels * 3
        else:
            self.temporal_pool = nn.Sequential(nn.AdaptiveMaxPool1d(adaptive_pool_size), nn.Flatten(start_dim=1))
            classifier_in = fused_channels * adaptive_pool_size

        if classifier_hidden and classifier_hidden > 0:
            hidden = int(classifier_hidden)
        else:
            # Avola-style classifier width: 12288 -> 6146 for four classes.
            hidden = (classifier_in + n_classes) // 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(classifier_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x = self.attention(x)
        outs = []
        for stream, stream_attention in zip(self.streams, self.stream_attentions):
            out = stream(x)
            out = stream_attention(out)
            outs.append(out)
        fused = torch.cat(outs, dim=1)
        fused = self.temporal_encoder(fused)
        return self.classifier(self.temporal_pool(fused))


class FBPowerNet(nn.Module):
    """
    Compact FBCSP/FBCNet-inspired classifier for filter-bank log-variance maps.

    Input shape: (batch, n_bands, n_channels, n_windows). The model first learns
    local spectral filters, then a spatial projection across electrodes, then
    pools short temporal windows. This is deliberately biased toward classical
    motor-imagery evidence: mu/beta bandpower over motor cortex.
    """

    def __init__(
        self,
        n_bands,
        n_channels,
        n_windows,
        n_classes,
        spectral_filters=32,
        spatial_filters=64,
        dropout_p=0.5,
        hidden=256,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_windows = n_windows
        self.spectral = nn.Sequential(
            nn.Conv2d(1, spectral_filters, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(spectral_filters),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(spectral_filters, spatial_filters, kernel_size=(1, n_channels), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )
        pooled_features = spatial_filters * 6
        self.temporal_score = nn.Linear(spatial_filters * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_features, hidden),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        # (B, bands, channels, windows) -> (B*windows, 1, bands, channels)
        b, bands, channels, windows = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b * windows, 1, bands, channels)
        x = self.spectral(x)
        x = self.spatial(x).squeeze(-1)
        # Frequency max+mean pooling -> (B, windows, spatial_filters * 2)
        x = torch.cat([x.amax(dim=-1), x.mean(dim=-1)], dim=1)
        x = x.view(b, windows, -1)
        mean_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        attn = torch.softmax(self.temporal_score(x), dim=1)
        attn_pool = (x * attn).sum(dim=1)
        # Max pooling is a useful safeguard for short discriminative ERD/ERS bursts.
        return self.classifier(torch.cat([mean_pool, max_pool, attn_pool], dim=1))


class EEGNetBaseline(nn.Module):
    """
    Compact EEGNet-style baseline.

    This is included as a reviewer-facing comparator, not as the proposed model.
    Input is (batch, channels, time), matching the rest of this script.
    """

    def __init__(self, n_channels, n_times, n_classes, dropout_p=0.5, f1=8, depth_multiplier=2, f2=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth_multiplier, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_p),
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_p),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))


class ShallowConvNetBaseline(nn.Module):
    """
    Shallow ConvNet-style baseline inspired by Braindecode.

    This is a second reviewer-facing neural baseline commonly used for EEG MI.
    """

    def __init__(self, n_channels, n_times, n_classes, dropout_p=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False)
        self.spatial = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout_p)
        with torch.no_grad():
            flat = self._features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def _features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        return self.classifier(self._features(x.unsqueeze(1)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_dict(y_true, y_pred):
    labels = np.arange(int(max(np.max(y_true), np.max(y_pred))) + 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def metrics_for_json(metrics, include_loss=True):
    out = {}
    for key, value in metrics.items():
        if key == "loss" and not include_loss:
            continue
        if isinstance(value, (list, tuple)):
            out[key] = value
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = float(value)
    return out


def crop_sample_count(args, dataset):
    if getattr(args, "train_crop_seconds", 0.0) <= 0:
        return 0
    sample = dataset[0][0]
    if sample.ndim != 2:
        return 0
    # PhysioNet EEGMMI is 160 Hz. Use the actual epoch length as a safeguard.
    return min(sample.shape[-1], max(1, int(round(args.train_crop_seconds * 160.0))))


def fixed_time_crops(x, crop_samples, n_crops):
    if crop_samples <= 0 or x.ndim != 3 or x.shape[-1] <= crop_samples or n_crops <= 1:
        return [x]
    max_start = x.shape[-1] - crop_samples
    offsets = np.linspace(0, max_start, num=n_crops).round().astype(int).tolist()
    return [x[:, :, start : start + crop_samples] for start in offsets]


def summarize_metric(values):
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    return {
        "mean": float(np.mean(values)),
        "std": std,
        "ci95": float(1.96 * std / np.sqrt(max(len(values), 1))),
    }


def compare_against_summary(current_fold_results, reference_path):
    if not reference_path:
        return None
    if not os.path.exists(reference_path):
        logger.warning("Comparison summary not found: %s", reference_path)
        return None
    with open(reference_path, "r") as f:
        reference = json.load(f)
    ref_folds = reference.get("folds", [])
    n = min(len(current_fold_results), len(ref_folds))
    if n == 0:
        return None
    comparison = {}
    for metric in ["accuracy", "balanced_accuracy", "f1"]:
        if metric not in current_fold_results[0]["best_metrics"] or metric not in ref_folds[0].get("best_metrics", {}):
            continue
        current_values = np.asarray([current_fold_results[i]["best_metrics"][metric] for i in range(n)], dtype=np.float64)
        reference_values = np.asarray([ref_folds[i]["best_metrics"][metric] for i in range(n)], dtype=np.float64)
        diff = current_values - reference_values
        item = {
            "n_folds": int(n),
            "current_mean": float(np.mean(current_values)),
            "reference_mean": float(np.mean(reference_values)),
            "mean_difference": float(np.mean(diff)),
        }
        try:
            from scipy.stats import wilcoxon

            item["wilcoxon_p"] = float(wilcoxon(current_values, reference_values).pvalue)
        except Exception as exc:
            item["wilcoxon_p"] = None
            item["wilcoxon_note"] = str(exc)
        comparison[metric] = item
    return comparison


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            preds.append(logits.argmax(dim=-1).cpu())
            labels.append(y.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    out = metrics_dict(y_true, y_pred)
    out["loss"] = total_loss / max(len(loader), 1)
    return out, y_true, y_pred


def evaluate_with_crops(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    crop_samples = int(getattr(args, "_crop_samples", 0) or 0)
    eval_crops = int(getattr(args, "eval_crops", 1) or 1)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits_per_crop = []
            for x_crop in fixed_time_crops(x, crop_samples, eval_crops):
                logits_per_crop.append(model(x_crop))
            logits = torch.stack(logits_per_crop, dim=0).mean(dim=0)
            total_loss += criterion(logits, y).item()
            preds.append(logits.argmax(dim=-1).cpu())
            labels.append(y.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    out = metrics_dict(y_true, y_pred)
    out["loss"] = total_loss / max(len(loader), 1)
    return out, y_true, y_pred


def make_class_weighted_criterion(dataset, train_idx, n_classes, device, mode):
    if mode == "none":
        return nn.CrossEntropyLoss()
    labels = np.asarray(dataset.labels)[train_idx]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (n_classes * counts)
    logger.info("Class-weighted loss weights: %s", weights.round(4).tolist())
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def make_balanced_sampler(dataset, train_idx, n_classes):
    labels = np.asarray(dataset.labels)[train_idx]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    sample_weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_idx),
        replacement=True,
    )


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def build_multistream_model(args, n_classes, device):
    return build_neural_model(args, n_classes, device)


def build_neural_model(args, n_classes, device, sample_shape=None):
    if args.architecture == "eegnet":
        if sample_shape is None:
            raise ValueError("architecture=eegnet requires sample_shape")
        return EEGNetBaseline(
            n_channels=sample_shape[0],
            n_times=sample_shape[1],
            n_classes=n_classes,
            dropout_p=args.dropout,
        ).to(device)
    if args.architecture == "shallowconvnet":
        if sample_shape is None:
            raise ValueError("architecture=shallowconvnet requires sample_shape")
        return ShallowConvNetBaseline(
            n_channels=sample_shape[0],
            n_times=sample_shape[1],
            n_classes=n_classes,
            dropout_p=args.dropout,
        ).to(device)
    return MultiStream1DCNN(
        n_classes=n_classes,
        attention=args.attention,
        dropout_p=args.dropout,
        temporal_model=args.temporal_model,
        tcn_layers=args.tcn_layers,
        tcn_dropout=args.tcn_dropout,
        classifier_hidden=args.classifier_hidden,
    ).to(device)


def train_component_model(name, model, train_dataset, val_dataset, args, device, fold_dir):
    criterion = make_class_weighted_criterion(
        train_dataset,
        np.arange(len(train_dataset)),
        len(train_dataset.class_names),
        device,
        args.class_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_sampler = (
        make_balanced_sampler(train_dataset, np.arange(len(train_dataset)), len(train_dataset.class_names))
        if args.balanced_sampler
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    best_score = float("inf") if args.monitor_metric == "loss" else -1.0
    best_metrics = None
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_metrics["loss"], 5),
            "val_accuracy": round(val_metrics["accuracy"], 5),
            "val_f1": round(val_metrics["f1"], 5),
        }
        history.append(row)
        current_score = val_metrics["loss"] if args.monitor_metric == "loss" else val_metrics[args.monitor_metric]
        improved = current_score < best_score if args.monitor_metric == "loss" else current_score > best_score
        if improved:
            best_score = current_score
            best_metrics = metrics_for_json(val_metrics, include_loss=False)
            best_metrics["epoch"] = epoch
            best_metrics["monitor_metric"] = args.monitor_metric
            best_metrics["monitor_score"] = float(current_score)
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if epoch % args.log_every == 0 or epoch == 1:
            logger.info(
                "%s epoch=%d train_loss=%.4f val_acc=%.4f val_f1=%.4f best_%s=%.4f",
                name,
                epoch,
                train_loss,
                val_metrics["accuracy"],
                val_metrics["f1"],
                args.monitor_metric,
                best_score,
            )
        if patience >= args.early_stop_patience:
            logger.info("Early stop %s epoch=%d", name, epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    plot_learning_curve(history, os.path.join(fold_dir, f"{name}_learning_curve.png"))
    return best_metrics, history


@torch.no_grad()
def evaluate_hierarchical(rest_model, mi_model, loader, device):
    rest_model.eval()
    mi_model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        rest_pred = rest_model(x).argmax(dim=-1)
        mi_pred = mi_model(x).argmax(dim=-1) + 1
        pred = torch.where(rest_pred == 0, torch.zeros_like(mi_pred), mi_pred)
        preds.append(pred.cpu())
        labels.append(y.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return metrics_dict(y_true, y_pred), y_true, y_pred


def plot_learning_curve(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_accuracy"] for h in history], color="tab:green", label="val acc")
    ax2.set_ylabel("Accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_channel_bars(channel_scores, title, out_path, top_n=10):
    order = np.argsort(channel_scores)[::-1][:top_n]
    names = [channel_scores.channel_names[i] if hasattr(channel_scores, "channel_names") else str(i) for i in order]


def signal_channel_importance(dataset, out_dir):
    """Rank channels per class using class-vs-rest absolute mean bandpower difference."""
    os.makedirs(out_dir, exist_ok=True)
    x = torch.stack(dataset.samples).numpy()
    y = dataset.labels
    if x.ndim == 4:
        # Filter-bank log-variance features: average across bands and windows,
        # leaving one discriminability score per electrode.
        power = np.mean(x, axis=(1, 3))
    else:
        power = np.mean(x * x, axis=-1)
    summary = {}
    for class_idx, class_name in enumerate(dataset.class_names):
        in_class = power[y == class_idx]
        rest = power[y != class_idx]
        scores = np.abs(in_class.mean(axis=0) - rest.mean(axis=0)) / (rest.std(axis=0) + 1e-6)
        order = np.argsort(scores)[::-1]
        top3 = [(dataset.channel_names[i], float(scores[i])) for i in order[:3]]
        summary[class_name] = top3

        fig, ax = plt.subplots(figsize=(10, 4))
        top = order[:12]
        ax.bar([dataset.channel_names[i] for i in top], scores[top])
        ax.set_title(f"Top channel discriminability: {class_name}")
        ax.set_ylabel("|class-rest mean power| / rest std")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"channels_{class_name}.png"), dpi=180)
        plt.close(fig)
    with open(os.path.join(out_dir, "top3_channels_signal.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def attention_channel_importance(model, dataset, device, out_dir):
    attention = getattr(model, "attention", None)
    if not isinstance(attention, (ChannelAttention, StatsChannelAttention)):
        return None
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    buckets = defaultdict(list)
    with torch.no_grad():
        for x, y in dataset:
            weights = attention.weights(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            buckets[int(y)].append(weights)

    summary = {}
    for class_idx, values in buckets.items():
        scores = np.mean(values, axis=0)
        order = np.argsort(scores)[::-1]
        class_name = dataset.class_names[class_idx]
        summary[class_name] = [(dataset.channel_names[i], float(scores[i])) for i in order[:3]]
        fig, ax = plt.subplots(figsize=(10, 4))
        top = order[:12]
        ax.bar([dataset.channel_names[i] for i in top], scores[top])
        ax.set_title(f"Learned input-SE attention: {class_name}")
        ax.set_ylabel("Mean attention weight")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"attention_{class_name}.png"), dpi=180)
        plt.close(fig)
    with open(os.path.join(out_dir, "top3_channels_attention.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def saliency_channel_importance(model, dataset, device, out_dir, indices=None, max_per_class=64):
    """
    Class-specific electrode importance using input-gradient saliency.

    This is the evidence plot we need for "which 3-4 of the 64 electrodes matter
    for each class." Unlike plain input attention, saliency is conditioned on the
    target class score, so the result can differ for left/right/fists/feet.
    """
    if max_per_class <= 0:
        return None
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    if indices is None:
        indices = range(len(dataset))

    by_class = defaultdict(list)
    for idx in indices:
        y = int(dataset.labels[idx])
        if len(by_class[y]) < max_per_class:
            by_class[y].append(idx)

    summary = {}
    for class_idx, class_indices in by_class.items():
        scores = []
        for idx in class_indices:
            x, _ = dataset[idx]
            x = x.unsqueeze(0).to(device)
            x.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            logits[0, class_idx].backward()
            # Grad * input highlights channels whose local changes influence the class logit.
            channel_scores = (x.grad.detach().abs() * x.detach().abs()).squeeze(0)
            if channel_scores.ndim == 3:
                channel_scores = channel_scores.mean(dim=(0, 2))
            else:
                channel_scores = channel_scores.mean(dim=-1)
            scores.append(channel_scores.cpu().numpy())
        mean_scores = np.mean(scores, axis=0)
        order = np.argsort(mean_scores)[::-1]
        class_name = dataset.class_names[class_idx]
        summary[class_name] = [(dataset.channel_names[i], float(mean_scores[i])) for i in order[:3]]

        fig, ax = plt.subplots(figsize=(10, 4))
        top = order[:12]
        ax.bar([dataset.channel_names[i] for i in top], mean_scores[top])
        ax.set_title(f"Gradient electrode importance: {class_name}")
        ax.set_ylabel("Mean |gradient x input|")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"saliency_{class_name}.png"), dpi=180)
        plt.close(fig)

    with open(os.path.join(out_dir, "top3_channels_saliency.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def sync_check(args):
    os.makedirs(args.output_dir, exist_ok=True)
    subjects = [s for s in range(1, 110) if s not in PhysioNetStageDataset.EXCLUDED_SUBJECTS]
    subjects = subjects[: args.sync_subjects]
    attached_edf_index = index_attached_edfs()
    rows = []
    for subject_id in subjects:
        runs = CLASS_CONFIGS[args.class_mode]["runs"][: args.sync_runs]
        edf_paths = resolve_edf_paths(
            subject_id, runs, args.data_dir, attached_edf_index
        )
        for run, path in zip(runs, edf_paths):
            raw = mne.io.read_raw_edf(path, preload=False, stim_channel="auto", verbose=False)
            mne.datasets.eegbci.standardize(raw)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            inv_event_id = {code: name for name, code in event_id.items()}
            ann_counts = Counter(raw.annotations.description.tolist())
            logger.info(
                "SYNC S%03d R%02d sfreq=%.1f duration=%.2fs event_id=%s annotation_counts=%s",
                subject_id,
                run,
                raw.info["sfreq"],
                raw.times[-1],
                event_id,
                dict(ann_counts),
            )
            for event_idx, event in enumerate(events[: args.sync_events]):
                sample, _, code = event
                onset = sample / raw.info["sfreq"]
                label = inv_event_id[int(code)]
                mapped = map_physionet_event(args.class_mode, run, label)
                class_name = None if mapped is None else CLASS_CONFIGS[args.class_mode]["class_names"][mapped]
                start = onset + args.tmin
                stop = onset + args.tmax
                row = {
                    "subject": subject_id,
                    "run": run,
                    "event_idx": event_idx,
                    "event": label,
                    "class": class_name,
                    "event_onset_s": round(onset, 4),
                    "epoch_start_s": round(start, 4),
                    "epoch_stop_s": round(stop, 4),
                    "window_s": round(args.tmax - args.tmin, 4),
                    "valid_window": bool(start >= 0.0 and stop <= raw.times[-1]),
                }
                rows.append(row)
                logger.info(
                    "  event=%s class=%s onset=%.2fs -> epoch=[%.2f, %.2f] valid=%s",
                    label,
                    class_name,
                    onset,
                    start,
                    stop,
                    row["valid_window"],
                )

    out_path = os.path.join(args.output_dir, "sync_check.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("Sync check saved: %s", out_path)
    return rows


def run_hierarchical(args, dataset, metadata, device):
    """
    Two-stage classifier for imbalanced rest-heavy 4-class setup.

    Stage 1 decides whether the trial is rest or motor imagery.
    Stage 2 only sees motor-imagery trials and decides left/right/feet.
    The final prediction is:
      rest_model says rest  -> rest
      rest_model says motor -> mi_model output + 1
    """
    if dataset.class_names != ["rest", "left_fist_imagery", "right_fist_imagery", "both_feet_imagery"]:
        raise ValueError("architecture=hierarchical currently expects class_mode=mi4_rest")
    if args.feature_mode != "raw":
        raise ValueError("architecture=hierarchical currently expects feature_mode=raw")

    indices = np.arange(len(dataset))
    if args.split_mode == "subject":
        splitter = GroupKFold(n_splits=args.k_folds)
        split_iter = splitter.split(indices, dataset.labels, groups=dataset.subject_ids)
    else:
        splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(indices, dataset.labels)

    fold_results = []
    rest_names = ["rest", "motor_imagery"]
    mi_names = ["left_fist_imagery", "right_fist_imagery", "both_feet_imagery"]
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if fold < args.start_fold:
            logger.info("Skipping hierarchical fold %d/%d (--start_fold=%d)", fold + 1, args.k_folds, args.start_fold)
            continue
        fold_start = time.time()
        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        logger.info("Hierarchical fold %d/%d train=%d val=%d", fold + 1, args.k_folds, len(train_idx), len(val_idx))

        rest_train = LabelMappedSubset(
            dataset,
            train_idx,
            lambda y: 0 if y == 0 else 1,
            rest_names,
        )
        rest_val = LabelMappedSubset(
            dataset,
            val_idx,
            lambda y: 0 if y == 0 else 1,
            rest_names,
        )
        mi_train_idx = [int(i) for i in train_idx if int(dataset.labels[i]) != 0]
        mi_val_idx = [int(i) for i in val_idx if int(dataset.labels[i]) != 0]
        mi_train = LabelMappedSubset(
            dataset,
            mi_train_idx,
            lambda y: y - 1,
            mi_names,
        )
        mi_val = LabelMappedSubset(
            dataset,
            mi_val_idx,
            lambda y: y - 1,
            mi_names,
        )
        logger.info("  Stage 1 counts: %s", rest_train.class_counts())
        logger.info("  Stage 2 counts: %s", mi_train.class_counts())

        rest_model = build_multistream_model(args, 2, device)
        rest_best, rest_history = train_component_model(
            f"fold={fold} rest_vs_motor",
            rest_model,
            rest_train,
            rest_val,
            args,
            device,
            fold_dir,
        )

        mi_model = build_multistream_model(args, 3, device)
        mi_best, mi_history = train_component_model(
            f"fold={fold} mi_subclass",
            mi_model,
            mi_train,
            mi_val,
            args,
            device,
            fold_dir,
        )

        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        combined_metrics, y_true, y_pred = evaluate_hierarchical(rest_model, mi_model, val_loader, device)
        logger.info(
            "fold=%d hierarchical val_acc=%.4f val_f1=%.4f rest_f1=%.4f mi_f1=%.4f",
            fold,
            combined_metrics["accuracy"],
            combined_metrics["f1"],
            rest_best["f1"],
            mi_best["f1"],
        )
        torch.save(
            {
                "rest_model_state_dict": rest_model.state_dict(),
                "mi_model_state_dict": mi_model.state_dict(),
                "fold": fold,
                "metrics": combined_metrics,
                "rest_metrics": rest_best,
                "mi_metrics": mi_best,
                "metadata": metadata,
            },
            os.path.join(args.output_dir, f"fold_{fold}_hierarchical_best.pth"),
        )
        with open(os.path.join(fold_dir, "hierarchical_predictions.json"), "w") as f:
            json.dump(
                {
                    "y_true": y_true.astype(int).tolist(),
                    "y_pred": y_pred.astype(int).tolist(),
                    "class_names": dataset.class_names,
                },
                f,
            )

        fold_result = {
            "fold": fold,
            "best_metrics": {**metrics_for_json(combined_metrics), "epoch": None},
            "rest_stage_metrics": rest_best,
            "mi_stage_metrics": mi_best,
            "rest_history": rest_history,
            "mi_history": mi_history,
            "time_s": round(time.time() - fold_start, 1),
            "attention_top3": None,
            "saliency_top3": None,
        }
        with open(os.path.join(args.output_dir, f"fold_{fold}_metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=2)
        fold_results.append(fold_result)
        del rest_model, mi_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = build_summary(metadata, fold_results, args.compare_to_summary)
    summary["hierarchical_note"] = "Stage 1 rest-vs-motor, stage 2 left-vs-right-vs-feet."
    with open(os.path.join(args.output_dir, "all_folds_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "RESULT acc=%.4f +/- %.4f bal_acc=%.4f +/- %.4f f1=%.4f +/- %.4f",
        summary["accuracy"]["mean"],
        summary["accuracy"]["std"],
        summary["balanced_accuracy"]["mean"],
        summary["balanced_accuracy"]["std"],
        summary["f1"]["mean"],
        summary["f1"]["std"],
    )
    return summary


def run_classical_baseline(args, dataset, metadata):
    if args.feature_mode != "raw":
        raise ValueError("CSP baselines require feature_mode=raw")
    x = None if args.architecture == "majority" else torch.stack(dataset.samples).numpy()
    y = np.asarray(dataset.labels)
    indices = np.arange(len(dataset))
    if args.split_mode == "subject":
        splitter = GroupKFold(n_splits=args.k_folds)
        split_iter = splitter.split(indices, y, groups=dataset.subject_ids)
    else:
        splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(indices, y)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if fold < args.start_fold:
            logger.info("Skipping classical fold %d/%d (--start_fold=%d)", fold + 1, args.k_folds, args.start_fold)
            continue
        logger.info("Classical %s fold %d/%d train=%d val=%d", args.architecture, fold + 1, args.k_folds, len(train_idx), len(val_idx))
        if args.architecture == "majority":
            values, counts = np.unique(y[train_idx], return_counts=True)
            majority_class = int(values[np.argmax(counts)])
            pred = np.full(len(val_idx), majority_class, dtype=int)
            model = None
        else:
            csp = CSP(
                n_components=args.csp_components,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False,
            )
        if args.architecture == "csp_lda":
            classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            model = make_pipeline(csp, classifier)
        elif args.architecture == "csp_svm":
            classifier = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=args.svm_c, gamma=args.svm_gamma, class_weight="balanced"),
            )
            model = make_pipeline(csp, classifier)
        elif args.architecture == "majority":
            pass
        else:
            raise ValueError(f"Unknown classical architecture: {args.architecture}")

        t0 = time.time()
        if model is not None:
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[val_idx])
        metrics = metrics_dict(y[val_idx], pred)
        logger.info(
            "fold=%d %s acc=%.4f bal_acc=%.4f f1=%.4f",
            fold,
            args.architecture,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["f1"],
        )
        fold_result = {
            "fold": fold,
            "train_subjects": sorted(set(np.asarray(dataset.subject_ids)[train_idx].astype(int).tolist())),
            "val_subjects": sorted(set(np.asarray(dataset.subject_ids)[val_idx].astype(int).tolist())),
            "best_metrics": metrics_for_json(metrics),
            "time_s": round(time.time() - t0, 1),
        }
        with open(os.path.join(args.output_dir, f"fold_{fold}_metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=2)
        fold_results.append(fold_result)

    summary = build_summary(metadata, fold_results, args.compare_to_summary)
    with open(os.path.join(args.output_dir, "all_folds_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "RESULT acc=%.4f +/- %.4f bal_acc=%.4f +/- %.4f f1=%.4f +/- %.4f",
        summary["accuracy"]["mean"],
        summary["accuracy"]["std"],
        summary["balanced_accuracy"]["mean"],
        summary["balanced_accuracy"]["std"],
        summary["f1"]["mean"],
        summary["f1"]["std"],
    )
    return summary


def build_summary(metadata, fold_results, compare_to_summary=None):
    best_metrics = [f["best_metrics"] for f in fold_results]
    per_class_f1 = np.asarray([m["per_class_f1"] for m in best_metrics], dtype=float)
    confusion_sum = np.sum(
        np.asarray([m["confusion_matrix"] for m in best_metrics], dtype=int),
        axis=0,
    )
    summary = {
        **metadata,
        "accuracy": summarize_metric([m["accuracy"] for m in best_metrics]),
        "balanced_accuracy": summarize_metric([m["balanced_accuracy"] for m in best_metrics]),
        "precision": summarize_metric([m["precision"] for m in best_metrics]),
        "recall": summarize_metric([m["recall"] for m in best_metrics]),
        "f1": summarize_metric([m["f1"] for m in best_metrics]),
        "per_class_f1": {
            class_name: {
                "mean": float(per_class_f1[:, class_idx].mean()),
                "std": float(per_class_f1[:, class_idx].std()),
            }
            for class_idx, class_name in enumerate(metadata["class_names"])
        },
        "confusion_matrix_sum": confusion_sum.tolist(),
        # Backward-compatible fields used by your older notebooks.
        "mean_accuracy": float(np.mean([m["accuracy"] for m in best_metrics])),
        "std_accuracy": float(np.std([m["accuracy"] for m in best_metrics])),
        "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in best_metrics])),
        "std_balanced_accuracy": float(np.std([m["balanced_accuracy"] for m in best_metrics])),
        "mean_f1": float(np.mean([m["f1"] for m in best_metrics])),
        "std_f1": float(np.std([m["f1"] for m in best_metrics])),
        "folds": fold_results,
    }
    comparison = compare_against_summary(fold_results, compare_to_summary)
    if comparison:
        summary["comparison_to_reference"] = comparison
    return summary


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    if args.sync_check:
        return sync_check(args)

    dataset = PhysioNetStageDataset(
        data_dir=args.data_dir,
        class_mode=args.class_mode,
        preprocess=args.preprocess,
        feature_mode=args.feature_mode,
        channel_set=args.channel_set,
        fb_windows=args.fb_windows,
        tmin=args.tmin,
        tmax=args.tmax,
        max_subjects=args.max_subjects,
        balance_classes=args.balance_classes,
    )
    metadata = {
        "class_mode": args.class_mode,
        "class_names": dataset.class_names,
        "class_counts": dataset.class_counts(),
        "preprocess": args.preprocess,
        "feature_mode": args.feature_mode,
        "channel_set": args.channel_set,
        "fb_windows": args.fb_windows,
        "architecture": args.architecture,
        "class_weights": args.class_weights,
        "balanced_sampler": args.balanced_sampler,
        "monitor_metric": args.monitor_metric,
        "attention": args.attention,
        "temporal_model": args.temporal_model,
        "split_mode": args.split_mode,
        "tmin": args.tmin,
        "tmax": args.tmax,
        "balance_classes": args.balance_classes,
        "samples": len(dataset),
        "subjects": int(len(set(dataset.subject_ids.tolist()))),
        "channel_names": dataset.channel_names,
        "comparison_reference": args.compare_to_summary,
        "train_crop_seconds": args.train_crop_seconds,
        "eval_crops": args.eval_crops,
        "time_mask_prob": args.time_mask_prob,
        "channel_dropout_prob": args.channel_dropout_prob,
        "skip_interpretability": args.skip_interpretability,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "dataset_summary.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if args.skip_interpretability:
        logger.info("Skipping signal/channel interpretation for screening run")
    else:
        signal_top3 = signal_channel_importance(dataset, args.output_dir)
        logger.info("Signal top-3 channels per class: %s", signal_top3)
    if args.analyze_only:
        return metadata

    if args.architecture == "hierarchical":
        return run_hierarchical(args, dataset, metadata, device)
    if args.architecture in {"majority", "csp_lda", "csp_svm"}:
        return run_classical_baseline(args, dataset, metadata)

    if args.split_mode == "subject":
        splitter = GroupKFold(n_splits=args.k_folds)
        split_iter = splitter.split(np.arange(len(dataset)), dataset.labels, groups=dataset.subject_ids)
    else:
        splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(np.arange(len(dataset)), dataset.labels)

    fold_results = []
    args._crop_samples = crop_sample_count(args, dataset)
    if args._crop_samples:
        logger.info(
            "Using cropped training: %.2fs -> %d samples, eval_crops=%d",
            args.train_crop_seconds,
            args._crop_samples,
            args.eval_crops,
        )
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if fold < args.start_fold:
            logger.info("Skipping fold %d/%d (--start_fold=%d)", fold + 1, args.k_folds, args.start_fold)
            continue
        logger.info("Fold %d/%d train=%d val=%d", fold + 1, args.k_folds, len(train_idx), len(val_idx))
        if args.architecture == "fbpower":
            sample_shape = tuple(dataset[0][0].shape)
            if len(sample_shape) != 3:
                raise ValueError("architecture=fbpower requires feature_mode=fb_logvar")
            model = FBPowerNet(
                n_bands=sample_shape[0],
                n_channels=sample_shape[1],
                n_windows=sample_shape[2],
                n_classes=len(dataset.class_names),
                dropout_p=args.dropout,
                hidden=args.fb_hidden,
            ).to(device)
        else:
            model = build_neural_model(
                args,
                len(dataset.class_names),
                device,
                sample_shape=tuple(dataset[0][0].shape),
            )
        criterion = make_class_weighted_criterion(
            dataset,
            train_idx,
            len(dataset.class_names),
            device,
            args.class_weights,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_sampler = make_balanced_sampler(dataset, train_idx, len(dataset.class_names)) if args.balanced_sampler else None
        train_loader = DataLoader(
            WindowedSubset(
                dataset,
                train_idx,
                crop_samples=args._crop_samples,
                random_crop=bool(args._crop_samples),
                time_mask_prob=args.time_mask_prob,
                channel_dropout_prob=args.channel_dropout_prob,
            ),
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        best_score = float("inf") if args.monitor_metric == "loss" else -1.0
        best = None
        patience = 0
        history = []
        fold_start = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics, _, _ = evaluate_with_crops(model, val_loader, criterion, device, args)
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_metrics["loss"], 5),
                "val_accuracy": round(val_metrics["accuracy"], 5),
                "val_f1": round(val_metrics["f1"], 5),
            }
            history.append(row)
            current_score = val_metrics["loss"] if args.monitor_metric == "loss" else val_metrics[args.monitor_metric]
            improved = current_score < best_score if args.monitor_metric == "loss" else current_score > best_score
            if improved:
                best_score = current_score
                best = metrics_for_json(val_metrics, include_loss=False)
                best["epoch"] = epoch
                best["monitor_metric"] = args.monitor_metric
                best["monitor_score"] = float(current_score)
                patience = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "fold": fold,
                        "epoch": epoch,
                        "metrics": best,
                        "metadata": metadata,
                    },
                    os.path.join(args.output_dir, f"fold_{fold}_best.pth"),
                )
            else:
                patience += 1
            if epoch % args.log_every == 0 or epoch == 1:
                logger.info(
                    "fold=%d epoch=%d train_loss=%.4f val_acc=%.4f val_f1=%.4f best_%s=%.4f",
                    fold,
                    epoch,
                    train_loss,
                    val_metrics["accuracy"],
                    val_metrics["f1"],
                    args.monitor_metric,
                    best_score,
                )
            if patience >= args.early_stop_patience:
                logger.info("Early stop fold=%d epoch=%d", fold, epoch)
                break

        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        plot_learning_curve(history, os.path.join(fold_dir, "learning_curve.png"))
        best_checkpoint = torch.load(
            os.path.join(args.output_dir, f"fold_{fold}_best.pth"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])
        if args.skip_interpretability:
            attention_top3 = None
            saliency_top3 = None
        else:
            attention_top3 = attention_channel_importance(model, dataset, device, fold_dir)
            saliency_top3 = saliency_channel_importance(
                model,
                dataset,
                device,
                fold_dir,
                indices=val_idx,
                max_per_class=args.saliency_samples,
            )
        fold_result = {
            "fold": fold,
            "train_subjects": sorted(set(np.asarray(dataset.subject_ids)[train_idx].astype(int).tolist())),
            "val_subjects": sorted(set(np.asarray(dataset.subject_ids)[val_idx].astype(int).tolist())),
            "best_metrics": best,
            "history": history,
            "time_s": round(time.time() - fold_start, 1),
            "attention_top3": attention_top3,
            "saliency_top3": saliency_top3,
        }
        with open(os.path.join(args.output_dir, f"fold_{fold}_metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=2)
        fold_results.append(fold_result)

    summary = build_summary(metadata, fold_results, args.compare_to_summary)
    with open(os.path.join(args.output_dir, "all_folds_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "RESULT acc=%.4f +/- %.4f bal_acc=%.4f +/- %.4f f1=%.4f +/- %.4f",
        summary["accuracy"]["mean"],
        summary["accuracy"]["std"],
        summary["balanced_accuracy"]["mean"],
        summary["balanced_accuracy"]["std"],
        summary["f1"]["mean"],
        summary["f1"]["std"],
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=[
            "none",
            "final75",
            "final75_balanced",
            "final75_f1",
            "final75_hier",
            "paper_subject_final75",
            "paper_subject_final75_crop",
            "paper_csp_lda",
            "paper_csp_svm",
            "paper_eegnet",
            "paper_shallowconvnet",
            "fbpower75",
            "fbpower75_balanced",
        ],
        default="none",
        help=(
            "Use final75 for the high-accuracy flowchart setup: "
            "rest/left/right/feet + bandpass_zscore + stream ECA + large Avola classifier. "
            "Use fbpower75 for the filter-bank log-variance spectral-spatial model."
        ),
    )
    parser.add_argument("--data_dir", default="./eeg_data")
    parser.add_argument("--output_dir", default="./stage_results")
    parser.add_argument("--class_mode", choices=sorted(CLASS_CONFIGS), default="binary_lr")
    parser.add_argument("--preprocess", choices=["none", "zscore", "bandpass", "bandpass_zscore"], default="none")
    parser.add_argument("--feature_mode", choices=["raw", "fb_logvar"], default="raw")
    parser.add_argument("--channel_set", choices=["all", "motor"], default="all")
    parser.add_argument("--fb_windows", type=int, default=4)
    parser.add_argument("--tmin", type=float, default=-0.5)
    parser.add_argument("--tmax", type=float, default=4.1)
    parser.add_argument("--balance_classes", action="store_true")
    parser.add_argument(
        "--architecture",
        choices=[
            "majority",
            "multistream",
            "fbpower",
            "hierarchical",
            "csp_lda",
            "csp_svm",
            "eegnet",
            "shallowconvnet",
        ],
        default="multistream",
    )
    parser.add_argument("--class_weights", choices=["none", "balanced"], default="none")
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--monitor_metric", choices=["accuracy", "f1", "loss"], default="accuracy")
    parser.add_argument(
        "--attention",
        choices=[
            "none",
            "input_se",
            "input_stats_se",
            "stream_eca",
            "input_se_stream_eca",
            "input_stats_stream_eca",
        ],
        default="none",
    )
    parser.add_argument("--split_mode", choices=["trial", "subject"], default="trial")
    parser.add_argument(
        "--temporal_model",
        choices=["none", "tcn", "tcn_attn", "tcn_pool8", "tcn_multipool"],
        default="none",
    )
    parser.add_argument("--tcn_layers", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument(
        "--start_fold",
        type=int,
        default=0,
        help="Skip folds before this zero-based fold index. Use 9 to run only fold 10 of a 10-fold experiment.",
    )
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--classifier_hidden",
        type=int,
        default=1024,
        help="Classifier hidden units. Use 0 for Avola-style midpoint width.",
    )
    parser.add_argument("--fb_hidden", type=int, default=256)
    parser.add_argument("--csp_components", type=int, default=12)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_gamma", default="scale")
    parser.add_argument("--compare_to_summary", default=None)
    parser.add_argument("--train_crop_seconds", type=float, default=0.0)
    parser.add_argument("--eval_crops", type=int, default=1)
    parser.add_argument("--time_mask_prob", type=float, default=0.0)
    parser.add_argument("--channel_dropout_prob", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--saliency_samples", type=int, default=64)
    parser.add_argument(
        "--skip_interpretability",
        action="store_true",
        help="Skip signal, attention, and saliency plots during fast model screening.",
    )
    parser.add_argument("--sync_check", action="store_true")
    parser.add_argument("--sync_subjects", type=int, default=2)
    parser.add_argument("--sync_runs", type=int, default=2)
    parser.add_argument("--sync_events", type=int, default=12)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    provided_flags = {
        token[2:].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }

    def preset_default(name, value):
        if name not in provided_flags:
            setattr(args, name, value)

    if args.preset in {
        "final75",
        "final75_balanced",
        "final75_f1",
        "final75_hier",
        "paper_subject_final75",
        "paper_subject_final75_crop",
    }:
        preset_default("class_mode", "mi4_rest")
        preset_default("preprocess", "bandpass_zscore")
        preset_default("feature_mode", "raw")
        preset_default("channel_set", "all")
        preset_default("architecture", "hierarchical" if args.preset == "final75_hier" else "multistream")
        preset_default("class_weights", "balanced" if args.preset in {"final75_f1", "final75_hier"} else "none")
        preset_default("monitor_metric", "f1" if args.preset in {"final75_f1", "final75_hier"} else "accuracy")
        preset_default("tmin", 0.0)
        preset_default("tmax", 4.0)
        preset_default("balance_classes", args.preset == "final75_balanced")
        preset_default("attention", "stream_eca")
        preset_default("temporal_model", "none")
        preset_default("tcn_layers", 0)
        preset_default("tcn_dropout", 0.0)
        preset_default("epochs", 80 if args.preset == "final75_hier" else 100)
        preset_default("batch_size", 64)
        preset_default("k_folds", 10)
        preset_default("split_mode", "subject" if args.preset in {"paper_subject_final75", "paper_subject_final75_crop"} else "trial")
        preset_default("early_stop_patience", 10)
        preset_default("lr", 1e-4)
        preset_default("weight_decay", 1e-4)
        preset_default("dropout", 0.5)
        preset_default("classifier_hidden", 0)
        if args.preset == "paper_subject_final75_crop":
            preset_default("train_crop_seconds", 3.0)
            preset_default("eval_crops", 5)
            preset_default("time_mask_prob", 0.10)
            preset_default("channel_dropout_prob", 0.05)
            preset_default("early_stop_patience", 12)
        if "num_workers" not in provided_flags:
            args.num_workers = max(args.num_workers, 4)
    elif args.preset in {"paper_csp_lda", "paper_csp_svm", "paper_eegnet", "paper_shallowconvnet"}:
        preset_default("class_mode", "mi4_rest")
        preset_default("preprocess", "bandpass_zscore")
        preset_default("feature_mode", "raw")
        preset_default("channel_set", "all")
        if args.preset == "paper_csp_lda":
            preset_default("architecture", "csp_lda")
        elif args.preset == "paper_csp_svm":
            preset_default("architecture", "csp_svm")
        elif args.preset == "paper_eegnet":
            preset_default("architecture", "eegnet")
        else:
            preset_default("architecture", "shallowconvnet")
        preset_default("class_weights", "balanced" if args.preset in {"paper_eegnet", "paper_shallowconvnet"} else "none")
        preset_default("monitor_metric", "f1")
        preset_default("tmin", 0.0)
        preset_default("tmax", 4.0)
        preset_default("balance_classes", False)
        preset_default("attention", "none")
        preset_default("temporal_model", "none")
        preset_default("epochs", 80)
        preset_default("batch_size", 64)
        preset_default("k_folds", 10)
        preset_default("split_mode", "subject")
        preset_default("early_stop_patience", 10)
        preset_default("lr", 1e-3 if args.preset == "paper_eegnet" else 1e-4)
        preset_default("weight_decay", 1e-4)
        preset_default("dropout", 0.5)
        preset_default("csp_components", 12)
        if "num_workers" not in provided_flags:
            args.num_workers = max(args.num_workers, 4)
    elif args.preset in {"fbpower75", "fbpower75_balanced"}:
        preset_default("class_mode", "mi4_rest")
        preset_default("preprocess", "none")
        preset_default("feature_mode", "fb_logvar")
        preset_default("channel_set", "motor")
        preset_default("fb_windows", 4)
        preset_default("architecture", "fbpower")
        preset_default("class_weights", "balanced")
        preset_default("monitor_metric", "f1")
        if "balanced_sampler" not in provided_flags:
            args.balanced_sampler = True
        preset_default("tmin", 0.5)
        preset_default("tmax", 4.0)
        preset_default("balance_classes", args.preset == "fbpower75_balanced")
        preset_default("attention", "none")
        preset_default("temporal_model", "none")
        preset_default("tcn_layers", 0)
        preset_default("tcn_dropout", 0.0)
        preset_default("epochs", 120)
        preset_default("batch_size", 256)
        preset_default("k_folds", 10)
        preset_default("early_stop_patience", 15)
        preset_default("lr", 3e-4)
        preset_default("weight_decay", 1e-3)
        preset_default("dropout", 0.35)
        preset_default("fb_hidden", 256)
        if "num_workers" not in provided_flags:
            args.num_workers = max(args.num_workers, 4)
    return args


if __name__ == "__main__":
    run(parse_args())
