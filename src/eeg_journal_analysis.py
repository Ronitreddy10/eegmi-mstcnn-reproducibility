#!/usr/bin/env python3
"""
Reproduce the manuscript's independent neurophysiological analysis.

This script is intentionally separate from the classifier training pipeline.
It keeps the selected classifier results frozen and produces the physiological
evidence reported in manuscript Figures 6--8 and Tables 9--10:

1. Example signal before and after 4-40 Hz preprocessing.
2. Subject-level C3/Cz/C4 class-average waveforms with 95% confidence bands.
3. Mu (8-13 Hz) and beta (13-30 Hz) scalp topographies for all four classes.
4. Paired class-contrast topographies.
5. Subject-level Friedman and Wilcoxon/Holm statistical tests.
The physiological bandpower branch never z-scores individual epochs because
epoch-wise z-scoring removes absolute amplitude/power differences.
"""

import argparse
import csv
import json
import logging
import math
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from eeg_stage_ablation import (
    CLASS_CONFIGS,
    PhysioNetStageDataset,
    index_attached_edfs,
    map_physionet_event,
    resolve_edf_paths,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eeg_journal_analysis")

CLASS_NAMES = CLASS_CONFIGS["mi4_rest"]["class_names"]
CLASS_LABELS = {
    "rest": "Rest",
    "left_fist_imagery": "Left fist imagery",
    "right_fist_imagery": "Right fist imagery",
    "both_feet_imagery": "Both feet imagery",
}
CLASS_COLORS = {
    "rest": "#5c677d",
    "left_fist_imagery": "#0077b6",
    "right_fist_imagery": "#d1495b",
    "both_feet_imagery": "#2a9d8f",
}
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
ROIS = {
    "left_motor": ["FC3", "C3", "CP3"],
    "midline_motor": ["FCz", "Cz", "CPz"],
    "right_motor": ["FC4", "C4", "CP4"],
}
WAVE_CHANNELS = ["C3", "Cz", "C4"]
CONTRASTS = {
    "left_minus_right": (["left_fist_imagery"], ["right_fist_imagery"]),
    "hands_minus_feet": (
        ["left_fist_imagery", "right_fist_imagery"],
        ["both_feet_imagery"],
    ),
    "motor_imagery_minus_rest": (
        ["left_fist_imagery", "right_fist_imagery", "both_feet_imagery"],
        ["rest"],
    ),
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and not fieldnames:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if not len(p_values):
        return p_values
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    n = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def rank_biserial_paired(x, y):
    diff = np.asarray(x) - np.asarray(y)
    diff = diff[np.abs(diff) > 1e-12]
    if not len(diff):
        return 0.0
    ranks = rankdata(np.abs(diff))
    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()
    return float((positive - negative) / max(positive + negative, 1e-12))


def bootstrap_mean_ci(values, rng, n_boot=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        value = float(np.mean(values)) if len(values) else float("nan")
        return value, value
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def safe_wilcoxon(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.allclose(x, y):
        return 0.0, 1.0
    result = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def standardize_raw(raw):
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def bandpower_db(epoch_data, sfreq):
    n_fft = min(512, epoch_data.shape[-1])
    psd, freqs = mne.time_frequency.psd_array_welch(
        epoch_data,
        sfreq=sfreq,
        fmin=4.0,
        fmax=40.0,
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        average="mean",
        verbose=False,
    )
    output = {}
    for name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        power = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
        output[name] = 10.0 * np.log10(power + 1e-20)
    return output


def load_subject_summary(subject_id, args, keep_sample=False):
    runs = CLASS_CONFIGS["mi4_rest"]["runs"]
    paths = resolve_edf_paths(subject_id, runs, args.data_dir, args.attached_edf_index)
    by_class = {name: {"waveforms": [], "mu": [], "beta": []} for name in CLASS_NAMES}
    sample = None
    info = None
    channel_names = None
    sfreq = None

    for run, path in zip(runs, paths):
        raw = mne.io.read_raw_edf(path, preload=True, stim_channel="auto", verbose=False)
        standardize_raw(raw)
        raw_before = raw.copy() if keep_sample and sample is None else None
        raw.filter(l_freq=4.0, h_freq=40.0, fir_design="firwin", verbose=False)
        sfreq = float(raw.info["sfreq"])
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        inverse = {int(code): str(name) for name, code in event_id.items()}
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
        reject = {"eeg": args.reject_uv * 1e-6} if args.reject_uv > 0 else None
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=args.tmin,
            tmax=args.tmax,
            baseline=None,
            proj=False,
            picks=picks,
            reject=reject,
            preload=True,
            verbose=False,
        )
        data = epochs.get_data(copy=True)
        labels = epochs.events[:, 2]
        if channel_names is None:
            channel_names = list(epochs.ch_names)
            info = epochs.info.copy()

        if raw_before is not None:
            channel = args.sample_channel if args.sample_channel in raw.ch_names else "C3"
            start = int(max(0.0, args.sample_start) * sfreq)
            stop = int(min(raw.times[-1], args.sample_start + args.sample_seconds) * sfreq)
            sample = {
                "channel": channel,
                "sfreq": sfreq,
                "before": raw_before.get_data(picks=[channel], start=start, stop=stop)[0],
                "after": raw.get_data(picks=[channel], start=start, stop=stop)[0],
            }

        powers = bandpower_db(data, sfreq)
        wave_indices = [channel_names.index(ch) for ch in WAVE_CHANNELS]
        for epoch_index, code in enumerate(labels):
            event_name = inverse.get(int(code))
            mapped = map_physionet_event("mi4_rest", run, event_name)
            if mapped is None:
                continue
            class_name = CLASS_NAMES[mapped]
            by_class[class_name]["waveforms"].append(data[epoch_index, wave_indices])
            by_class[class_name]["mu"].append(powers["mu"][epoch_index])
            by_class[class_name]["beta"].append(powers["beta"][epoch_index])

    subject_result = {}
    for class_name, values in by_class.items():
        if not values["mu"]:
            return None, sample, info, channel_names, sfreq
        subject_result[class_name] = {
            "waveforms": np.mean(values["waveforms"], axis=0),
            "mu": np.mean(values["mu"], axis=0),
            "beta": np.mean(values["beta"], axis=0),
            "n_trials": len(values["mu"]),
        }
    return subject_result, sample, info, channel_names, sfreq


def make_synthetic_data(args):
    rng = np.random.default_rng(args.seed)
    n_subjects = args.max_subjects or 12
    sfreq = 160.0
    n_times = int(round((args.tmax - args.tmin) * sfreq)) + 1
    montage = mne.channels.make_standard_montage("standard_1005")
    channel_names = [
        ch for ch in montage.ch_names
        if ch in {
            "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
            "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
            "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
            "FT7", "FT8", "T7", "T8", "P3", "Pz", "P4", "O1", "Oz", "O2",
        }
    ]
    info = mne.create_info(channel_names, sfreq, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    class_array = []
    times = np.arange(n_times) / sfreq
    for subject in range(n_subjects):
        subject_result = {}
        for class_name in CLASS_NAMES:
            wave = rng.normal(0, 0.15e-6, size=(len(WAVE_CHANNELS), n_times))
            mu = rng.normal(-100, 1.0, size=len(channel_names))
            beta = rng.normal(-105, 1.0, size=len(channel_names))
            if class_name == "left_fist_imagery":
                mu[channel_names.index("C4")] += 3.0
                wave[2] += 0.5e-6 * np.sin(2 * np.pi * 10 * times)
            elif class_name == "right_fist_imagery":
                mu[channel_names.index("C3")] += 3.0
                wave[0] += 0.5e-6 * np.sin(2 * np.pi * 10 * times)
            elif class_name == "both_feet_imagery":
                beta[channel_names.index("Cz")] += 3.0
                wave[1] += 0.5e-6 * np.sin(2 * np.pi * 20 * times)
            subject_result[class_name] = {
                "waveforms": wave,
                "mu": mu,
                "beta": beta,
                "n_trials": 20,
            }
        class_array.append((subject + 1, subject_result))
    sample = {
        "channel": "C3",
        "sfreq": sfreq,
        "before": 1e-5 * (np.sin(2 * np.pi * 10 * times) + 0.7 * np.sin(2 * np.pi * 60 * times)),
        "after": 1e-5 * np.sin(2 * np.pi * 10 * times),
    }
    return class_array, sample, info, channel_names, sfreq


def collect_signal_data(args):
    if args.synthetic_smoke:
        return make_synthetic_data(args)
    excluded = PhysioNetStageDataset.EXCLUDED_SUBJECTS
    subjects = [s for s in range(1, 110) if s not in excluded]
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    results = []
    sample = info = channel_names = sfreq = None
    for position, subject_id in enumerate(subjects, start=1):
        try:
            result, possible_sample, possible_info, possible_channels, possible_sfreq = (
                load_subject_summary(subject_id, args, keep_sample=(sample is None))
            )
            if result is not None:
                results.append((subject_id, result))
            if sample is None and possible_sample is not None:
                sample = possible_sample
            if info is None:
                info = possible_info
            if channel_names is None:
                channel_names = possible_channels
            if sfreq is None:
                sfreq = possible_sfreq
        except Exception as exc:
            logger.warning("Skipping S%03d during journal analysis: %s", subject_id, exc)
        if position % 10 == 0 or position == len(subjects):
            logger.info("Journal analysis loaded %d/%d subjects; complete=%d", position, len(subjects), len(results))
    if not results:
        raise RuntimeError("No complete subjects available for journal analysis.")
    return results, sample, info, channel_names, sfreq


def stack_signal_data(results, channel_names):
    subject_ids = np.asarray([subject_id for subject_id, _ in results], dtype=int)
    n_subjects = len(results)
    n_classes = len(CLASS_NAMES)
    n_channels = len(channel_names)
    n_times = results[0][1][CLASS_NAMES[0]]["waveforms"].shape[-1]
    waveforms = np.zeros((n_subjects, n_classes, len(WAVE_CHANNELS), n_times))
    powers = {
        band: np.zeros((n_subjects, n_classes, n_channels))
        for band in BANDS
    }
    trial_counts = np.zeros((n_subjects, n_classes), dtype=int)
    for si, (_, subject_result) in enumerate(results):
        for ci, class_name in enumerate(CLASS_NAMES):
            waveforms[si, ci] = subject_result[class_name]["waveforms"]
            trial_counts[si, ci] = subject_result[class_name]["n_trials"]
            for band in BANDS:
                powers[band][si, ci] = subject_result[class_name][band]
    return subject_ids, waveforms, powers, trial_counts


def plot_sample_signal(sample, output_dir):
    if sample is None:
        return None
    n = min(len(sample["before"]), len(sample["after"]))
    times = np.arange(n) / sample["sfreq"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    axes[0].plot(times, sample["before"][:n] * 1e6, color="#6c757d", lw=0.9)
    axes[0].set_title(f"Before bandpass filtering (average-referenced): {sample['channel']}")
    axes[1].plot(times, sample["after"][:n] * 1e6, color="#0077b6", lw=0.9)
    axes[1].set_title("After average reference and 4-40 Hz bandpass filtering")
    for axis in axes:
        axis.set_ylabel("Amplitude (µV)")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    path = output_dir / "sample_signal_before_after.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_waveforms(waveforms, sfreq, output_dir):
    times = np.arange(waveforms.shape[-1]) / sfreq
    fig, axes = plt.subplots(len(WAVE_CHANNELS), 1, figsize=(11, 9), sharex=True)
    for channel_index, (axis, channel) in enumerate(zip(axes, WAVE_CHANNELS)):
        for class_index, class_name in enumerate(CLASS_NAMES):
            values = waveforms[:, class_index, channel_index] * 1e6
            mean = values.mean(axis=0)
            ci = 1.96 * values.std(axis=0, ddof=1) / math.sqrt(len(values))
            axis.plot(times, mean, color=CLASS_COLORS[class_name], lw=1.3, label=CLASS_LABELS[class_name])
            axis.fill_between(times, mean - ci, mean + ci, color=CLASS_COLORS[class_name], alpha=0.14)
        axis.set_title(channel)
        axis.set_ylabel("Amplitude (µV)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time after cue onset (s)")
    fig.suptitle("Subject-level class-average motor-region EEG (mean ± 95% CI)", y=1.01)
    fig.tight_layout()
    path = output_dir / "class_waveforms_motor_channels.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def topomap(axis, values, info, title, vlim=None, cmap="RdBu_r"):
    image, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=axis,
        show=False,
        contours=4,
        cmap=cmap,
        vlim=vlim,
        sensors=True,
    )
    axis.set_title(title, fontsize=9)
    return image


def plot_class_topographies(powers, info, output_dir):
    fig, axes = plt.subplots(
        len(BANDS),
        len(CLASS_NAMES),
        figsize=(13.5, 6.8),
        constrained_layout=True,
    )
    for band_index, band in enumerate(BANDS):
        means = powers[band].mean(axis=0)
        vmin, vmax = float(means.min()), float(means.max())
        row_image = None
        for class_index, class_name in enumerate(CLASS_NAMES):
            row_image = topomap(
                axes[band_index, class_index],
                means[class_index],
                info,
                f"{band.capitalize()}: {CLASS_LABELS[class_name]}",
                vlim=(vmin, vmax),
                cmap="viridis",
            )
        colorbar = fig.colorbar(
            row_image,
            ax=axes[band_index, :],
            shrink=0.75,
            pad=0.02,
        )
        colorbar.set_label(f"{band.capitalize()} log bandpower (dB)", fontsize=9)
    fig.suptitle("Subject-averaged log bandpower topographies", fontsize=13)
    path = output_dir / "bandpower_class_topographies.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def contrast_values(power_array, positive_names, negative_names):
    positive = power_array[:, [CLASS_NAMES.index(name) for name in positive_names]].mean(axis=1)
    negative = power_array[:, [CLASS_NAMES.index(name) for name in negative_names]].mean(axis=1)
    return positive - negative


def plot_contrast_topographies(powers, info, output_dir):
    fig, axes = plt.subplots(
        len(BANDS),
        len(CONTRASTS),
        figsize=(11.5, 6.8),
        constrained_layout=True,
    )
    for band_index, band in enumerate(BANDS):
        maps = []
        for positive, negative in CONTRASTS.values():
            maps.append(contrast_values(powers[band], positive, negative).mean(axis=0))
        limit = max(float(np.abs(np.stack(maps)).max()), 1e-6)
        row_image = None
        for contrast_index, (contrast_name, values) in enumerate(zip(CONTRASTS, maps)):
            row_image = topomap(
                axes[band_index, contrast_index],
                values,
                info,
                f"{band.capitalize()}: {contrast_name.replace('_', ' ')}",
                vlim=(-limit, limit),
            )
        colorbar = fig.colorbar(
            row_image,
            ax=axes[band_index, :],
            shrink=0.75,
            pad=0.02,
        )
        colorbar.set_label(f"{band.capitalize()} difference (dB)", fontsize=9)
    fig.suptitle("Paired class-contrast log-bandpower topographies", fontsize=13)
    path = output_dir / "bandpower_contrast_topographies.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def compute_roi_statistics(powers, channel_names, output_dir, seed):
    rng = np.random.default_rng(seed)
    omnibus_rows = []
    pairwise_rows = []
    roi_values = {}
    for band, power_array in powers.items():
        roi_values[band] = {}
        for roi_name, channels in ROIS.items():
            indices = [channel_names.index(ch) for ch in channels if ch in channel_names]
            values = power_array[:, :, indices].mean(axis=2)
            roi_values[band][roi_name] = values
            statistic, p_value = friedmanchisquare(*[values[:, ci] for ci in range(len(CLASS_NAMES))])
            omnibus_rows.append({
                "band": band,
                "roi": roi_name,
                "n_subjects": len(values),
                "friedman_chi_square": float(statistic),
                "p_value": float(p_value),
            })
            local_rows = []
            for first in range(len(CLASS_NAMES)):
                for second in range(first + 1, len(CLASS_NAMES)):
                    x, y = values[:, first], values[:, second]
                    statistic_w, p_w = safe_wilcoxon(x, y)
                    ci_low, ci_high = bootstrap_mean_ci(x - y, rng)
                    local_rows.append({
                        "band": band,
                        "roi": roi_name,
                        "class_a": CLASS_NAMES[first],
                        "class_b": CLASS_NAMES[second],
                        "n_subjects": len(x),
                        "mean_difference_db": float(np.mean(x - y)),
                        "ci95_low_db": ci_low,
                        "ci95_high_db": ci_high,
                        "wilcoxon_statistic": statistic_w,
                        "p_value": p_w,
                        "rank_biserial": rank_biserial_paired(x, y),
                    })
            adjusted = holm_adjust([row["p_value"] for row in local_rows])
            for row, adjusted_p in zip(local_rows, adjusted):
                row["holm_p_value"] = float(adjusted_p)
                row["significant_0_05"] = bool(adjusted_p < 0.05)
                pairwise_rows.append(row)
    write_csv(output_dir / "roi_omnibus_statistics.csv", omnibus_rows)
    write_csv(output_dir / "roi_pairwise_statistics.csv", pairwise_rows)
    return roi_values, omnibus_rows, pairwise_rows


def plot_roi_values(roi_values, output_dir):
    fig, axes = plt.subplots(1, len(BANDS), figsize=(13, 5), sharey=False)
    x = np.arange(len(ROIS))
    width = 0.18
    for axis, band in zip(axes, BANDS):
        for class_index, class_name in enumerate(CLASS_NAMES):
            means, errors = [], []
            for roi_name in ROIS:
                values = roi_values[band][roi_name][:, class_index]
                means.append(values.mean())
                errors.append(1.96 * values.std(ddof=1) / math.sqrt(len(values)))
            axis.bar(
                x + (class_index - 1.5) * width,
                means,
                width,
                yerr=errors,
                capsize=2,
                color=CLASS_COLORS[class_name],
                label=CLASS_LABELS[class_name],
            )
        axis.set_title(f"{band.capitalize()} band")
        axis.set_xticks(x, [name.replace("_", "\n") for name in ROIS])
        axis.set_ylabel("Log bandpower (dB)")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Motor-ROI bandpower by condition (mean ± 95% CI)", y=1.02)
    fig.tight_layout()
    path = output_dir / "motor_roi_bandpower.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def compute_channel_statistics(powers, channel_names, output_dir):
    rows = []
    for band, power_array in powers.items():
        for contrast_name, (positive, negative) in CONTRASTS.items():
            differences = contrast_values(power_array, positive, negative)
            local = []
            for channel_index, channel in enumerate(channel_names):
                values = differences[:, channel_index]
                _, p_value = safe_wilcoxon(values, np.zeros_like(values))
                local.append({
                    "band": band,
                    "contrast": contrast_name,
                    "channel": channel,
                    "mean_difference_db": float(values.mean()),
                    "p_value": p_value,
                    "rank_biserial": rank_biserial_paired(values, np.zeros_like(values)),
                })
            adjusted = holm_adjust([row["p_value"] for row in local])
            for row, adjusted_p in zip(local, adjusted):
                row["holm_p_value"] = float(adjusted_p)
                row["significant_0_05"] = bool(adjusted_p < 0.05)
                rows.append(row)
    write_csv(output_dir / "channel_contrast_statistics.csv", rows)
    return rows


def compute_top_electrodes(powers, channel_names, output_dir):
    rows = []
    for band, power_array in powers.items():
        for class_index, class_name in enumerate(CLASS_NAMES):
            current = power_array[:, class_index]
            others = np.delete(power_array, class_index, axis=1).mean(axis=1)
            difference = current - others
            effect = difference.mean(axis=0) / (difference.std(axis=0, ddof=1) + 1e-9)
            order = np.argsort(np.abs(effect))[::-1][:10]
            for rank, channel_index in enumerate(order, start=1):
                rows.append({
                    "band": band,
                    "class": class_name,
                    "rank": rank,
                    "channel": channel_names[channel_index],
                    "class_vs_others_standardized_effect": float(effect[channel_index]),
                    "mean_difference_db": float(difference[:, channel_index].mean()),
                })
    write_csv(output_dir / "bandpower_top_electrodes.csv", rows)
    return rows


def plot_analysis_diagram(output_dir):
    fig, axis = plt.subplots(figsize=(12.5, 6.2))
    axis.axis("off")
    blocks = [
        ("PhysioNet EEGBCI\n103 subjects | 64 electrodes", "#dbeafe"),
        ("Event synchronization\n0-4 s | 641 samples", "#dcfce7"),
        ("Physiology preprocessing\naverage reference | 4-40 Hz | 300 µV rejection", "#fef3c7"),
        ("Subject-level EEG evidence\nmu/beta power | C3/Cz/C4 waveforms", "#fce7f3"),
        ("Class-difference statistics\nFriedman | Wilcoxon + Holm", "#ede9fe"),
        ("Reported outputs\nFigures 6-8 | Tables 9-10", "#e0f2fe"),
    ]
    centers = [
        (0.18, 0.72),
        (0.50, 0.72),
        (0.82, 0.72),
        (0.18, 0.28),
        (0.50, 0.28),
        (0.82, 0.28),
    ]
    for (text, color), (x_center, y_center) in zip(blocks, centers):
        axis.text(
            x_center,
            y_center,
            text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.75",
                "facecolor": color,
                "edgecolor": "#334155",
                "linewidth": 1.2,
            },
        )
    arrows = [
        ((0.31, 0.72), (0.37, 0.72)),
        ((0.63, 0.72), (0.69, 0.72)),
        ((0.82, 0.61), (0.18, 0.39)),
        ((0.31, 0.28), (0.37, 0.28)),
        ((0.63, 0.28), (0.69, 0.28)),
    ]
    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=axis.transAxes,
            arrowprops={
                "arrowstyle": "->",
                "color": "#334155",
                "lw": 1.7,
                "connectionstyle": "arc3,rad=0" if start[1] == end[1] else "arc3,rad=-0.18",
            },
        )
    axis.set_title(
        "Independent EEG class-difference analysis branch",
        fontsize=14,
        pad=18,
    )
    path = output_dir / "journal_analysis_block_diagram.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def run_signal_analysis(args, output_dir):
    results, sample, info, channel_names, sfreq = collect_signal_data(args)
    subject_ids, waveforms, powers, trial_counts = stack_signal_data(results, channel_names)
    np.savez_compressed(
        output_dir / "subject_level_eeg_features.npz",
        subject_ids=subject_ids,
        waveforms=waveforms,
        mu=powers["mu"],
        beta=powers["beta"],
        trial_counts=trial_counts,
        class_names=np.asarray(CLASS_NAMES),
        channel_names=np.asarray(channel_names),
    )
    artifacts = [
        plot_sample_signal(sample, output_dir),
        plot_waveforms(waveforms, sfreq, output_dir),
        plot_class_topographies(powers, info, output_dir),
        plot_contrast_topographies(powers, info, output_dir),
        plot_analysis_diagram(output_dir),
    ]
    roi_values, omnibus, pairwise = compute_roi_statistics(
        powers, channel_names, output_dir, args.seed
    )
    artifacts.append(plot_roi_values(roi_values, output_dir))
    channel_stats = compute_channel_statistics(powers, channel_names, output_dir)
    top_electrodes = compute_top_electrodes(powers, channel_names, output_dir)
    summary = {
        "n_complete_subjects": len(subject_ids),
        "subject_ids": subject_ids.tolist(),
        "class_names": CLASS_NAMES,
        "channel_names": channel_names,
        "sfreq": sfreq,
        "epoch_seconds": [args.tmin, args.tmax],
        "epoch_samples": int(waveforms.shape[-1]),
        "artifact_rejection_peak_to_peak_uv": args.reject_uv,
        "preprocessing": "average reference plus 4-40 Hz bandpass; no epoch-wise z-score",
        "bands_hz": BANDS,
        "trial_counts_total": {
            class_name: int(trial_counts[:, index].sum())
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "significant_roi_pairwise_tests": int(sum(row["significant_0_05"] for row in pairwise)),
        "significant_channel_contrasts": int(sum(row["significant_0_05"] for row in channel_stats)),
        "top_electrodes_rows": len(top_electrodes),
        "artifacts": [str(path.name) for path in artifacts if path],
    }
    write_json(output_dir / "signal_analysis_summary.json", summary)
    logger.info(
        "Signal analysis complete: subjects=%d significant_roi_pairs=%d significant_channel_contrasts=%d",
        summary["n_complete_subjects"],
        summary["significant_roi_pairwise_tests"],
        summary["significant_channel_contrasts"],
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the manuscript neurophysiological analysis")
    parser.add_argument("--data_dir", default="./eeg_data")
    parser.add_argument("--output_dir", default="./journal_eeg_analysis")
    parser.add_argument("--synthetic_smoke", action="store_true")
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--reject_uv", type=float, default=300.0)
    parser.add_argument("--sample_channel", default="C3")
    parser.add_argument("--sample_start", type=float, default=0.0)
    parser.add_argument("--sample_seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.attached_edf_index = index_attached_edfs()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    manifest = {
        "purpose": "Independent EEG class-difference and band-power analysis",
        "classes": CLASS_NAMES,
        "signal_analysis": run_signal_analysis(args, output_dir),
    }
    write_json(output_dir / "journal_analysis_manifest.json", manifest)
    logger.info("Journal analysis artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
