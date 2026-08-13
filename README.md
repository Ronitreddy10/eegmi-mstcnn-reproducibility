# Nested subject-wise MST-CNN evaluation on PhysioNet EEGMMIDB

Reproducibility package for four-class, inter-subject EEG motor-imagery decoding with a multi-scale temporal convolutional neural network (MST-CNN).

This repository implements the corrected experimental protocol used in the revised manuscript: nested subject-wise evaluation, controlled model-capacity ablation, neural baseline comparisons, computational profiling, checkpoint-selection diagnostics, multiple random seeds for the 83.08 M reference model, and effective-number class-balanced training.

The implemented cue-aligned epoch spans 0--4 s at 160 Hz. Because MNE includes both temporal endpoints, each epoch contains **641 samples** (`0.000, 0.00625, ..., 4.000 s`).

## What this repository establishes

- The outer evaluation is one **10-fold subject-wise grouped cross-validation** procedure. It is not a separate subject-specific CV plus another 10-fold CV.
- Within every outer-training partition, a grouped inner-validation subset selects the checkpoint by macro-F1.
- The untouched outer-test subjects are evaluated once and are never used for early stopping or checkpoint selection.
- Four MST-CNN capacities are compared at seed 42: 0.55 M, 4.96 M, 19.94 M, and 83.08 M parameters.
- EEGNet, ShallowConvNet, DeepConvNet1D, and ResNet1D are evaluated on the same outer folds.
- Parameters, MACs, FLOPs (`2 × MACs`), and batch-1 NVIDIA Tesla T4 inference latency are reported.
- The 83.08 M reference model is replicated with seeds 42, 123, and 2026 using both unweighted and effective-number cross-entropy.
- Accuracy, balanced accuracy, macro-F1, and per-class recall are retained for every outer fold.
- Full inner-validation histories support best-checkpoint versus final-epoch diagnostics.

## Verified experiment scope

The completed analysis contains 13 experiment definitions and 130 outer-fold result files.

| Analysis | Models / conditions | Seeds |
|---|---|---|
| Capacity ablation | MST-CNN 0.55 M, 4.96 M, 19.94 M, 83.08 M | 42 |
| Neural baselines | EEGNet, ShallowConvNet, DeepConvNet1D, ResNet1D | 42 |
| Seed robustness | MST-CNN 83.08 M, unweighted CE | 42, 123, 2026 |
| Class-imbalance analysis | MST-CNN 83.08 M, effective-number CE | 42, 123, 2026 |

The 19.94 M result is a seed-42 capacity finding. It is not presented as a three-seed or class-balanced result.

### Exact capacity controls

The capacity ablation uses compound width/pooling variants of the same four-stream MST-CNN design. It is **not** a classifier-only ablation.

| Variant | Stream maximum channels | Stream output channels | Adaptive-pool length | Classifier hidden width | Exact parameters |
|---|---:|---:|---:|---:|---:|
| 0.55 M | 48 | 24 | 12 | 128 | 551,144 |
| 4.96 M | 128 | 48 | 32 | 450 | 4,956,614 |
| 19.94 M | 192 | 64 | 48 | 1,250 | 19,936,294 |
| 83.08 M | 256 | 64 | 48 | 6,146 | 83,080,458 |

Kernel sizes, the four-stream topology, two ConvBlocks per stream, adaptive pooling, feature fusion, dropout, and the four-class output remain common. The controlled channel widths, pooled feature dimension, and classifier width determine the capacity points.

## Key checked results

At seed 42, the 19.94 M MST-CNN reached `70.23 ± 1.72%` accuracy, `58.27 ± 2.18%` balanced accuracy, and `59.68 ± 2.02%` macro-F1 over the 10 outer folds. Its performance was statistically comparable with the 83.08 M configuration while using 76.0% fewer parameters, 40.4% fewer FLOPs, and 30.1% lower measured batch-1 latency.

Across the three 83.08 M seeds, effective-number weighting improved balanced accuracy by 0.98 percentage points and increased recall for left-fist, right-fist, and both-feet imagery, while raw accuracy decreased by 2.09 points. See [`analysis_outputs/`](analysis_outputs/) for the checked aggregate tables, paired statistics, per-class recall, computational-cost rows, and checkpoint diagnostics.

## Repository layout

```text
.
├── README.md
├── METHOD_PROTOCOL.md
├── CITATION.cff
├── requirements.txt
├── kaggle_reviewer_safe_13.ipynb
├── configs/
│   ├── reviewer_safe_13.json
│   ├── full_multiseed.json
│   └── stage1_single_seed.json
├── src/
│   ├── eeg_stage_ablation.py
│   ├── run_robustness_study.py
│   ├── run_complete_batch.py
│   ├── run_grid.py
│   ├── summarize_grid.py
│   ├── analyze_results.py
│   ├── analyze_training_diagnostics.py
│   ├── generate_confusion_figures.py
│   └── eeg_journal_analysis.py
├── tests/
│   └── smoke_test.py
└── analysis_outputs/
    ├── reviewer_safe_13_results_export.zip
    ├── supplementary_figure_s2_capacity_cost.png
    ├── class_recall_diagnostic.png
    ├── figure_s1_training_diagnostics.png
    ├── figure4_reference_83m_aggregate_confusion.png
    ├── figure5_reference_83m_per_fold_confusions.png
    └── physiology/
        ├── signal_analysis_summary.json
        ├── roi_omnibus_statistics.csv
        ├── roi_pairwise_statistics.csv
        ├── channel_contrast_statistics.csv
        ├── bandpower_top_electrodes.csv
        ├── class_waveforms_motor_channels.png
        ├── motor_roi_bandpower.png
        ├── bandpower_class_topographies.png
        └── bandpower_contrast_topographies.png
```

## Dataset

The code uses the [PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)](https://physionet.org/content/eegmmidb/). The EEG recordings are not redistributed in this repository. Users must obtain the dataset from PhysioNet and comply with the terms stated on the dataset page.

The corrected analysis retains 103 subjects after applying the manuscript's epoch-integrity criteria and uses rest, left-fist imagery, right-fist imagery, and both-feet imagery.

## Kaggle reproduction

The easiest reproduction route is [`kaggle_reviewer_safe_13.ipynb`](kaggle_reviewer_safe_13.ipynb).

1. Create a Kaggle notebook and import the supplied notebook file.
2. Attach the public PhysioNet EEG Motor Movement/Imagery dataset containing the EDF files.
3. Enable an NVIDIA GPU.
4. Keep Internet enabled so the notebook can clone this public repository, or attach the repository files as a Kaggle dataset.
5. Run all cells.
6. Download `reviewer_safe_13_results_export.zip` from `/kaggle/working`.
7. If a session reaches its wall-time boundary, attach the exported ZIP to the next session and run the notebook again. Completed folds are detected and skipped.

The reviewer grid is defined in [`configs/reviewer_safe_13.json`](configs/reviewer_safe_13.json). Do not change the outer folds, epochs, patience, batch size, or refit mode when reproducing the reported manuscript run.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the structural smoke test:

```bash
python tests/smoke_test.py
```

Run a configured experiment grid:

```bash
python src/run_grid.py \
  --grid configs/reviewer_safe_13.json \
  --data-dir /path/to/eegmmidb \
  --output-root robustness_results
```

Aggregate completed jobs:

```bash
python src/summarize_grid.py --results robustness_results
```

Regenerate the checked manuscript statistics and figures directly from the completed export:

```bash
python src/analyze_results.py \
  --archive analysis_outputs/reviewer_safe_13_results_export.zip \
  --output-dir reproduced_analysis

python src/analyze_training_diagnostics.py \
  --archive analysis_outputs/reviewer_safe_13_results_export.zip \
  --output-dir reproduced_analysis

python src/generate_confusion_figures.py \
  --archive analysis_outputs/reviewer_safe_13_results_export.zip \
  --output-dir reproduced_analysis
```

Regenerate the independent neurophysiological analysis used for manuscript Figures 6--8 and Tables 9--10:

```bash
python src/eeg_journal_analysis.py \
  --data_dir /path/to/eegmmidb \
  --output_dir reproduced_physiology \
  --reject_uv 300
```

This branch uses average re-referencing, 4--40 Hz filtering, a 300 µV peak-to-peak rejection criterion, Welch mu/beta band-power estimation, subject-level aggregation, Friedman tests, paired Wilcoxon tests with Holm correction, and electrode-level standardized effects. It is independent of classifier training.

## Corrected checkpoint protocol

For every outer fold:

1. Reserve the outer-test subjects.
2. Split the remaining subjects into grouped inner-training and inner-validation sets.
3. Select the epoch using only inner-validation macro-F1.
4. Preserve that checkpoint without refitting.
5. Evaluate the untouched outer-test subjects once.

Every fold stores subject IDs, random seed, loss condition, class weights, selected epoch, full inner-validation history, best and final inner-validation metrics, outer-test metrics, computational costs, and per-class recall.

## Result files

- `experiment_summary.csv`: fold-aggregated metrics and computational costs by experiment and seed.
- `multiseed_summary.csv`: three-seed summaries for the 83.08 M loss conditions.
- `per_class_recall.csv`: class-wise recall used for the imbalance analysis.
- `performance_rows.csv`: checked manuscript table rows.
- `manuscript_results.json`: consolidated numerical results and paired tests.
- `training_diagnostics_summary.json`: best-checkpoint versus final-epoch diagnostics.
- `supplementary_figure_s2_capacity_cost.png`: Supplementary Figure S2, the capacity-performance and computational-cost comparison.
- `class_recall_diagnostic.png`: supporting three-seed per-class recall diagnostic; it is not manuscript Figure 5.
- `figure_s1_training_diagnostics.png`: checkpoint-selection curves and epoch distributions.
- `figure4_reference_83m_aggregate_confusion.png`: manuscript Figure 4.
- `figure5_reference_83m_per_fold_confusions.png`: manuscript Figure 5.
- `reviewer_safe_13_results_export.zip`: the checked 13-experiment, 130-fold Kaggle result archive used to regenerate the statistics and figures.

[`analysis_outputs/README.md`](analysis_outputs/README.md) records the completed archive checksum, fold counts, validation checks, and the provenance of the committed outputs.

## Methodological record

[`METHOD_PROTOCOL.md`](METHOD_PROTOCOL.md) gives the concise protocol that should be cited when describing the corrected evaluation. It also records the distinction between the seed-42 capacity study and the three-seed 83.08 M robustness study.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The manuscript citation and DOI should replace the repository-only citation after publication.

## Data and licensing note

The PhysioNet dataset is governed by its own terms and is not included here. No separate reuse licence has yet been assigned to the source code; public visibility of this repository does not change the authors' copyright.
