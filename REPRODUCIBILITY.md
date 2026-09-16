# Independent reproduction instructions

## Dataset download

Use PhysioNet EEG Motor Movement Imagery Database version 1.0.0. The official record is:

https://physionet.org/content/eegmmidb/1.0.0/

Download the complete database from the command line with:

```bash
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

The resulting directory should contain subject folders such as `S001` and EDF files such as `S001R04.edf`. Runs 4, 6, 8, 10, 12, and 14 are required for each eligible subject. Point `--data-dir` to the directory above the subject folders. On Kaggle, upload the downloaded EDF tree as a private dataset and use its `/kaggle/input/...` path.

The loader verifies the required files before training. It does not silently substitute subjects or omit missing runs.

## Subject cohorts

Classification uses 103 subjects: subjects 1 through 109 excluding 43, 88, 89, 92, 100, and 104.

The updated physiology analysis contains 85 subjects. These are the classification-eligible subjects retaining at least one trial in every class after the documented 300 microvolt peak-to-peak rejection rule. The exact list, per-class annotation counts, rejected trials, retained trials, and totals are provided in:

- `supplementary/physiology_included_subject_trial_counts.csv`
- `supplementary/Supplementary_Physiology_Cohort_and_Trial_Counts.xlsx`

The 18 classification-eligible subjects excluded from the updated physiology cohort and their class-specific reasons are in `supplementary/physiology_excluded_subjects.csv`. Every excluded subject loses all usable trials from at least one class under the 300 microvolt peak-to-peak rule.

## Preprocessing

`src/eeg_stage_ablation.py` contains the classifier preprocessing and model definitions. `src/eeg_journal_analysis.py` contains the physiology pipeline and rejection audit.

Both pipelines standardize PhysioNet channel names and filter continuous runs before epoch extraction. The classifier uses the EDF channel order (or the explicitly selected motor-channel subset), retains the recording reference, and then applies per-channel epoch z-scoring. The physiology pipeline additionally attaches the MNE `standard_1005` montage and applies an average EEG reference before filtering; it does not apply epoch-wise z-scoring. These differences reflect the executed code and are stated explicitly so that the two analyses are not accidentally conflated.

The runtime FIR record is:

- band-pass edges: 4 and 40 Hz;
- design: `firwin` FIR;
- window: Hamming;
- phase: zero;
- implementation: `mne.io.Raw.filter`;
- filtering stage: continuous data before epoch extraction;
- padding: `reflect_limited`;
- sampling frequency: 160 Hz;
- realized filter length: 265 samples;
- realized filter order: 264;
- automatic transition bandwidths: 2 Hz lower and 10 Hz upper;
- realized minus 6 dB cutoff frequencies: 3 and 45 Hz.

The machine-readable record is `supplementary/fir_filter_details.csv`.

## Evaluation and training

The implemented protocol is described in `METHOD_PROTOCOL.md`. The primary settings are:

- 10-fold outer `GroupKFold` by subject;
- grouped 15 percent inner validation split from outer-training subjects;
- checkpoint selected by inner-validation macro-F1;
- untouched outer-test subjects evaluated once;
- no refit after checkpoint selection;
- maximum 60 epochs and patience 8;
- batch size 64;
- AdamW, learning rate `1e-4`, weight decay `1e-4`;
- dropout 0.5;
- seed 42 for reviewer ablations;
- seeds 42, 123, and 2026 for the manuscript multi-seed reference runs.

Exact experiment definitions are in `configs/`. The Kaggle entry point is `kaggle_entrypoint.py`.

## Fold assignments predictions and confusion matrices

Completed outputs are stored in `results/reviewer_2026/analysis/`:

- `fold_assignments.csv` records inner-training, inner-validation, and outer-test subjects for every outer fold;
- `fold_level_predictions.csv` records the experiment, outer fold, dataset sample index, subject, true label, and predicted label for every untouched outer-test prediction;
- `fold_confusion_matrices.csv` records every 4 by 4 outer-test fold confusion matrix in long form;
- `aggregate_confusion_matrices.csv` records the pooled confusion matrix for each experiment;
- `all_subject_metrics.csv` records subject-level accuracy, balanced accuracy, and macro-F1;
- `subject_bootstrap_95ci.csv` contains 10,000-resample subject-block bootstrap confidence intervals;
- `expanded_wilcoxon_effect_sizes.csv` contains paired subject-level differences, confidence intervals, rank-biserial effect sizes, raw p-values, and Holm-adjusted p-values;
- `fold_level_paired_comparisons.csv` contains the secondary fold-level summary.

Regenerate these tables with:

```bash
python src/analyze_reviewer_results.py \
  --results-root /path/to/eeg_mstcnn_reviewer_results/experiments \
  --output-dir /path/to/output/analysis \
  --bootstrap-resamples 10000 \
  --seed 42
```

## Software and hardware

Exact runtime records are in `results/reviewer_2026/environment/`.

The neural experiments ran in Kaggle with Python 3.12.13, MNE 1.12.1, NumPy 2.0.2, scikit-learn 1.6.1, SciPy 1.16.3, Matplotlib 3.10.0, and PyTorch 2.10.0 with CUDA 12.8. The Kaggle environment exposed two Tesla T4 GPUs. The training process selected the default CUDA device and did not use `DataParallel`, so each process used one GPU.

The final CSP and LDA, physiology audit, and table refresh ran on CPU with the same package versions and the CPU build of PyTorch 2.10.0.

## Verification tests

Run the structural and analysis tests with:

```bash
python tests/smoke_test.py
python tests/analysis_smoke_test.py
python tests/csp_smoke_test.py
```

These tests use synthetic data and confirm code structure and output generation. They do not create manuscript performance results.
