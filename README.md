# EEG MST-CNN reviewer experiment package

This is the ready-to-upload Kaggle package for the manuscript reviewer experiments. It preserves the manuscript protocol and adds selectable temporal kernels, selectable cue-relative epoch windows, held-out prediction saving, subject-level statistics and plots, 10,000-resample subject bootstrap confidence intervals, expanded paired Wilcoxon tables with effect sizes, a matched CSP+LDA baseline, a physiology inclusion/rejection audit, receptive-field reporting, and an explicit FIR-filter record.

No EEG files are included. Download the [PhysioNet EEG Motor Movement/Imagery Database version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/) and upload the EDF tree as a Kaggle dataset. The loader accepts a Kaggle dataset root with nested folders and indexes files named `S###R##.edf`. The required runs are 4, 6, 8, 10, 12, and 14 for every eligible subject; the program stops with a missing-subject report instead of silently running an incomplete manuscript cohort.

For a complete independent-reproduction guide, including command-line download instructions, subject lists, fold assignments, saved predictions, confusion matrices, package versions, and GPU records, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The physiology cohort and trial-count supplement is in [`supplementary/`](supplementary/README.md).

## Protocol that must remain fixed

- 103 eligible subjects: 1–109 except 43, 88, 89, 92, 100, and 104.
- Four classes: rest, left-fist imagery, right-fist imagery, and both-feet imagery.
- Outer evaluation: 10-fold `GroupKFold` by subject.
- Inner validation: grouped 15% split from the outer-training subjects.
- Selection: maximum inner-validation macro-F1.
- The selected checkpoint is retained; there is no refit.
- Maximum 60 epochs, early-stopping patience 8, batch size 64.
- AdamW, learning rate `1e-4`, weight decay `1e-4`, dropout `0.5`.
- Classifier preprocessing: 4–40 Hz zero-phase FIR before epoching, then per-epoch per-channel z-scoring.
- Seed 42 for reviewer ablations.

The kernel and temporal-window studies use the practical 19.94M four-stream configuration as their base. Single- and fewer-stream models naturally have fewer parameters; the generated table reports parameters, FLOPs, and latency beside performance so scale count is not presented as a parameter-matched test.

## Kaggle steps

1. Upload `eeg_mstcnn_reviewer_kaggle.zip` as a private Kaggle dataset, or upload it directly to a notebook session.
2. Create a Kaggle notebook, attach the package and an EEGMMIDB dataset containing all required EDF files, and enable a GPU accelerator.
3. In the first cell, locate and extract the package:

```python
from pathlib import Path
import zipfile

package_zip = next(Path("/kaggle/input").rglob("eeg_mstcnn_reviewer_kaggle.zip"))
package_dir = Path("/kaggle/working/eeg_mstcnn_reviewer_kaggle")
with zipfile.ZipFile(package_zip) as archive:
    archive.extractall(package_dir)
print(package_dir)
```

4. Install the pinned-compatible dependencies (Kaggle usually already has most of them):

```python
!pip install -q -r {package_dir}/requirements.txt
```

5. Set `DATA_DIR` to the attached EEGMMIDB dataset root. It may contain nested subject directories:

```python
DATA_DIR = "/kaggle/input/YOUR-EEGMMIDB-DATASET"
```

6. Run one stage. Start with the kernel experiment because it directly tests the paper's multi-scale claim:

```python
!python {package_dir}/kaggle_entrypoint.py \
  --data-dir "{DATA_DIR}" \
  --stage kernels \
  --max-wall-hours 11
```

7. Download `/kaggle/working/eeg_mstcnn_reviewer_results_checkpoint.zip` after every session. For a new session, attach that ZIP and resume with the same command plus:

```text
--resume-archive /kaggle/input/YOUR-CHECKPOINT-DATASET/eeg_mstcnn_reviewer_results_checkpoint.zip
```

Completed folds and experiments are skipped. A fold is never stopped mid-write.

8. Run the remaining stages in this order:

```python
!python {package_dir}/kaggle_entrypoint.py --data-dir "{DATA_DIR}" --stage windows --max-wall-hours 11 --resume-archive "/kaggle/input/YOUR-CHECKPOINT-DATASET/eeg_mstcnn_reviewer_results_checkpoint.zip"

!python {package_dir}/kaggle_entrypoint.py --data-dir "{DATA_DIR}" --stage baselines --max-wall-hours 11 --resume-archive "/kaggle/input/YOUR-CHECKPOINT-DATASET/eeg_mstcnn_reviewer_results_checkpoint.zip"

!python {package_dir}/kaggle_entrypoint.py --data-dir "{DATA_DIR}" --stage physiology --max-wall-hours 11 --resume-archive "/kaggle/input/YOUR-CHECKPOINT-DATASET/eeg_mstcnn_reviewer_results_checkpoint.zip"
```

The `baselines` stage runs EEGNet, ShallowConvNet, DeepConvNet1D, ResNet1D, and CSP+LDA. The classical baseline uses the same 103 subjects, four classes, 4–40 Hz preprocessing, 0–4 s epochs, and outer subject folds; it never uses trial-randomized validation.

After the neural reviewer experiments are complete, `--stage remaining-cpu`
runs only the outstanding CSP+LDA baseline and physiology audit, then refreshes
the final statistical tables. It does not train or rerun a neural network and
does not require a GPU.

If the Kaggle runtime is long enough, `--stage all` runs the core stages in one invocation. Three additional scale controls (`9+11`, `11+13`, and `7+13`) are available through `--stage optional-kernels` after the core reviewer grid.

## Required experiment grid

The core grid is in `configs/reviewer_experiments.json`:

- Kernels: `7`, `9`, `11`, `13`, `7+9`, `7+9+11`, `7+9+11+13`.
- Windows: `0–2 s`, `0–3 s`, `0–4 s` (the full-kernel reference above), and `1–4 s`.
- Neural baselines: four existing manuscript baselines under their existing effective-number CE setting.
- Classical baseline: CSP with 8 components and LDA, fitted inside each outer training fold.

## Generated results

The checkpoint contains:

```text
eeg_mstcnn_reviewer_results/
├── RUN_STATUS.json
├── run_environment.json
├── experiments/
│   └── <experiment>/
│       ├── fold_01.json ... fold_10.json
│       └── summary.json
├── analysis/
│   ├── experiment_performance_summary.csv
│   ├── all_subject_metrics.csv
│   ├── subject_distribution_summary.csv
│   ├── subject_bootstrap_95ci.csv
│   ├── expanded_wilcoxon_effect_sizes.csv
│   ├── fold_assignments.csv
│   ├── fold_level_predictions.csv
│   ├── fold_confusion_matrices.csv
│   ├── aggregate_confusion_matrices.csv
│   ├── subject_distribution_<experiment>.png
│   ├── receptive_field.csv
│   └── receptive_field.json
└── physiology/
    ├── physiology_subject_audit.csv
    ├── signal_analysis_summary.json
    └── figures and physiology statistics
```

Each neural and CSP fold file retains untouched outer-test `sample_indices`, `subject_ids`, `y_true`, and `y_pred`. Confidence intervals use 10,000 percentile bootstrap resamples of whole subject prediction blocks with seed 42. The paired model-comparison table aligns the same held-out subjects and reports mean and median differences, a bootstrap 95% CI for the mean difference, paired rank-biserial correlation, raw Wilcoxon p, and Holm-adjusted p.

The physiology audit reports all 103 classification-eligible subjects, trial rejection counts by class under the 300 µV peak-to-peak rule, why any subject lacks a complete four-class physiology record, which class triggered exclusion, and whether retained trial counts are equal across classes.

Completed reviewer outputs are committed under [`results/reviewer_2026/`](results/reviewer_2026/README.md). These exports let readers inspect the exact fold assignments, predictions, confusion matrices, subject-level metrics, bootstrap intervals, and paired statistical tables without rerunning training.

## Local structural checks

These checks use synthetic tensors and do not create manuscript performance results:

```bash
python tests/smoke_test.py
python src/eeg_journal_analysis.py --synthetic_smoke --output_dir /tmp/physiology_smoke
```

To verify parsing without training, use:

```bash
python kaggle_entrypoint.py --help
python src/run_robustness_study.py --help
python src/run_csp_lda.py --help
```

## Interpretation safeguards

- Treat subject-disjoint out-of-fold predictions as the primary evaluation unit.
- Fold-level summaries are descriptive because outer training sets overlap.
- Report effect sizes and confidence intervals beside Wilcoxon p-values.
- Do not call the four-stream ablation parameter matched unless a separate capacity-matched design is run.
- Receptive-field results describe the convolutional pathway before adaptive pooling. They do not establish a physiological mechanism by themselves.
- Batch-1 latency is computational inference latency only; it does not establish end-to-end online BCI performance.
