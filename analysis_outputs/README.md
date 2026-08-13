# Checked analysis-output provenance

These files were generated from the completed Kaggle export used for the revised manuscript.

- Archive filename: `reviewer_safe_13_results_export-3.zip`
- Archive SHA-256: `a58a4370e8ce39765d6145829a164cf0114dc981bf3faa9e9da31419d2129318`
- Completed experiment summaries: 13
- Completed outer-fold records: 130
- Outer folds per experiment: 10
- Unique retained subjects: 103
- Outer-test coverage: every retained subject appears exactly once per experiment
- Subject integrity: inner-training, inner-validation, and outer-test subject sets are disjoint in every fold
- Reported refit mode: `none`; the inner-selected checkpoint is retained

The checked archive is committed here as `reviewer_safe_13_results_export.zip`. The committed CSV and JSON values were recomputed from it and compared at full stored precision. The figure SHA-256 values are:

- `supplementary_figure_s2_capacity_cost.png`: `a28da30741b8358943841071f7602b791272e08f79f29b5a4e802f0e38a95beb`
- `class_recall_diagnostic.png`: `7f47780aa9adec20c18459dfc736f45de12e1d0e96ff25d83470bc1f3b3f69df`
- `figure_s1_training_diagnostics.png`: `7ac41ff4a5f93cb29db480ade05af1ae8e30fa16e2e9920ba22432716ba22899`
- `figure4_reference_83m_aggregate_confusion.png`: `4c12b2cf897d3c71a1712bba8384d66a6b9b34c2da8d62ac1844727fd339d5f9`
- `figure5_reference_83m_per_fold_confusions.png`: `2fc60ff8a70305b9ae546c8e021c2acfc1b732345c369a0749ee8ccc47506c2d`

Regenerate the outputs with:

```bash
python src/analyze_results.py --archive analysis_outputs/reviewer_safe_13_results_export.zip --output-dir reproduced_analysis
python src/analyze_training_diagnostics.py --archive analysis_outputs/reviewer_safe_13_results_export.zip --output-dir reproduced_analysis
python src/generate_confusion_figures.py --archive analysis_outputs/reviewer_safe_13_results_export.zip --output-dir reproduced_analysis
python src/eeg_journal_analysis.py --data_dir /path/to/eegmmidb/files --output_dir reproduced_physiology --reject_uv 300
```

The `physiology/` directory contains the checked tabular outputs and manuscript
Figures 6--8 from the independent neurophysiological analysis. The analysis
retained 76 artefact-screened subjects at the documented 300 microvolt
peak-to-peak threshold. It is independent of the classifier and does not provide
features to the MST-CNN.
