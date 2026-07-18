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

- `figure4_capacity_cost.png`: `357c23f8b53e8e1e1fb51aa7b09e466e289d0d1961abaa05da375af596f4fef8`
- `figure5_class_recall.png`: `7f47780aa9adec20c18459dfc736f45de12e1d0e96ff25d83470bc1f3b3f69df`
- `figure_s1_training_diagnostics.png`: `7ac41ff4a5f93cb29db480ade05af1ae8e30fa16e2e9920ba22432716ba22899`

Regenerate the outputs with:

```bash
python src/analyze_results.py --archive analysis_outputs/reviewer_safe_13_results_export.zip --output-dir reproduced_analysis
python src/analyze_training_diagnostics.py --archive analysis_outputs/reviewer_safe_13_results_export.zip --output-dir reproduced_analysis
```
