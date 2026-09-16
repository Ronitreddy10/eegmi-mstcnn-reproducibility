# Completed reviewer outputs

This directory contains machine-readable exports from the completed reviewer experiment grid.

`analysis/` contains fold assignments, flat outer-test predictions, per-fold and aggregate confusion matrices, subject-level metrics and distributions, bootstrap confidence intervals, paired effect-size tables, and performance summaries.

`environment/` contains the saved Kaggle software and hardware records for the GPU window runs, optional kernel runs, and final CPU analysis run.

Class labels are 0 for rest, 1 for left-fist imagery, 2 for right-fist imagery, and 3 for both-feet imagery.

The source fold JSON files remain in the downloadable Kaggle checkpoint. The flat CSV exports in this directory are intended for independent inspection without parsing nested JSON.
