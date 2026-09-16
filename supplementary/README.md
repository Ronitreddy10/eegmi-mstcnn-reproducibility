# Physiology cohort supplement

`Supplementary_Physiology_Cohort_and_Trial_Counts.xlsx` is the reader-facing supplementary workbook. It contains:

- the 85 subjects in the updated physiology analysis;
- class-specific annotated, rejected, and retained trial counts for every included subject;
- total and average retained trials for each class;
- the 18 classification-eligible subjects excluded from the physiology cohort;
- recorded exclusion reasons and the class that triggered exclusion where supported by the audit;
- retained-trial equality checks;
- the complete FIR implementation record.

The CSV files provide the same information in machine-readable form. `physiology_subject_audit_all_103.csv` is the unabridged transparent audit across all 103 classification-eligible subjects.

The updated 85-subject cohort totals are 5,225 rest trials, 1,448 left-fist imagery trials, 1,436 right-fist imagery trials, and 1,460 both-feet imagery trials. Their per-subject averages are 61.47, 17.04, 16.89, and 17.18 trials, respectively.

Retained class counts are not equal for any of the 85 subjects when rest is included. This is expected because the dataset contributes more rest annotations than imagery annotations. Five subjects have equal retained counts across the three imagery classes.

All 18 excluded subjects lose every retained trial from at least one class under the 300 microvolt rule. The exclusion file records the triggering class and retained counts for each subject.
