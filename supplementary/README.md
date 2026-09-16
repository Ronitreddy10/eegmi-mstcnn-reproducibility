# Physiology cohort supplement

`Supplementary_Physiology_Cohort_and_Trial_Counts.xlsx` is the reader-facing supplementary workbook. It contains:

- the 76 archived manuscript physiology subjects;
- class-specific annotated, rejected, and retained trial counts for every included subject;
- total and average retained trials for each class;
- the 27 classification-eligible subjects outside the physiology cohort;
- recorded exclusion reasons and the class that triggered exclusion where supported by the audit;
- retained-trial equality checks;
- the complete FIR implementation record.

The CSV files provide the same information in machine-readable form. `physiology_subject_audit_all_103.csv` is the unabridged transparent audit across all 103 classification-eligible subjects.

The archived 76-subject cohort totals are 5,097 rest trials, 1,404 left-fist imagery trials, 1,385 right-fist imagery trials, and 1,401 both-feet imagery trials. Their per-subject averages are 67.07, 18.47, 18.22, and 18.43 trials, respectively.

Retained class counts are not equal for any of the 76 subjects when rest is included. This is expected because the dataset contributes more rest annotations than imagery annotations. Five subjects have equal retained counts across the three imagery classes.

Eighteen of the 27 excluded subjects lose all retained trials from at least one class under the 300 microvolt rule. Nine others are absent from the archived 76-subject list even though the current audit retains at least one trial in every class; no additional historical exclusion criterion is present in the saved code or output metadata.
