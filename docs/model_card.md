# Model card: NHANES diabetes screening classifier

## Model overview

A calibrated logistic-regression classifier that estimates whether an adult
NHANES participant reported a clinician diagnosis of diabetes (`DIQ010 = 1`).
It is an educational portfolio model, not a diagnostic device or a prospective
risk model.

## Intended use

- Demonstrate reproducible clinical-data ingestion, validation and modelling.
- Demonstrate leakage-safe preprocessing, calibration and threshold selection.
- Support technical review of code, assumptions, evaluation and limitations.
- **Not intended for individual clinical decisions or deployment.**

## Data

- Source: NHANES 2017-March 2020 pre-pandemic public-use release.
- Analysis cohort: 9421 adults with a usable target.
- Training set: 7536 participants.
- Held-out test set: 1885 participants.
- Target: self-reported diagnosed diabetes, not laboratory-confirmed diabetes.
- Default predictors: age, BMI, mean systolic and diastolic blood pressure,
  poverty-income ratio, sex, race/ethnicity and education.
- Fasting glucose is excluded by default because it is target-adjacent and only
  available in a subsample.

## Training and threshold selection

- Median imputation and standardisation for numeric variables.
- Most-frequent imputation and one-hot encoding for categorical variables.
- Class-balanced logistic regression.
- Five-fold stratified CV tunes regularisation strength using ROC AUC.
- Sigmoid calibration is fitted within cross-validation.
- The operating threshold is chosen from out-of-fold training probabilities to
  meet the prespecified sensitivity target while maximising specificity.
- The test set remains untouched until final evaluation.

## Held-out performance

| Metric                    |   Estimate | 95% CI      |
|:--------------------------|-----------:|:------------|
| ROC AUC                   |      0.796 | 0.773-0.819 |
| Average precision         |      0.363 | 0.317-0.422 |
| Brier score               |      0.111 | 0.102-0.120 |
| Accuracy                  |      0.684 | 0.663-0.706 |
| Precision / PPV           |      0.299 | 0.268-0.331 |
| Sensitivity / recall      |      0.811 | 0.766-0.855 |
| Specificity               |      0.662 | 0.638-0.684 |
| F1 score                  |      0.437 | 0.399-0.474 |
| Negative predictive value |      0.951 | 0.939-0.964 |

Operating threshold: **0.1357**.

Percentile confidence intervals are based on 1,000 bootstrap resamples of the
held-out test set. They quantify sampling variation in this test split only.

## Subgroup evaluation

These are descriptive checks, not proof of fairness. Small group counts and the
absence of uncertainty intervals mean differences should be interpreted cautiously.

### Sex

| group   |   selection_rate |   tpr |   fpr |   support |   positive_cases |   negative_cases |   roc_auc |
|:--------|-----------------:|------:|------:|----------:|-----------------:|-----------------:|----------:|
| Female  |            0.384 | 0.791 | 0.32  |       981 |              134 |              847 |     0.782 |
| Male    |            0.437 | 0.828 | 0.359 |       904 |              151 |              753 |     0.808 |

### Age group

| group   |   selection_rate |   tpr |   fpr |   support |   positive_cases |   negative_cases |   roc_auc |
|:--------|-----------------:|------:|------:|----------:|-----------------:|-----------------:|----------:|
| 18-39   |            0.015 | 0.182 | 0.012 |       669 |               11 |              658 |     0.733 |
| 40-59   |            0.311 | 0.556 | 0.266 |       578 |               90 |              488 |     0.698 |
| 60+     |            0.912 | 0.973 | 0.888 |       638 |              184 |              454 |     0.636 |

### Race and ethnicity

| group              |   selection_rate |   tpr |   fpr |   support |   positive_cases |   negative_cases |   roc_auc |
|:-------------------|-----------------:|------:|------:|----------:|-----------------:|-----------------:|----------:|
| Mexican American   |            0.39  | 0.771 | 0.325 |       241 |               35 |              206 |     0.803 |
| Non-Hispanic Asian |            0.333 | 0.75  | 0.271 |       216 |               28 |              188 |     0.78  |
| Non-Hispanic Black |            0.46  | 0.831 | 0.398 |       493 |               71 |              422 |     0.787 |
| Non-Hispanic White |            0.389 | 0.778 | 0.322 |       668 |               99 |              569 |     0.791 |
| Other Hispanic     |            0.436 | 0.848 | 0.345 |       181 |               33 |              148 |     0.812 |
| Other/Multiracial  |            0.465 | 1     | 0.313 |        86 |               19 |               67 |     0.871 |

## Limitations

- Cross-sectional classification cannot estimate future diabetes risk.
- The self-reported target is vulnerable to recall error and undiagnosed disease.
- A single random hold-out split is used; there is no external validation.
- NHANES survey weights are used for descriptive prevalence estimates, but the
  predictive model is an unweighted individual-level portfolio analysis.
- The statsmodels baseline uses complete cases and heteroskedasticity-robust
  standard errors; it does not implement the full NHANES complex survey design.
- Race/ethnicity and sex are survey variables with limited categories and should
  not be interpreted as biological causes.
- Threshold utility depends on the costs of false positives and false negatives.

## Reproducibility

Run the scripts in the order shown in the repository README. Generated metrics
and figures are versioned for transparent review. The seed, split, features,
hyperparameters and selected operating threshold are recorded in
`artifacts/metrics/training_summary.json`.

## Figures

- `reports/figures/roc_test.png`
- `reports/figures/pr_test.png`
- `reports/figures/calibration_test.png`
- `reports/figures/confusion_matrix_test.png`
