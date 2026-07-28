# Methodology

## Study design

This project is a cross-sectional classification study using the NHANES
2017-March 2020 pre-pandemic public-use release. The analysis unit is one adult
participant. The work is a reproducible portfolio demonstration and not a
clinical validation study.

## Target

`DIQ010` asks whether a doctor or health professional has told the participant
that they have diabetes. Responses are mapped as:

- yes (`1`) -> positive;
- no (`2`) -> negative;
- borderline, refused, unknown and missing -> excluded.

The resulting target is self-reported diagnosed diabetes. It is not equivalent
to laboratory-confirmed disease and does not identify undiagnosed diabetes.

## Cohort

The demographics, diabetes questionnaire, body-measures and blood-pressure
files are linked by `SEQN`. The fasting-glucose file is optional. The pipeline
checks required variables and one-row-per-person keys before merging.

Participants are retained when they:

1. are aged 18 years or older; and
2. have a target response that maps to yes or no.

This produced 9,421 adults, including 1,423 target-positive participants.

## Data quality

NHANES questionnaire non-response codes are converted to missing values.
Repeated blood-pressure measurements are averaged across available readings.
Values outside deliberately broad plausibility ranges are set to missing:

- BMI: 10-100 kg/m²;
- glucose: 20-700 mg/dL;
- mean systolic blood pressure: 60-260 mmHg;
- mean diastolic blood pressure: 30-180 mmHg.

The ranges are data-quality rules, not diagnostic thresholds.

## Descriptive statistics

The examination weight `WTMECPRP` is used for population-oriented descriptive
means and prevalence estimates. Unweighted values are reported alongside them.
The implementation does not calculate design-based uncertainty using strata
and primary sampling units.

## Statistical baseline

The complete-case baseline is:

```text
target_diabetes ~ age + sex + BMI + mean systolic blood pressure
                  + race/ethnicity + education
```

The model is fitted with statsmodels logistic regression and HC3 robust
standard errors. It is an interpretable association analysis and does not
account for the full NHANES complex survey design.

## Predictive features

The default predictive model includes continuous age, BMI, mean systolic blood
pressure, mean diastolic blood pressure and poverty-income ratio, plus
categorical sex, race/ethnicity and education.

Derived age and BMI categories are excluded from the predictor set to avoid
duplicating the same information in continuous and categorical form. They are
retained for EDA and subgroup reporting.

Fasting glucose is excluded by default because it is strongly target-adjacent
and measured in a subsample. An opt-in sensitivity analysis is supported.

## Split and preprocessing

A stratified random 80/20 split with `random_state=42` creates training and
held-out test sets. All learned preprocessing is contained in a scikit-learn
pipeline:

- numeric median imputation;
- numeric standardisation;
- categorical most-frequent imputation;
- one-hot encoding with ignored unknown levels.

Therefore no imputation, scaling or category information is learned from the
test set.

## Model selection

The estimator is class-balanced logistic regression. Regularisation strength
`C` is selected from `[0.1, 1.0, 3.0, 10.0]` by five-fold stratified
cross-validation, with ROC AUC as the selection metric.

## Calibration and threshold

Sigmoid calibration is fitted within stratified cross-validation. To avoid
choosing the threshold from predictions made on the same observations used to
fit a model, calibrated out-of-fold probabilities are generated for the
training set.

The threshold search considers each observed out-of-fold probability and
selects the threshold with the greatest specificity among those achieving the
prespecified sensitivity target of at least 0.80. Ties favour the higher
threshold.

After selection, the calibrated estimator is refitted using the full training
set. The held-out test set is scored once with the fixed threshold.

## Evaluation

Threshold-free metrics:

- ROC AUC;
- average precision;
- Brier score.

Threshold-dependent metrics:

- sensitivity;
- specificity;
- precision;
- negative predictive value;
- F1 score;
- accuracy;
- confusion-matrix counts.

Percentile 95% confidence intervals are calculated using 1,000 deterministic
bootstrap resamples of the held-out test set. The threshold remains fixed
during resampling.

Decision-curve net benefit is exported over thresholds from 0.05 to 0.95.

## Subgroup diagnostics

Selection rate, true-positive rate, false-positive rate, support, positive
count, negative count and ROC AUC are reported by:

- sex;
- age group;
- race/ethnicity.

These descriptive diagnostics do not have confidence intervals and are not a
complete fairness analysis.

## Reproducibility

The training summary records the cohort sizes, features, seed, CV configuration,
best parameters, CV score, sensitivity target, selected threshold and
out-of-fold operating metrics. Test outputs separately record the held-out
metrics and bootstrap intervals.
