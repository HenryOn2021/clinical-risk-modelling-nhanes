\# NHANES Diabetes Showcase

A fast, credible end-to-end clinical data science portfolio project using NHANES 2017–March 2020 pre-pandemic data.



!\[ROC curve](reports/figures/roc\_test.png)

!\[Calibration curve](reports/figures/calibration\_test.png)



\## Why this project

This repository demonstrates a production-style workflow for a public clinical dataset:

\-reproducible data ingestion from CDC XPT files

\-cohort definition and schema/QC checks

\-missingness analysis and outlier handling

\-weighted descriptive statistics

\-interpretable baseline modelling with `statsmodels`

\-leakage-safe ML pipeline with `scikit-learn`

\-calibration, thresholding, subgroup fairness checks, and a model card



\## Assumption

Target = self-reported diabetes from NHANES diabetes questionnaire (`DIQ010`).

\## Quick start

```bash

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

pip install -e ".\[dev]"

pre-commit install

python scripts/run\_ingest.py

python scripts/build\_dataset.py

python scripts/run\_eda.py

python scripts/run\_baseline.py

python scripts/run\_ml.py --minimum-sensitivity 0.80

python scripts/run\_evaluation.py --decision-curve

python scripts/export\_assets.py

