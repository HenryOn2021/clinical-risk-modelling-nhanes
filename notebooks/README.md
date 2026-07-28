# Guided notebooks

These notebooks explain and inspect the same pipeline implemented in `src/` and
orchestrated by `scripts/`. They deliberately reuse the package functions and
saved artifacts instead of duplicating modelling logic.

## Recommended order

1. `00_project_overview.ipynb` - repository map, headline results and figures.
2. `01_data_and_cohort.ipynb` - source tables, validation, linkage, cleaning,
   target definition and cohort construction.
3. `02_eda_and_baseline.ipynb` - missingness, survey-weighted summaries,
   subgroup prevalence and the interpretable statistical baseline.
4. `03_modelling_and_evaluation.ipynb` - split, features, cross-validation,
   calibration, threshold selection, held-out metrics and subgroup diagnostics.

Each `.ipynb` has a paired Jupytext `py:percent` file. Edit either representation
and run `jupytext --sync notebooks/*.ipynb` to keep the pair aligned.

## Launch

From the repository root:

```bash
uv sync --extra dev
uv run jupyter lab
```

Alternatively, create and activate `.venv` manually as documented in the main
README, then run:

```bash
jupyter lab
```

The notebooks expect the current working directory to be either the repository
root or `notebooks/`. They use the versioned aggregate outputs and the local data
files supplied with the corrected review archive.

## Source of truth

The ordered command-line scripts remain the authoritative reproducible
workflow. Notebook `03` contains an opt-in cell for rerunning model training and
evaluation; it is disabled by default because the versioned outputs already
capture the verified run.
