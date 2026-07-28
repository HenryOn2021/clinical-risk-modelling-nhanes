from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_STEMS = [
    "00_project_overview",
    "01_data_and_cohort",
    "02_eda_and_baseline",
    "03_modelling_and_evaluation",
]


def test_notebook_pairs_are_present_and_valid() -> None:
    for stem in NOTEBOOK_STEMS:
        notebook_path = NOTEBOOK_DIR / f"{stem}.ipynb"
        paired_source_path = NOTEBOOK_DIR / f"{stem}.py"

        assert notebook_path.exists()
        assert paired_source_path.exists()

        notebook = nbformat.read(notebook_path, as_version=4)
        assert notebook.cells
        assert notebook.metadata["jupytext"]["formats"] == "ipynb,py:percent"
        assert notebook.metadata["kernelspec"]["name"] == "python3"


def test_distributed_notebooks_are_clean_and_safe() -> None:
    for stem in NOTEBOOK_STEMS:
        notebook_path = NOTEBOOK_DIR / f"{stem}.ipynb"
        notebook = nbformat.read(notebook_path, as_version=4)

        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert cell.outputs == []
                source_lines = [line.strip() for line in cell.source.splitlines()]
                assert "RUN_TRAINING = True" not in source_lines
