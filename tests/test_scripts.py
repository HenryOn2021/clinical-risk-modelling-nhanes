import runpy
from pathlib import Path

import pytest

SCRIPT_PATHS = sorted((Path(__file__).parents[1] / "scripts").glob("*.py"))


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda path: path.name)
def test_script_imports(script_path):
    runpy.run_path(str(script_path), run_name="not_main")
