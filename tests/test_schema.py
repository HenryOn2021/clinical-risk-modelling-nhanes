import pandas as pd
import pytest

from nhanes_showcase.schema import SchemaError, ensure_unique_key, require_columns


def test_require_columns_lists_missing_fields():
    with pytest.raises(SchemaError, match="MISSING"):
        require_columns(pd.DataFrame({"SEQN": [1]}), ["SEQN", "MISSING"], "DEMO")


def test_unique_key_rejects_duplicates():
    with pytest.raises(SchemaError, match="duplicate"):
        ensure_unique_key(pd.DataFrame({"SEQN": [1, 1]}), table_name="DEMO")
