from __future__ import annotations

import pandas as pd

from .config import ID_COL
from .data_catalog import get_required_columns


class SchemaError(ValueError):
    """Raised when a table does not meet the expected contract."""


def require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{table_name}: missing required columns {missing}")


def ensure_unique_key(df: pd.DataFrame, key: str = ID_COL, table_name: str = "table") -> None:
    if key not in df.columns:
        raise SchemaError(f"{table_name}: missing key column {key}")
    if df[key].duplicated().any():
        n_dup = int(df[key].duplicated().sum())
        raise SchemaError(f"{table_name}: found {n_dup} duplicate {key} values")


def validate_numeric_range(
    df: pd.DataFrame,
    column: str,
    low: float | None = None,
    high: float | None = None,
    table_name: str = "table",
) -> None:
    if column not in df.columns:
        return
    series = pd.to_numeric(df[column], errors="coerce")
    if low is not None and (series.dropna() < low).any():
        raise SchemaError(f"{table_name}: {column} contains values below {low}")
    if high is not None and (series.dropna() > high).any():
        raise SchemaError(f"{table_name}: {column} contains values above {high}")


def validate_demo(df: pd.DataFrame) -> None:
    require_columns(df, get_required_columns()["DEMO"], "DEMO")
    ensure_unique_key(df, table_name="DEMO")
    validate_numeric_range(df, "RIDAGEYR", 0, 120, "DEMO")
    validate_numeric_range(df, "RIAGENDR", 1, 2, "DEMO")


def validate_diq(df: pd.DataFrame) -> None:
    require_columns(df, get_required_columns()["DIQ"], "DIQ")
    ensure_unique_key(df, table_name="DIQ")


def validate_bmx(df: pd.DataFrame) -> None:
    require_columns(df, get_required_columns()["BMX"], "BMX")
    ensure_unique_key(df, table_name="BMX")
    validate_numeric_range(df, "BMXBMI", 10, 100, "BMX")


def validate_bpxo(df: pd.DataFrame) -> None:
    require_columns(df, get_required_columns()["BPXO"], "BPXO")
    ensure_unique_key(df, table_name="BPXO")
    for column in ["BPXOSY1", "BPXOSY2", "BPXOSY3"]:
        validate_numeric_range(df, column, 40, 300, "BPXO")
    for column in ["BPXODI1", "BPXODI2", "BPXODI3"]:
        validate_numeric_range(df, column, 20, 200, "BPXO")


def validate_glu(df: pd.DataFrame) -> None:
    require_columns(df, get_required_columns()["GLU"], "GLU")
    ensure_unique_key(df, table_name="GLU")
    validate_numeric_range(df, "LBXGLU", 20, 700, "GLU")


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    required = {"DEMO", "DIQ", "BMX", "BPXO"}
    missing = required - set(tables)
    if missing:
        raise SchemaError(f"Missing required tables: {sorted(missing)}")
    validate_demo(tables["DEMO"])
    validate_diq(tables["DIQ"])
    validate_bmx(tables["BMX"])
    validate_bpxo(tables["BPXO"])
    if "GLU" in tables:
        validate_glu(tables["GLU"])
