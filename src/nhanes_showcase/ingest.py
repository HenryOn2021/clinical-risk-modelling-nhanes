from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, ensure_project_dirs
from .data_catalog import get_file_catalog

logger = logging.getLogger(__name__)


def _decode_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].apply(
            lambda x: x.decode("utf-8", errors="ignore").strip() if isinstance(x, bytes) else x
        )
    return out


def load_xpt(path_or_url: str | Path) -> pd.DataFrame:
    df = pd.read_sas(path_or_url, format="xport")
    df.columns = [str(c).upper() for c in df.columns]
    return _decode_object_columns(df)


def download_file(url: str, out_path: Path, overwrite: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        logger.info("Using existing file: %s", out_path)
        return out_path

    temporary_path = out_path.with_suffix(f"{out_path.suffix}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "nhanes-diabetes-showcase/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        if temporary_path.stat().st_size == 0:
            raise OSError(f"Downloaded file is empty: {url}")
        temporary_path.replace(out_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info("Downloaded %s -> %s", url, out_path)
    return out_path


def download_catalog(
    catalog: dict[str, str] | None = None, raw_dir: Path = RAW_DIR, overwrite: bool = False
) -> dict[str, Path]:
    ensure_project_dirs()
    catalog = catalog or get_file_catalog()
    downloaded: dict[str, Path] = {}
    for name, url in catalog.items():
        out_path = raw_dir / f"{name}.xpt"
        downloaded[name] = download_file(url, out_path, overwrite=overwrite)
    return downloaded


def load_local_tables(
    raw_dir: Path = RAW_DIR, names: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    names = names or list(get_file_catalog().keys())
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        path = raw_dir / f"{name}.xpt"
        if path.exists():
            tables[name] = load_xpt(path)
    return tables


def load_required_tables(
    overwrite: bool = False, include_glucose: bool = False
) -> dict[str, pd.DataFrame]:
    catalog = get_file_catalog(include_glucose=include_glucose)
    download_catalog(catalog, RAW_DIR, overwrite=overwrite)
    return load_local_tables(RAW_DIR, list(catalog.keys()))
