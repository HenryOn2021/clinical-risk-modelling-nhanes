#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging

from nhanes_showcase.config import ensure_project_dirs
from nhanes_showcase.data_catalog import get_file_catalog
from nhanes_showcase.ingest import download_catalog

logging.basicConfig(level=logging.INFO,format="%(levelname)s%(message)s")

def main() -> None:
    parser = argparse.ArgumentParser(description="Download NHANES XPT files from CDC")
    parser.add_argument("--overwrite",action="store_true",help="Re-download existing files")
    args = parser.parse_args()

    ensure_project_dirs()
    files = download_catalog(get_file_catalog(), overwrite=args.overwrite)
    for name, path in files.items():
        print(f"{name}: {path}")
        
if __name__ == "__main__":
    main()