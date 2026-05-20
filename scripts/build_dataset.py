#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from nhanes_showcase.clean import build_adult_analysis_frame, summarise_qc
from nhanes_showcase.config import INTERIM_DIR, PROCESSED_DIR, ensure_project_dirs
from nhanes_showcase.features import define_target, engineer_features, finalise_analysis_dataset
from nhanes_showcase.ingest import download_catalog,load_local_tables
from nhanes_showcase.data_catalog import get_file_catalog

def main() -> None:
    parser = argparse.ArgumentParser(description="Build analysis dataset")
    parser.add_argument("--include-glucose", action="store_true", help="Keep fasting glucose as a feature")
    parser.add_argument("--overwrite-downloads", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    download_catalog(get_file_catalog(), overwrite=args.overwrite_downloads)
    tables = load_local_tables()
    adult = build_adult_analysis_frame(tables)
    adult = define_target(adult)
    adult = engineer_features(adult)

    interim_path = INTERIM_DIR / "nhanes_adult_interim.parquet"
    adult.to_parquet(interim_path, index=False)

    analysis = finalise_analysis_dataset(adult, include_glucose=args.include_glucose)
    analysis_path = PROCESSED_DIR / "analysis_dataset.parquet"
    analysis.to_parquet(analysis_path, index=False)

    qc = summarise_qc(adult)
    (INTERIM_DIR / "qc_summary.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    
    print(f"Saved interim dataset: {interim_path}")
    print(f"Saved analysis dataset: {analysis_path}")

if __name__ == "__main__":
    main()