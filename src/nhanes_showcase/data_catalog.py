from __future__ import annotations

CDC_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

FILE_CATALOG = {
    "DEMO": f"{CDC_BASE}/P_DEMO.xpt",
    "DIQ": f"{CDC_BASE}/P_DIQ.xpt",
    "BMX": f"{CDC_BASE}/P_BMX.xpt",
    "BPXO": f"{CDC_BASE}/P_BPXO.xpt",
    "GLU": f"{CDC_BASE}/P_GLU.xpt",
}

REQUIRED_COLUMNS = {
    "DEMO": [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
        "DMDEDUC2",
        "INDFMPIR",
        "WTMECPRP",
    ],
    "DIQ": ["SEQN", "DIQ010"],
    "BMX": ["SEQN", "BMXBMI"],
    "BPXO": ["SEQN", "BPXOSY1", "BPXODI1"],
    "GLU": ["SEQN", "LBXGLU"],
}


def get_file_catalog(include_glucose: bool = False) -> dict[str, str]:
    names = ["DEMO", "DIQ", "BMX", "BPXO"]
    if include_glucose:
        names.append("GLU")
    return {name: FILE_CATALOG[name] for name in names}


def get_required_columns() -> dict[str, list[str]]:
    return REQUIRED_COLUMNS.copy()
