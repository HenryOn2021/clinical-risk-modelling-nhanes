from __future__ import annotations

CDC_BASE="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"

FILE_CATALOG = {
    "DEMO":f"{CDC_BASE}/P_DEMO.xpt",
    "DIQ":f"{CDC_BASE}/P_DIQ.xpt",
    "BMX":f"{CDC_BASE}/P_BMX.xpt",
    "BPXO":f"{CDC_BASE}/P_BPXO.xpt",
    "GLU":f"{CDC_BASE}/P_GLU.xpt",
}

REQUIRED_COLUMNS = {
    "DEMO":["SEQN","RIDAGEYR","RIAGENDR"],
    "DIQ":["SEQN","DIQ010"],
    "BMX":["SEQN","BMXBMI"],
    "BPXO":["SEQN"],
    "GLU":["SEQN"],
}

def get_file_catalog() -> dict[str,str]: 
    return FILE_CATALOG.copy()
def get_required_columns() -> dict[str,list[str]]: 
    return REQUIRED_COLUMNS.copy()