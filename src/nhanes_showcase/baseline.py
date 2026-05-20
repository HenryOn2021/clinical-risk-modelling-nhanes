from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DEFAULT_BASELINE_FORMULA = """
target_diabetes ~ age_years + C(sex) + bmi + systolic_bp_mean
+ C(race_ethnicity) + C(education_level)
"""

def prepare_baseline_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "target_diabetes", 
        "age_years", 
        "sex", 
        "bmi", 
        "systolic_bp_mean", 
        "race_ethnicity", 
        "education_level",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing].dropna().copy()

def fit_logit_formula(df: pd.DataFrame, formula: str = DEFAULT_BASELINE_FORMULA):
    return smf.logit(formula=formula, data=df).fit(disp=False)

def odds_ratio_table(result) -> pd.DataFrame:
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "odds_ratio": np.exp(result.params.values),
            "ci_low": np.exp(ci[0].values),
            "ci_high": np.exp(ci[1].values),
            "p_value": result.pvalues.values,
        }
    )
    return table.loc[table["term"] != "Intercept"].reset_index(drop=True)

def save_baseline_outputs(result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_summary.txt").write_text(result.summary2().as_text(), encoding="utf-8")
    odds_ratio_table(result).to_csv(out_dir / "odds_ratios.csv", index=False)