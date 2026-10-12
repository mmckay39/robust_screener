# rank.py  ────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import pandas as pd

from calc.settings import (
    VAL_STATIC_W,  VAL_TREND_W,
    QUAL_STATIC_W, QUAL_ROIC_W, QUAL_OPM_W,
    GR_EBIT_W,     GR_REV_W,    GR_FCF_W,
    FINAL_VAL_W,   FINAL_QUAL_W, FINAL_GR_W,
)

from calc.settings import (
    sector_value_split,
    sector_quality_split,
    sector_pillar_weights,
)

# ------------------------------- raw‑factor groups ---------------------------
VALUE_FACTORS = [
    "trailing_earnings_yield", "forward_earnings_yield",
    "ev_ebit_yield", "ev_ebitda_yield",
    "fcf_yield", "sales_yield",
]

QUALITY_FACTORS = [
    "roic", "gpa", "accruals_ratio", "operating_profit_margin",
]

BOOST_COLS = [
    "earnings_yield_boost", "roic_boost", "op_margin_boost",
    "ebit_boost", "revenue_boost", "fcf_cagr_boost",
]

# ------------------------------ helper ---------------------------------------
def _sector_pct_rank(s: pd.Series) -> pd.Series:
    """0‑1 percentile ranks inside each sector."""
    return s.rank(pct=True, method="average")


def _dynamic_value_score(row) -> float:
    """
    Blend static & trend for Value, but if the earnings_yield boost is missing,
    fall back to 100% static.
    """
    # 1) load your sector‐specific weight split
    w_static, w_trend = sector_value_split(row["sector"])
    static_pct = row.get("VALUE_SCORE_STATIC", 0.0)
    trend_pct  = row.get("earnings_yield_boost_pct")

    # 2) if no boost data, use 100% static
    if pd.isna(trend_pct):
        return static_pct

    # 3) otherwise blend
    total = w_static + w_trend
    return (w_static/total) * static_pct + (w_trend/total) * trend_pct


def _dynamic_quality_score(row) -> float:
    """
    Combines static QUALITY_SCORE_STATIC with available trend boosts
    (roic_boost_pct and op_margin_boost_pct), using sector‐specific
    splits and falling back to static if *both* boosts are missing.
    """
    # 1) Grab the sector‐specific weights
    w_static, w_trend = sector_quality_split(row["sector"])
    static_pct = row.get("QUALITY_SCORE_STATIC", 0.0)

    # 2) Collect whichever boost‐pct values exist
    boosts = [
        row.get("roic_boost_pct"),
        row.get("op_margin_boost_pct")
    ]
    valid_boosts = [b for b in boosts if not pd.isna(b)]

    # 3) If *no* boost is available, fall back to 100% static
    if not valid_boosts:
        return static_pct

    # 4) Otherwise average the available boosts
    trend_pct = float(sum(valid_boosts) / len(valid_boosts))

    # 5) If that average somehow is NaN, again use static only
    if pd.isna(trend_pct):
        return static_pct

    # 6) Renormalize weights and compute blended quality score
    total = w_static + w_trend
    return (
        (w_static / total) * static_pct
      + (w_trend  / total) * trend_pct
    )

def _dynamic_growth_score(row) -> float | None:
    parts = [
        (GR_EBIT_W, row.get("ebit_boost_pct")),
        (GR_REV_W,  row.get("revenue_boost_pct")),
        (GR_FCF_W,  row.get("fcf_cagr_boost_pct")),
    ]

    # Filter to valid data points
    valid = [(w, v) for w, v in parts if isinstance(v, (int, float)) and not pd.isna(v)]

    if not valid:
        return None  # no valid growth data

    total_weight = sum(w for w, _ in valid)
    score = sum((w / total_weight) * v for w, v in valid)

    return score



def _robust(row) -> float | None:
    """
    Final composite score using sector‑specific pillar weights.
    Dynamically reweights based on which components are present.
    """
    v_w, q_w, g_w = sector_pillar_weights(row["sector"])
    parts = [
        (v_w, row.get("VALUE_SCORE")),
        (q_w, row.get("QUALITY_SCORE")),
        (g_w, row.get("GROWTH_SCORE")),
    ]

    valid = [(w, v) for w, v in parts if isinstance(v, (int, float)) and not pd.isna(v)]
    if not valid:
        return None

    total_weight = sum(w for w, _ in valid)
    return sum((w / total_weight) * v for w, v in valid)



# --------------------------- main entry --------------------------------------
def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies sector‐neutral percentile ranks, dynamic weighting, and computes:
      • VALUE_SCORE_STATIC, VALUE_SCORE
      • QUALITY_SCORE_STATIC, QUALITY_SCORE
      • GROWTH_SCORE
      • ROBUST_SCORE
    Eliminates tickers missing too many static factors.
    """
    # 1) Hard eliminations
    df["value_missing_ct"]   = df[VALUE_FACTORS].isna().sum(axis=1)
    df["quality_missing_ct"] = df[QUALITY_FACTORS].isna().sum(axis=1)
    mask = (df["value_missing_ct"] <= 3) & (df["quality_missing_ct"] <= 2)
    df = df.loc[mask].copy()

    # 2) Static factor percentiles
    for col in VALUE_FACTORS + QUALITY_FACTORS:
        if col in df:
            df[f"{col} pct"] = df.groupby("sector")[col].transform(_sector_pct_rank)

    # 3) Boost percentiles
    for col in BOOST_COLS:
        if col in df:
            df[f"{col}_pct"] = df.groupby("sector")[col].transform(_sector_pct_rank)

    # 4) Static pillar scores
    df["VALUE_SCORE_STATIC"] = df[[f"{c} pct" for c in VALUE_FACTORS if f"{c} pct" in df]].mean(axis=1)
    df["QUALITY_SCORE_STATIC"] = df[[f"{c} pct" for c in QUALITY_FACTORS if f"{c} pct" in df]].mean(axis=1)

    # 5) Dynamic pillar scores
    df["VALUE_SCORE"]   = df.apply(_dynamic_value_score,   axis=1)
    df["QUALITY_SCORE"] = df.apply(_dynamic_quality_score, axis=1)
    df["GROWTH_SCORE"]  = df.apply(_dynamic_growth_score,  axis=1)

    # 6) Robust composite
    df["ROBUST_SCORE"] = df.apply(_robust, axis=1)

    # 7) Cleanup
    df.drop(columns=["value_missing_ct","quality_missing_ct"], inplace=True)

    return df

# Handy list for column ordering
FINAL_ORDER = [
    # raw statics
    "trailing_earnings_yield", "forward_earnings_yield",
    "ev_ebit_yield", "ev_ebitda_yield", "fcf_yield",
    "sales_yield", "roic", "gpa", "accruals_ratio", "operating_profit_margin",
    # static pct
    "trailing_earnings_yield pct", "forward_earnings_yield pct",
    "ev_ebit_yield pct", "ev_ebitda_yield pct", "fcf_yield pct",
    "sales_yield pct", "roic pct", "gpa pct", "accruals_ratio pct", "operating_profit_margin pct",
    # boosts raw
    "earnings_yield_boost", "ebit_boost", "fcf_cagr_boost",
    "revenue_boost", "roic_boost", "op_margin_boost",
    # boosts pct
    "earnings_yield_boost_pct", "ebit_boost_pct", "fcf_cagr_boost_pct",
    "revenue_boost_pct", "roic_boost_pct", "op_margin_boost_pct",
    # pillar scores
    "VALUE_SCORE_STATIC", "QUALITY_SCORE_STATIC", 
    "VALUE_SCORE", "QUALITY_SCORE", 
    "GROWTH_SCORE",    
    "ROBUST_SCORE",
    "HQM_SCORE",
    "DOLLAR_VOLUME", 
    "TURNOVER_RATIO",
    "price"
]
