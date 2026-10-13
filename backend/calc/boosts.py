from __future__ import annotations
# boosts.py
# ─────────────────────────────────────────────────────────────────────────────
# Trend‑based “boost” factors, each returning a single float or None.
#
# Design rules
# ------------
# • Signature:   func(df: pd.DataFrame) -> float | None
# • If *df* is not a DataFrame, missing required columns, or too few points,
#   return None.
# • All slopes are computed on the most‑recent 8 quarterly observations.
# • Each slope is normalised by a scale factor so different boosts live on
#   roughly comparable ranges; we still z‑score across the universe later.
#
# Implemented boosts
# ------------------
#   ebit_boost            — EBIT trend  (positive = improving earnings)
#   revenue_boost         — Sales trend
#   op_margin_boost       — Operating‑margin trend   (percentage units)
#   roic_boost            — ROIC trend               (percentage units)
#   fcf_boost             — Free‑cash‑flow trend
#   pe_multiple_boost     — NEGATIVE PE slope (falling PE is good)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

from calc.stats import slope_uniform

__all__ = [
    "ebit_boost",
    "revenue_boost",
    "op_margin_boost",
    "roic_boost",
    "fcf_cagr_boost",
    "earnings_yield_boost",
]
# ─────────────────────────────────────────────────────────────────────────────
# Boosts
# ─────────────────────────────────────────────────────────────────────────────

def ebit_boost(iq_df: pd.DataFrame, *, recent_quarters: int = 8) -> float | None:
    """
    Compute a dimension‑less “trend boost” for EBIT.

    Parameters
    ----------
    iq_df : pd.DataFrame
        Quarterly *income‑statement* history containing at least:
            • 'date'  – period‑end date (any sortable type)
            • 'ebit'  – earnings before interest & tax
        Rows may be in any order; the function will sort internally.
    recent_quarters : int, default 8
        How many of the most‑recent **non‑zero** quarters to include
        when fitting the trend. 8 ≈ 2 years of data.

    Returns
    -------
    float | None
        • Positive → upward EBIT trend (good).  
        • Negative → deteriorating EBIT (bad).  
        • `None`   → insufficient data ( <4 usable quarters ) or
                     mean |EBIT| == 0 (prevents divide‑by‑zero).

    Method
    ------
    1. Sort **newest→oldest** and keep ≤12 quarters (safety‑cap).
    2. Remove NaNs / zeros so we don’t treat “no EBIT” as real data.
    3. Take the latest `recent_quarters` valid rows.
    4. Require at least **4** data points to avoid noisy slopes.
    5. Sort **oldest→newest** and fit an OLS slope against an
       implicit x‑axis `[0, 1, 2, …]` via `slope_uniform(y)`.
    6. Normalise by `mean(|EBIT|)` so the boost is *dimension‑less*
       and comparable across differently‑sized companies.
    """

    # Guard clauses: empty input or missing column
    if iq_df is None or iq_df.empty or "ebit" not in iq_df.columns:
        return None

    # 1️⃣  Standardise order → newest first, cap at 12 rows
    iq_df = iq_df.sort_values("date", ascending=False).head(12)

    # 2️⃣  Keep rows where EBIT is not NaN/None/0
    valid = iq_df.loc[~iq_df["ebit"].isin((None, 0)), ["date", "ebit"]]

    # 3️⃣  Grab the most‑recent `recent_quarters` valid rows
    recent = valid.head(recent_quarters)

    # 4️⃣  Need a minimum history depth
    if len(recent) < 4:
        return None

    # 5️⃣  Oldest → newest for slope calculation
    recent = recent.sort_values("date")

    # Use evenly‑spaced x‑axis [0, 1, …] because quarters are uniform
    slope = slope_uniform(recent["ebit"].to_numpy())
    if slope is None:
        return None

    # 6️⃣  Scale by average absolute EBIT to get a unit‑less coefficient
    mean_abs_ebit = np.abs(recent["ebit"]).mean()
    if mean_abs_ebit == 0:
        return None

    return slope / mean_abs_ebit

def revenue_boost(iq_df: pd.DataFrame, *, years: int = 4) -> float | None:
    """
    Compute a **CAGR‑style boost** for revenue growth.

    Parameters
    ----------
    iq_df : pd.DataFrame
        Quarterly income‑statement history that includes:
            • 'date'     – period‑end date (datetime‑like)
            • 'revenue'  – quarterly revenue
    years : int, default 4
        Number of most‑recent **fiscal years** to evaluate
        (4 years ≈ 16 quarters).

    Returns
    -------
    float | None
        • Positive  → expanding top‑line trajectory.  
        • Negative  → shrinking revenue.  
        • `None`    → insufficient or invalid data (explained below).

    Algorithm
    ---------
    1. Sort newest→oldest, cap at 12 quarters (safety).
    2. Aggregate quarterly revenue to **calendar‑year totals**.
    3. Keep the latest `years` with non‑zero revenue; require ≥2 years.
    4. Compute CAGR:  
           CAGR = (Rev_last / Rev_first) ** (1 / Δyears) − 1
       (Δyears uses the actual year difference to handle gaps.)
    """

    # ------------------------------------------------------------------
    # 0️⃣  Guard clauses
    # ------------------------------------------------------------------
    if (
        iq_df is None
        or iq_df.empty
        or "revenue" not in iq_df.columns
        or "date" not in iq_df.columns
    ):
        return None

    # ------------------------------------------------------------------
    # 1️⃣  Standardise quarter list → newest first, cap at 12 quarters
    # ------------------------------------------------------------------
    iq_df = iq_df.sort_values("date", ascending=False).head(12)

    # ------------------------------------------------------------------
    # 2️⃣  Aggregate to annual revenue (calendar year)
    # ------------------------------------------------------------------
    annual_rev = (
        iq_df
        .assign(year=iq_df["date"].dt.year)
        .groupby("year", sort=False)["revenue"]
        .sum()
        .reset_index()
        .sort_values("year", ascending=False)       # newest first
    )

    # Remove NaNs / zeros
    annual_rev = annual_rev.loc[~annual_rev["revenue"].isin((None, 0))]

    # Need at least two fiscal years for a CAGR
    if len(annual_rev) < 2:
        return None

    # ------------------------------------------------------------------
    # 3️⃣  Take the most‑recent <years> rows
    # ------------------------------------------------------------------
    recent = annual_rev.head(years)
    if len(recent) < 2:
        return None                # still need ≥2 years

    rev_latest   = recent.iloc[0]["revenue"]
    rev_earliest = recent.iloc[-1]["revenue"]
    year_latest  = recent.iloc[0]["year"]
    year_earliest= recent.iloc[-1]["year"]
    delta_years  = year_latest - year_earliest

    # ------------------------------------------------------------------
    # 4️⃣  CAGR calculation (no CAGR when either endpoint is ≤ 0)
    # ------------------------------------------------------------------
    if rev_latest <= 0 or rev_earliest <= 0 or delta_years <= 0:
        return None

    cagr = (rev_latest / rev_earliest) ** (1 / delta_years) - 1
    return float(cagr)

def op_margin_boost(
    margin_df: pd.DataFrame | None,
    *,
    recent_quarters: int = 8,
) -> float | None:
    """
    Trend boost for quarterly Operating‑Profit‑Margin (OPM).

    Parameters
    ----------
    margin_df : pd.DataFrame | None
        • Must contain columns 'date' and 'operatingProfitMargin'.
        • Expected newest→oldest or oldest→newest; we sort anyway.
    recent_quarters : int
        Fit the slope on up to this many most‑recent non‑zero quarters.

    Returns
    -------
    float | None
        Scaled slope (unit‑less).  None if < 4 usable rows.
    """
    # 0️⃣  Validate input
    if (
        margin_df is None
        or margin_df.empty
        or not {"date", "operatingProfitMargin"}.issubset(margin_df.columns)
    ):
        return None

    # 1️⃣  Newest→oldest, cap at 12 rows
    df = margin_df.sort_values("date", ascending=False).head(12)

    # 2️⃣  Filter out NaN / zero margins
    valid = df.loc[
        ~df["operatingProfitMargin"].isin((None, 0)),
        ["date", "operatingProfitMargin"],
    ]

    # 3️⃣  Keep at most `recent_quarters`
    recent = valid.head(recent_quarters)
    if len(recent) < 4:
        return None

    # 4️⃣  Oldest→newest for slope
    recent = recent.sort_values("date")

    slope = slope_uniform(recent["operatingProfitMargin"].to_numpy())
    if slope is None:
        return None

    mean_abs = np.abs(recent["operatingProfitMargin"]).mean()
    if mean_abs == 0:
        return None

    return float(slope / mean_abs)

def roic_boost(
    km_df: pd.DataFrame,
    *,
    years: int = 4
) -> float | None:
    """
    Trend‑based boost for **Return on Invested Capital (ROIC)**.

    Parameters
    ----------
    km_df : pd.DataFrame
        Annual **key‑metrics** dataframe that must contain:
            • 'date'                    – fiscal period‑end (datetime‑like)
            • 'returnOnInvestedCapital' – ROIC ratio (‑1 → 1+)
        Rows may be unordered; the function handles sorting.
    years : int, default 4
        Number of most‑recent **annual** observations to evaluate.

    Returns
    -------
    float | None
        • Positive  → ROIC is trending upward.  
        • Negative  → ROIC is deteriorating.  
        • `None`    → insufficient data or mean |ROIC| == 0.

    Method
    ------
    1. Sort newest→oldest, keep a small buffer (≤ 6 rows).
    2. Remove NaNs / zeros (zero = “missing” on FMP).
    3. Retain the latest `years` valid rows (need ≥ 2 to form a slope).
    4. Sort oldest→newest and fit an OLS slope against an implicit
       x‑axis `[0,1,2,…]` via `slope_uniform(y)`.
    5. Normalise by mean |ROIC| so the boost is *unit‑less* and
       comparable across companies.
    """

    # ------------------------------------------------------------------
    # 0️⃣  Early validation
    # ------------------------------------------------------------------
    if (
        km_df is None
        or km_df.empty
        or "returnOnInvestedCapital" not in km_df.columns
        or "date" not in km_df.columns
    ):
        return None

    # ------------------------------------------------------------------
    # 1️⃣  Standardise order & limit rows
    # ------------------------------------------------------------------
    km_df = km_df.sort_values("date", ascending=False).head(6)

    # ------------------------------------------------------------------
    # 2️⃣  Filter valid ROIC rows
    # ------------------------------------------------------------------
    valid = km_df.loc[
        ~km_df["returnOnInvestedCapital"].isin((None, 0)),
        ["date", "returnOnInvestedCapital"],
    ]

    # ------------------------------------------------------------------
    # 3️⃣  Take the newest `years` valid points
    # ------------------------------------------------------------------
    recent = valid.head(years)
    if len(recent) < 2:
        return None                          # need ≥2 points

    # ------------------------------------------------------------------
    # 4️⃣  Oldest→newest for slope calc
    # ------------------------------------------------------------------
    recent = recent.sort_values("date")
    slope = slope_uniform(recent["returnOnInvestedCapital"].to_numpy())
    if slope is None:
        return None

    # ------------------------------------------------------------------
    # 5️⃣  Scale by mean absolute ROIC
    # ------------------------------------------------------------------
    mean_abs = np.abs(recent["returnOnInvestedCapital"]).mean()
    if mean_abs == 0:
        return None

    return float(slope / mean_abs)           # dimension‑less trend score

def fcf_cagr_boost(
    cf_df: pd.DataFrame | None,
    *,
    years: int = 4
) -> float | None:
    """
    3‑ to 4‑year CAGR of free cash flow (annual totals).

    Returns
    -------
    float | None
        CAGR (unitless) or None if insufficient data.

    """
    if (
        cf_df is None
        or cf_df.empty
        or "freeCashFlow" not in cf_df.columns
        or "date" not in cf_df.columns
    ):
        return None


    # 1) newest→oldest, keep 16 quarters (4 years) in case we got quarterlies
    cf_df = cf_df.sort_values("date", ascending=False).head(16)

    # 2) Aggregate to calendar‑year FCF
    annual_fcf = (
        cf_df
        .assign(year=cf_df["date"].dt.year)
        .groupby("year", sort=False)["freeCashFlow"]
        .sum(min_count=1)             # keep NaN if all NaN
        .reset_index()
        .sort_values("year", ascending=False)
    )

    # remove non‑positive or missing
    annual_fcf = annual_fcf.loc[annual_fcf["freeCashFlow"] > 0]

    if len(annual_fcf) < 2:
        return None

    recent = annual_fcf.head(years)
    if len(recent) < 2:
        return None

    fcf_latest   = recent.iloc[0]["freeCashFlow"]
    fcf_earliest = recent.iloc[-1]["freeCashFlow"]
    year_latest  = recent.iloc[0]["year"]
    year_earliest= recent.iloc[-1]["year"]
    delta_years  = year_latest - year_earliest

    if fcf_earliest <= 0 or delta_years == 0:
        return None

    cagr = (fcf_latest / fcf_earliest) ** (1 / delta_years) - 1
    return float(cagr)


def earnings_yield_boost(ey_df: pd.DataFrame | None,
                         *, recent_qtrs: int = 8) -> float | None:
    """
    Quarterly earnings‑yield slope (unit‑less).  Positive = P/E compressing.
    """
    if ey_df is None or ey_df.empty:
        return None

    ey_df = (
        ey_df.dropna(subset=["earnings_yield"])
             .sort_values("date", ascending=False)
             .head(12)                                   # safety cap
    )

    if len(ey_df) < 4:
        return None

    recent = ey_df.head(recent_qtrs).sort_values("date")
    y = recent["earnings_yield"].to_numpy()
    slope = slope_uniform(y)
    if slope is None:
        return None

    mean_abs = np.abs(y).mean()
    return float(slope / mean_abs) if mean_abs else None
