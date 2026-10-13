# calc/fallbacks.py
"""Fallback helpers: pick the first usable value, or the latest annual value."""
from __future__ import annotations
from datetime import datetime, timezone

today = datetime.now(timezone.utc)

def annual_fallback(annual_df, field: str, ttm_value):
    """
    Replace a missing/zero TTM value with the most‑recent annual value
    (within the last 3 years).  Returns the original TTM value if OK.

    Parameters
    ----------
    annual_df    : DataFrame   # key‑metrics annual history, newest first
    annual_field : str         # e.g. "evToEBITDA"
    ttm_value    : float|None  # current TTM value

    Returns
    -------
    float | None
    """
    if ttm_value not in (None, 0):
        return ttm_value

    for idx in range(min(2, len(annual_df))):
        row = annual_df.iloc[idx]
        if (today - row["date"]).days < 365 * 3:            
            annual_val = row.get(field)
            if annual_val not in (None, 0):
                return annual_val
    return None

def try_fallbacks(*attempts):
    """
    Return the first non-(None, 0) value in attempts.
    """
    for val in attempts:
        if val not in (None, 0):
            return val
    return None
