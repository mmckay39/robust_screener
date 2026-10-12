"""
TTM helpers that **only** depend on pandas & numpy.
"""
from datetime import datetime, timezone, timedelta
from typing import Iterable, Mapping

import pandas as pd


#-----------------------------------------------------------------------------------------------------------------#
def calc_ttm(
    quarterly_data: pd.DataFrame | list[Mapping],
    fields: Iterable[str],
    min_quarters: int = 4,
    cutoff_days: int = 730, #1095 = 3Y, 730 = 2Y, 365 = 1Y
) -> dict[str, float | None]:
    """
    Compute TTM values by summing the *most recent* `min_quarters` (default 4)
    non‑zero quarters for each field.

    Parameters
    ----------
    quarterly_data : DataFrame | list[dict]
        Must contain a 'date' column/key with "YYYY‑MM‑DD" strings and the
        numeric `fields` you want to aggregate.
    fields : iterable[str]
        e.g., ["ebit", "grossProfit", "netIncome"]
    min_quarters : int, default 4
    cutoff_days : int, default 730  (≈ 2 years)

    Returns
    -------
    dict[field, float | None]
        { "ebit": 1_234_567, "netIncome": None, ... }
    """
    # ------------------------------------------------------------------ #
    # 1. Normalise input to DataFrame                                     #
    # ------------------------------------------------------------------ #
    if quarterly_data is None:
        return {field: None for field in fields}

    if isinstance(quarterly_data, list):
        df = pd.DataFrame(quarterly_data)
    else:
        df = quarterly_data.copy()

    if df.empty or "date" not in df.columns:
        return {field: None for field in fields}

    # ------------------------------------------------------------------ #
    # 2. Keep only recent quarters (≤ cutoff_days)                        #
    # ------------------------------------------------------------------ #
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=cutoff_days)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= cutoff_dt]

    # Sort newest → oldest so .head(min_quarters) gives latest values
    df = df.sort_values("date", ascending=False)

    # ------------------------------------------------------------------ #
    # 3. Aggregate TTM per field                                          #
    # ------------------------------------------------------------------ #
    ttm: dict[str, float | None] = {}

    for field in fields:
        if field not in df.columns:
            ttm[field] = None
            continue

        # select non‑zero, non‑null values
        vals = df.loc[df[field].replace(0, pd.NA).notna(), field].head(min_quarters)

        ttm[field] = vals.sum() if len(vals) == min_quarters else None

    return ttm