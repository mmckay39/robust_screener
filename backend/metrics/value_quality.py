from __future__ import annotations
"""
calc/value_quality.py
───────────────────────────────────────────────────────────────────────────────
Adds derived **Value** and **Quality / Profitability** metrics to the master
DataFrame that already contains raw TTM & point‑in‑time fields.

Call `add_value_quality_metrics(df)` *after* `run_metrics.py` has assembled its
DataFrame but **before** percentile‑ranking / scoring logic.

The function works in‑place (mutates the DataFrame) and also returns it so you
can write `df = add_value_quality_metrics(df)`.
───────────────────────────────────────────────────────────────────────────────"""

import numpy as np
import pandas as pd

__all__ = ["add_value_quality_metrics"]


# ── main API ─────────────────────────────────────────────────────────────────

def add_value_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived value & quality factors – yields and profitability."""

    # ── Value factors (all expressed as yields: higher = cheaper) ────────────
    df["trailing_earnings_yield"] = np.where(        # avoid /0
        df["priceToEarningsRatioTTM"].replace(0, np.nan).notna(),
        1 / df["priceToEarningsRatioTTM"],
        np.nan,
    )

    df["forward_earnings_yield"] = df["epsAvgTTM"] / df["price"]

    df["ev_ebitda_yield"] = 1 / df["evToEBITDATTM"]

    # EV/EBIT yield (guard against division‑by‑zero EV)
    df["ev_ebit_yield"] = np.where(
        df["enterpriseValue"].replace(0, np.nan).notna(),
        df["ebitTTM"] / df["enterpriseValue"],
        np.nan,
    )

    # FCF yield already supplied by FMP in yield form
    df["fcf_yield"] = df["freeCashFlowYieldTTM"]

    df["sales_yield"] = 1 / df["evToSalesTTM"]

    # ── Quality / Profitability factors ─────────────────────────────────────
    df["roic"] = df["returnOnInvestedCapitalTTM"]

    df["gpa"] = df["grossProfitTTM"] / df["totalAssets"]

    df["accruals_ratio"] = np.where(
        df["totalAssets"].replace(0, np.nan).notna(),
        (df["netIncomeTTM"] - df["operatingCashFlowTTM"]) / df["totalAssets"],
        np.nan,
    )

    df["operating_profit_margin"] = df["operatingProfitMarginTTM"]

    return df
