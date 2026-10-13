# calc/stats.py
"""Small numeric helpers: CAGR, OLS slope, scientific-format string."""
from __future__ import annotations
import numpy as np
from typing import Sequence, Iterable, Union

#math utilities
Number = Union[int, float, np.number]

# ------------------------------------------------------------------ #
# 1. CAGR (Compounded Annual Growth Rate)                            #
# ------------------------------------------------------------------ #
def cagr(start_value: Number, end_value: Number, years: float) -> float | None:
    """
    CAGR = (end / start) ** (1/years) - 1
    Returns None for invalid inputs (zeros, negatives, or years ≤ 0).
    """
    try:
        if start_value is None or end_value is None or start_value <= 0 or years <= 0:
            return None
        return (end_value / start_value) ** (1.0 / years) - 1.0
    except (ZeroDivisionError, OverflowError, TypeError):
        return None

#-----------------------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------ #
# 2. Linear slope (ordinary‑least‑squares β₁)                         #
# ------------------------------------------------------------------ #
def linear_slope(x: Iterable[Number], y: Iterable[Number]) -> float | None:
    """
    OLS slope of y on x. Handles list‑likes, pandas Index / Series, numpy arrays.
    • If `x` is a DatetimeIndex, it’s converted to ordinal days.
    Returns None when < 2 points or std(x) == 0.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.size != y_arr.size or x_arr.size < 2:
        return None

    # Convert datetime → float days since epoch
    if np.issubdtype(x_arr.dtype, np.datetime64):
        x_arr = x_arr.astype("datetime64[D]").astype(float)

    x_mean = x_arr.mean()
    y_mean = y_arr.mean()
    denom = np.dot(x_arr - x_mean, x_arr - x_mean)
    if denom == 0:
        return None  # all x identical

    slope = np.dot(x_arr - x_mean, y_arr - y_mean) / denom
    return float(slope)

def slope_uniform(y: Sequence[Number]) -> float | None:
    """
    OLS slope of y against an implicit x-axis [0, 1, 2, …].

    Keeps `linear_slope(x, y)` untouched for arbitrary x.
    """
    y_arr = np.asarray(y, dtype=float)
    if y_arr.size < 2 or np.all(np.isnan(y_arr)):
        return None
    x_arr = np.arange(y_arr.size, dtype=float)
    return linear_slope(x_arr, y_arr)          # <-- existing 2‑arg helper
