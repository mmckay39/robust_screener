from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Iterable, Union
from datetime import datetime, timezone

from core.fetchers import fetch_history

#math utilities
Number = Union[int, float, np.number]

today = datetime.now(timezone.utc)

#-----------------------------------------------------------------------------------------------------------------#
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

#-----------------------------------------------------------------------------------------------------------------#
# Format DOLLAR_VOLUME and TURNOVER_RATIO in scientific notation with 4 sig figs
def format_sci_4sf(x):
    return f"{x:.3e}" if pd.notna(x) else ""

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

def compute_roic_ttm(ticker: str) -> float | None:
    """
    Manually replicate key‑metrics‑ttm.returnOnInvestedCapitalTTM
    using income‑statement & balance‑sheet endpoints.
    • Pulls 7 quarters, takes the 4 most‑recent VALID quarters.
    • Returns None if < 4 good quarters or InvestedCapital_avg ≤ 0.
    """
    # get from income-statement and balance-sheet-statement
    # NOPAT: Net Operating Profit After Tax
    # NOPAT_TTM = Σ (EBIT × (1 – tax_rate))
    # InvestedCapital_avg = mean( (Debt + Equity − Cash) )
    # ROIC_TTM = NOPAT_TTM / InvestedCapital_avg
    # --- pull 7 most‑recent quarters (already sorted by 'date' DESC) ----------
    inc_df = fetch_history("income-statement",        ticker, "quarter", limit=7)
    bs_df  = fetch_history("balance-sheet-statement", ticker, "quarter", limit=7)

    if inc_df is None or bs_df is None or inc_df.empty or bs_df.empty:
        return None

    # --- build list of valid quarters (up to 4) --------------------------------
    nopat_list, ic_list = [], []

    # align by index (row 0 = newest). We’ll walk inc_df rows;
    # for each, take the *same‑indexed* row from bs_df (assumes FMP rows line up)
    for idx, inc_row in inc_df.iterrows():
        if len(nopat_list) == 4:                       # already have 4 good quarters
            break

        # ── income statement pieces ──────────────────────────────────────────
        ebit = inc_row.get("ebit")
        ebt  = inc_row.get("incomeBeforeTax") or inc_row.get("ebit")  # safe fallback
        tax_exp = inc_row.get("incomeTaxExpense")

        if any(v in (None, 0) for v in (ebit, ebt)) or ebt <= 0:
            continue                                   # skip invalid quarter

        tax_rate = (tax_exp / ebt) if tax_exp not in (None, 0) else 0.21
        nopat_q  = ebit * (1 - tax_rate)

        # ── balance‑sheet pieces (same idx) ──────────────────────────────────
        try:
            bs_row = bs_df.iloc[idx]
        except IndexError:
            continue                                   # mismatch in rows

        debt_q   = (bs_row.get("shortTermDebt") or 0) + (bs_row.get("longTermDebt") or 0)
        equity_q = bs_row.get("totalStockholdersEquity")
        cash_q   = bs_row.get("cashAndCashEquivalents")

        if equity_q in (None,) or cash_q is None:
            continue

        invested_cap_q = (debt_q + equity_q) - cash_q
        if invested_cap_q <= 0:
            continue                                   # avoid pathological negatives

        # ── collect ----------------------------------------------------------
        nopat_list.append(nopat_q)
        ic_list.append(invested_cap_q)

    # need a full TTM set (4 quarters)
    if len(nopat_list) < 4 or len(ic_list) < 4:
        return None

    nopat_ttm           = sum(nopat_list[:4])
    invested_cap_avg    = sum(ic_list[:4]) / 4

    if invested_cap_avg <= 0:
        return None

    return nopat_ttm / invested_cap_avg

def compute_fcf_yield_ttm(ticker: str, price: float) -> float | None:
    """
    Manual replica of key‑metrics‑ttm.freeCashFlowYieldTTM.
    • Pulls up to 7 quarterly cash‑flow and income‑statement rows.
    • Needs:   freeCashFlow  and  weightedAverageShsOutDil.
    • Formula: FCF_TTM  /  (price × shares_outstanding)
      (≈ FCF / market‑cap).  Returns None if data is incomplete.
    """

    if price in (None, 0):
        return None

    # ------------- pull up to 7 quarters (already date‑sorted) --------------
    cf_df = fetch_history("cash-flow-statement", ticker, "quarter", limit=7)
    inc_df = fetch_history("income-statement",  ticker, "quarter", limit=7)

    if cf_df is None or inc_df is None or cf_df.empty or inc_df.empty:
        return None

    # ------------- build FCF_TTM  (first 4 valid quarters) ------------------
    fcf_list = []
    for idx, cf_row in cf_df.iterrows():
        if len(fcf_list) == 4:
            break
        fcf_q = cf_row.get("freeCashFlow")
        if fcf_q not in (None,):
            fcf_list.append(fcf_q)

    if len(fcf_list) < 4:
        return None

    fcf_ttm = sum(fcf_list[:4])

    # ------------- get latest diluted shares outstanding --------------------
    shares = None
    for _, inc_row in inc_df.iterrows():          # walk newest→oldest
        shs = inc_row.get("weightedAverageShsOutDil")
        if shs not in (None, 0):
            shares = shs
            break

    if shares in (None, 0):
        return None

    # ------------- compute market‑cap and yield -----------------------------
    market_cap = price * shares
    if market_cap <= 0:
        return None

    fcf_yield = fcf_ttm / market_cap              # (≈ freeCashFlow / marketCap)
    # If you want the "per‑share" flavour →  fcf_yield = (fcf_ttm / shares) / price
    #FMP’s key-metrics-ttm.freeCashFlowYieldTTM is defined as:

    # Free Cash Flow / Market Capitalization
    # (i.e., FCF_TTM ÷ (Price × Shares Outstanding))

    return fcf_yield

# calc/ev_to_sales.py  ─────────────────────────────────────────────────────────
def compute_ev_to_sales_ttm(ticker: str, price: float) -> float | None:
    """
    Manual replication of key‑metrics‑ttm.evToSalesTTM
    (without calling key‑metrics endpoints).

    EV = MarketCap + TotalDebt – Cash
    Sales_TTM = Σ last‑4‑quarter revenues
    Returns None if data incomplete.
    """

    if price in (None, 0):
        return None

    # --- pull 7 quarters (already sorted newest→oldest) ----------------------
    inc_df = fetch_history("income-statement",  ticker, "quarter", limit=7)
    bs_df  = fetch_history("balance-sheet-statement", ticker, "quarter", limit=7)

    if inc_df is None or bs_df is None or inc_df.empty or bs_df.empty:
        return None

    # --- Sales_TTM = sum of first 4 valid quarterly revenues -----------------
    rev_list = []
    shares   = None

    for _, inc_row in inc_df.iterrows():
        if len(rev_list) == 4:
            break
        rev_q = inc_row.get("revenue")
        if rev_q not in (None,):
            rev_list.append(rev_q)

        # capture latest non‑zero diluted shares (for market‑cap)
        shs = inc_row.get("weightedAverageShsOutDil")
        if shares in (None, 0) and shs not in (None, 0):
            shares = shs

    if len(rev_list) < 4 or shares in (None, 0):
        return None

    sales_ttm = sum(rev_list[:4])

    # --- Enterprise Value from latest balance‑sheet row ----------------------
    bs_latest = bs_df.iloc[0]
    debt  = (bs_latest.get("shortTermDebt")  or 0) + (bs_latest.get("longTermDebt") or 0)
    cash  =  bs_latest.get("cashAndCashEquivalents") or 0

    market_cap = price * shares
    enterprise_value = market_cap + debt - cash

    if enterprise_value <= 0 or sales_ttm <= 0:
        return None

    return enterprise_value / sales_ttm

def quarterly_op_margin(
    income_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """
    Build a 2‑column quarterly DataFrame with:
        • date
        • operatingProfitMargin  (operatingIncome / revenue)
    Returns None if inputs are missing or empty.
    """
    if (
        income_df is None
        or income_df.empty
        or not {"date", "operatingIncome", "revenue"}.issubset(income_df.columns)
    ):
        return None

    df = income_df[["date", "operatingIncome", "revenue"]].copy()
    # Guard against div‑by‑zero or missing revenue
    df["operatingProfitMargin"] = (
        df["operatingIncome"] / df["revenue"].replace(0, np.nan)
    )
    # Drop rows where margin is NaN (missing or zero revenue)
    df = df.dropna(subset=["operatingProfitMargin"])
    return df[["date", "operatingProfitMargin"]]

def build_earnings_yield_df(income_df: pd.DataFrame,
                            price_df:  pd.DataFrame,
                            *,
                            tolerance_days: int = 7) -> pd.DataFrame | None:
    """
    Create a DataFrame with columns:
        ['date', 'symbol', 'period', 'eps', 'close', 'earnings_yield']

    • Matches each income‑statement row to the last trading‑day close
      **on or before** that date (≤ `tolerance_days` gap).
    • Calculates `earnings_yield = eps / close` (≡ 1 / PE).

    Returns None if inputs are missing or no rows can be matched.
    """
    req_inc = {"date", "eps"}
    req_px  = {"date", "close"}

    # ---- handle price_df with date as index --------------------------------
    px = price_df.copy()
    if "date" not in px.columns and getattr(px.index, "name", None) == "date":
        px = px.reset_index()           # bring the index into a 'date' column

    # ---- basic validation ---------------------------------------------------
    if (
        income_df is None or income_df.empty
        or price_df  is None or price_df.empty
        or not req_inc.issubset(income_df.columns)
        or not req_px.issubset(px.columns)
    ):
        return None

    # ---- prep income‑statement (quarterly) ----------------------------------
    inc = income_df.copy()
    inc["date"] = pd.to_datetime(inc["date"], utc=True, errors="coerce")
    inc = inc.dropna(subset=["date", "eps"]).sort_values("date")

    # keep only the columns we need
    keep_cols = ["date", "eps"]
    if "symbol" in inc.columns:  keep_cols.append("symbol")
    if "period" in inc.columns:  keep_cols.append("period")
    inc = inc[keep_cols]

    # ---- prep price data ----------------------------------------------------
    
    px["date"] = pd.to_datetime(px["date"], utc=True, errors="coerce")
    px = px.dropna(subset=["date", "close"]).sort_values("date")

    # ---- nearest‑merge (asof) ----------------------------------------------
    merged = pd.merge_asof(
        inc,
        px[["date", "close"]],
        on="date",
        direction="backward",                       # ≤ statement date
        tolerance=pd.Timedelta(f"{tolerance_days}D")
    )

    # any rows with no price within tolerance are dropped
    merged = merged.dropna(subset=["close"])

    if merged.empty:
        return None

    # ---- compute earnings yield --------------------------------------------
    merged["earnings_yield"] = merged["eps"] / merged["close"]

    # ---- final column order -------------------------------------------------
    cols = ["date"]
    if "symbol" in merged.columns: cols.append("symbol")
    if "period" in merged.columns: cols.append("period")
    cols += ["eps", "close", "earnings_yield"]

    out = merged[cols].reset_index(drop=True)

    # 1) Strip the timezone/timestamp so you just see YYYY-MM-DD
    out["date"] = out["date"].dt.date

    # 2) (Optional) Round the floats for display
    out["eps"]            = out["eps"].round(4)
    out["close"]          = out["close"].round(2)
    out["earnings_yield"] = out["earnings_yield"].round(6)

    return out

def try_fallbacks(*attempts, verbose_name=None):
    """
    Return the first non-(None, 0) value in attempts.
    Optionally prints which fallback succeeded.
    """
    for idx, val in enumerate(attempts):
        if val not in (None, 0):
            if verbose_name and idx > 0:  # print only if a fallback (not the first try)
                print(f"Used fallback {idx} for {verbose_name}")
            return val
    return None
