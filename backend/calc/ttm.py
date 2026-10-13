"""
TTM helpers that **only** depend on pandas.
"""
from typing import Iterable, Mapping

import numpy as np
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
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])

    # Window is anchored to the company's newest quarter, not today: FMP's
    # income/cash-flow stable endpoints often lag by many months, and a
    # today-based cutoff left fewer than min_quarters rows → every TTM was None.
    if not df.empty:
        cutoff_dt = df["date"].max() - pd.Timedelta(days=cutoff_days)
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


def compute_roic_ttm(inc_df: pd.DataFrame | None, bs_df: pd.DataFrame | None) -> float | None:
    """
    Manually replicate key‑metrics‑ttm.returnOnInvestedCapitalTTM
    using income‑statement & balance‑sheet endpoints.
    • Pulls 7 quarters, takes the 4 most‑recent VALID quarters.
    • Returns None if < 4 good quarters or InvestedCapital_avg ≤ 0.
    """
    # NOPAT: Net Operating Profit After Tax
    # NOPAT_TTM = Σ (EBIT × (1 – tax_rate))
    # InvestedCapital_avg = mean( (Debt + Equity − Cash) )
    # ROIC_TTM = NOPAT_TTM / InvestedCapital_avg
    # --- 7 most‑recent quarters (already sorted by 'date' DESC) ----------
    inc_df = inc_df.head(7) if inc_df is not None else None
    bs_df  = bs_df.head(7) if bs_df is not None else None

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

def compute_fcf_yield_ttm(cf_df: pd.DataFrame | None, inc_df: pd.DataFrame | None, price: float) -> float | None:
    """
    Manual replica of key‑metrics‑ttm.freeCashFlowYieldTTM.
    • Pulls up to 7 quarterly cash‑flow and income‑statement rows.
    • Needs:   freeCashFlow  and  weightedAverageShsOutDil.
    • Formula: FCF_TTM  /  (price × shares_outstanding)
      (≈ FCF / market‑cap).  Returns None if data is incomplete.
    """

    if price in (None, 0):
        return None

    # ------------- up to 7 quarters (already date‑sorted) --------------
    cf_df = cf_df.head(7) if cf_df is not None else None
    inc_df = inc_df.head(7) if inc_df is not None else None

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
def compute_ev_to_sales_ttm(inc_df: pd.DataFrame | None, bs_df: pd.DataFrame | None, price: float) -> float | None:
    """
    Manual replication of key‑metrics‑ttm.evToSalesTTM
    (without calling key‑metrics endpoints).

    EV = MarketCap + TotalDebt – Cash
    Sales_TTM = Σ last‑4‑quarter revenues
    Returns None if data incomplete.
    """

    if price in (None, 0):
        return None

    # --- 7 quarters (already sorted newest→oldest) ----------------------
    inc_df = inc_df.head(7) if inc_df is not None else None
    bs_df  = bs_df.head(7) if bs_df is not None else None

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
    inc["date"] = pd.to_datetime(inc["date"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    inc = inc.dropna(subset=["date", "eps"]).sort_values("date")

    # keep only the columns we need
    keep_cols = ["date", "eps"]
    if "symbol" in inc.columns:  keep_cols.append("symbol")
    if "period" in inc.columns:  keep_cols.append("period")
    inc = inc[keep_cols]

    # ---- prep price data ----------------------------------------------------
    
    px["date"] = pd.to_datetime(px["date"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
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
