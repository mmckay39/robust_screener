from __future__ import annotations
"""
run_metrics.py  ────────────────────────────────────────────────────────────────
Pulls key value/quality metrics for a chosen universe (S&P 500 by default),
builds a tidy DataFrame, applies the same PE-fallback logic you used in the
notebook, fills sensible fall-backs for EV/EBITDA, ROIC and EPS-avg, converts
zero-as-missing to NaN, and prints a quick completeness report.

Usage
-----
    python run_metrics.py                # default S&P 500
    python run_metrics.py --universe nasdaq
    python run_metrics.py --universe nyse

The script expects that your project folder contains:
    core/cache.py        (cached_get_json)
    core/fetchers.py     (fetch_ttm, fetch_single, fetch_statement, fetch_analyst_estimates)
    calc/ttm.py          (calc_ttm helper)
    calc/value_quality.py (add_value_quality_metrics)
    universes.py         (download_SP500, download_NASDAQ, …)

Edit `RATIO_FIELDS`, `KEY_METRIC_FIELDS`, etc. as your metric set evolves.
─────────────────────────────────────────────────────────────────────────────
"""
from datetime import datetime, timezone
from core.fundamentals_cache import refresh_if_needed
import pytz

today = datetime.now(timezone.utc)
# ── project imports ──────────────────────────────────────────────────────────
from core.fetchers import (
    fetch_ttm,              # universal TTM helper  (ratios-ttm, key-metrics-ttm, cash-flow-statement, …)
    fetch_single,           # single-object endpoints (quote-short, enterprise-values, …)
    fetch_history,        # quarterly / annual statements
    fetch_analyst_estimates,# forward EPS
)
from calc.ttm import (
    calc_ttm,
    quarterly_op_margin,
    build_earnings_yield_df,
    compute_roic_ttm,
    compute_fcf_yield_ttm,
    compute_ev_to_sales_ttm,
)
from metrics.value_quality import add_value_quality_metrics

from calc.boosts import (
    ebit_boost,
    revenue_boost,
    op_margin_boost,
    roic_boost,
    fcf_cagr_boost,
    earnings_yield_boost,
)

from calc.fallbacks import (
    annual_fallback,
    try_fallbacks,
)

from io import StringIO

import pandas as pd

# ── configuration ────────────────────────────────────────────────────────────
# fields we want to pull from income-statement endpoint
INCOME_FIELDS = [
    "ebit", 
    "grossProfit",
    "netIncome", 
    "ebitda", 
    "incomeTaxExpense", 
    "incomeBeforeTax",
    "weightedAverageShsOutDil",
    ]

# fields we want to pull from ratios endpoint
RATIO_FIELDS = [
    "priceToEarningsRatioTTM",
    "operatingProfitMarginTTM",
]

CASH_FLOW_FIELDS = [
    "operatingCashFlow",
]

# fields we want to pull from balance-sheet-statment endpoint 
BALANCE_SHEET_FIELDS = [
    "totalAssets",
    "totalStockholdersEquity",
    "shortTermDebt",
    "longTermDebt",
    "cashAndCashEquivalents"
]

# fields we want to pull from ratios-ttm endpoint
RATIO_TTM_FIELDS = [
    "priceToEarningsRatioTTM",
    "operatingProfitMarginTTM",
]

# fields we want to pull from key-metrics-ttm endpoint
KEY_METRIC_TTM_FIELDS = [
    "evToEBITDATTM",
    "freeCashFlowYieldTTM",
    "evToSalesTTM",
    "returnOnInvestedCapitalTTM",
]

# top‑level constants)
ZERO_INVALID_COLS = [
    "priceToEarningsRatioTTM", "evToEBITDATTM", "evToSalesTTM",
    "returnOnInvestedCapitalTTM", "freeCashFlowYieldTTM",
    "price", "enterpriseValue", "epsAvgTTM",
    # …add anything else where 0 means “missing”
]

def build_universe_df(tickers: list[str], universe: str) -> pd.DataFrame:

    # ➊ Refresh fundamentals cache (adds last_report_date, fetched_at)
    fund_df = refresh_if_needed(universe, tickers, build_row_fn=fetch_row)

    # ➋ Tidy: index by ticker & turn 0→pd.NA for known “zero means missing” cols
    fund_df = fund_df.set_index("ticker")
    invalid = [c for c in ZERO_INVALID_COLS if c in fund_df.columns]
    fund_df[invalid] = fund_df[invalid].replace(0, pd.NA)

    # ➌ Add static Value/Quality metrics (produces VALUE_SCORE_STATIC, QUALITY_SCORE_STATIC…)
    df = add_value_quality_metrics(fund_df)

    return df

def add_earnings_yield_boost(df: pd.DataFrame, price_mi: pd.DataFrame) -> pd.DataFrame:
    """
    Earnings‑yield trend boost: joins each ticker's cached quarterly EPS (eps_history)
    to its price history, then runs earnings_yield_boost. Needs prices, so it runs
    after update_universe() rather than inside the fundamentals fetch.
    """
    if "eps_history" not in df.columns:
        df["eps_history"] = None

    boosts = {}
    for tkr, raw in df["eps_history"].items():
        ey__quarterly_df = None
        try:
            ticker_price_df = price_mi.xs(tkr, level="ticker")  # gets just this ticker's data
            if not ticker_price_df.empty and isinstance(raw, str):
                ey__quarterly_df = build_earnings_yield_df(pd.read_json(StringIO(raw)), ticker_price_df)
        except KeyError:
            pass                                                # price history missing → boost stays None
        boosts[tkr] = earnings_yield_boost(ey__quarterly_df)

    df["earnings_yield_boost"] = pd.Series(boosts, dtype="float64")
    return df.drop(columns=["eps_history"])

def fetch_row(tkr: str) -> dict[str, float | None]:

    """
    Collect all metrics for *one* ticker and return a row-dict.
    """
    backups_used: list[str] = []  # fields that fell back to a backup value (counted in main.py)

    # Direct TTM pulls
    # --- key-metrics-ttm ------------------------------------------------
    k_m_ttm = fetch_ttm("key-metrics-ttm", tkr, KEY_METRIC_TTM_FIELDS) # "evToEBITDATTM", "freeCashFlowYieldTTM", "evToSalesTTM", "returnOnInvestedCapitalTTM"
    evToEBITDATTM = k_m_ttm.get("evToEBITDATTM") # get the primary evToEBITDATTM
    freeCashFlowYieldTTM = k_m_ttm.get("freeCashFlowYieldTTM") # get the primary freeCashFlowYieldTTM
    returnOnInvestedCapitalTTM = k_m_ttm.get("returnOnInvestedCapitalTTM") # get the primary returnOnInvestedCapitalTTM    

    #---ratios-ttm---
    r_ttm = fetch_ttm("ratios-ttm", tkr, RATIO_TTM_FIELDS) # "priceToEarningsRatioTTM", "operatingProfitMarginTTM"
    priceToEarningsRatioTTM  = r_ttm.get("priceToEarningsRatioTTM")
    operatingProfitMarginTTM = r_ttm.get("operatingProfitMarginTTM")
    
    # income-statement TTM calcs
    # Use the most recent 4 quarters of data from income_statment endpoint to calculate ebitTTM, grossProfitTTM, netIncomeTTM (backup), ebitdaTTM (backup)
    i_s = fetch_history("income-statement", tkr, "quarter", 12)
    i_s_ttm_calc = calc_ttm(i_s, INCOME_FIELDS)

    # cash-flow-statement TTM calcs
    # Use the most recent 4 quarters of data from ratios endpoint to calculate operatingCashFlowTTM
    c_f_s = fetch_history("cash-flow-statement", tkr, "quarter", 16)    
    c_f_s_ttm_calc = calc_ttm(c_f_s, CASH_FLOW_FIELDS)
    cf_annual = fetch_history("cash-flow-statement", tkr, "annual", 4)

    # balance-sheet pull shared by ROIC, EV-to-sales and total assets (newest row = row 0)
    bs_q = fetch_history("balance-sheet-statement", tkr, "quarter", 7)

    # --- price, forward EPS, enterprise value --------------------------------    
    # profile carries price too (matches quote-short to <0.03%), so one call covers price, sector and industry
    profile          = fetch_single("profile", tkr, ["sector", "industry", "price"])
    price            = profile.get("price")
    eps_avg          = fetch_analyst_estimates(tkr, ["epsAvg"]).get("epsAvg") # epsAvg works ✅, did 2025 in a list of future past
    # enterprise_value = fetch_single("enterprise-values", tkr, ["enterpriseValue"]).get("enterpriseValue") #enterpriseValue
    # total_assets     = fetch_single("balance-sheet-statement", tkr, ["totalAssets"]).get("totalAssets") # totalAssets
    enterprise_value = fetch_single("enterprise-values", tkr, ["enterpriseValue"], period="quarter", limit=1).get("enterpriseValue")
    total_assets = None
    if bs_q is not None and not bs_q.empty and pd.notna(bs_q.iloc[0].get("totalAssets")):
        total_assets = bs_q.iloc[0]["totalAssets"]

    # ── quarterly EPS for the earnings-yield boost (joined to prices in add_earnings_yield_boost) ──
    eps_keep = [c for c in ["date", "eps", "symbol", "period"] if i_s is not None and c in i_s.columns]
    eps_history = i_s[eps_keep].to_json(orient="records", date_format="iso") if eps_keep and not i_s.empty else None


    #-------------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------- Fallbacks------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------#
    # --------ratios-ttm fallbacks-----------------#
    # -----------trailing_earnings_yield fallback-------------------#
    
    # if that is still zero, calculate price/(netIncomeTTM/weightedAverageShsOutDil)
    if priceToEarningsRatioTTM in (None, 0):
        weightedAverageShsOutDil = None
        if i_s is not None and not i_s.empty and pd.notna(i_s.iloc[0].get("weightedAverageShsOutDil")):
            weightedAverageShsOutDil = i_s.iloc[0]["weightedAverageShsOutDil"] # weightedAverageShsOutDil
        netIncomeTTM             = i_s_ttm_calc.get("netIncome")           
                
        if all(isinstance(v, (int, float)) and v != 0 for v in (price, netIncomeTTM, weightedAverageShsOutDil)):
            try:
                eps_ttm_manual = netIncomeTTM / weightedAverageShsOutDil
                if eps_ttm_manual != 0:
                    priceToEarningsRatioTTM = price / eps_ttm_manual
                    backups_used.append("priceToEarningsRatioTTM")
            except ZeroDivisionError:
                pass

        # 2nd fallback: grab the single most‐recent annual P/E
        if priceToEarningsRatioTTM in (None, 0):
            df_ratios_annual = fetch_history("ratios", tkr, period="annual", limit=1)
            if (
                df_ratios_annual is not None 
                and not df_ratios_annual.empty 
                and "priceToEarningsRatio" in df_ratios_annual.columns
            ):
                pe = df_ratios_annual.iloc[0]["priceToEarningsRatio"]
                if isinstance(pe, (int, float)) and pe != 0:
                    priceToEarningsRatioTTM = pe
                    backups_used.append("priceToEarningsRatioTTM")

    # --------key-metrics-ttm fallbacks-----------------#
    # if any(k_m_ttm.get(field) in (None, 0) for field in KEY_METRIC_TTM_FIELDS): # if anything from key-metrics-ttm is missing
    # latest two annual rows, newest first
    k_m_annual = fetch_history("key-metrics", tkr, period="annual", limit=4) # we get everything from key-metrics

    # --- ev_ebitda fallback ---------------------------------------------------        
    if evToEBITDATTM in (None, 0) and enterprise_value not in (None, 0) and i_s_ttm_calc.get("ebitda") not in (None, 0): #if not there
        ebitdaTTM = i_s_ttm_calc.get("ebitda") # get from income-statement endpoint calc ttm
        evToEBITDATTM = enterprise_value/ebitdaTTM
        backups_used.append("evToEBITDATTM")

    if evToEBITDATTM in (None, 0) and k_m_annual is not None and not k_m_annual.empty:
        evToEBITDATTM  = annual_fallback(k_m_annual, "evToEBITDA", evToEBITDATTM)
        if evToEBITDATTM not in (None, 0):
            backups_used.append("evToEBITDATTM")
    # --- ROIC fallback -----------------------------------------------
    if returnOnInvestedCapitalTTM in (None, 0):
        returnOnInvestedCapitalTTM = compute_roic_ttm(i_s, bs_q) # special function for manual roic ttm calc
        if returnOnInvestedCapitalTTM not in (None, 0):
            backups_used.append("returnOnInvestedCapitalTTM")
    
    if returnOnInvestedCapitalTTM in (None, 0) and k_m_annual is not None and not k_m_annual.empty:
        returnOnInvestedCapitalTTM  = annual_fallback(k_m_annual, "returnOnInvestedCapital", returnOnInvestedCapitalTTM)
        if returnOnInvestedCapitalTTM not in (None, 0):
            backups_used.append("returnOnInvestedCapitalTTM")
    # --- freeCashFlowYieldTTM fallback -----------------------------------------------    
    if freeCashFlowYieldTTM in (None, 0):
        # get from special function for manual fcf yield calc
        freeCashFlowYieldTTM = compute_fcf_yield_ttm(c_f_s, i_s, price)
        if freeCashFlowYieldTTM not in (None, 0):
            backups_used.append("freeCashFlowYieldTTM")

    
    if freeCashFlowYieldTTM in (None, 0) and k_m_annual is not None and not k_m_annual.empty:
        freeCashFlowYieldTTM  = annual_fallback(k_m_annual, "freeCashFlowYield", freeCashFlowYieldTTM)
        if freeCashFlowYieldTTM not in (None, 0):
            backups_used.append("freeCashFlowYieldTTM")
    # --- evToSalesTTM  -----------------------------------------------
    evToSalesTTM = try_fallbacks(
        k_m_ttm.get("evToSalesTTM"),
        compute_ev_to_sales_ttm(i_s, bs_q, price),
        annual_fallback(k_m_annual, "evToSales", None) if k_m_annual is not None and not k_m_annual.empty else None,
    )
    if evToSalesTTM not in (None, 0) and k_m_ttm.get("evToSalesTTM") in (None, 0):
        backups_used.append("evToSalesTTM")

    #-----------operatingProfitMarginTTM fallback------------#
    
    if operatingProfitMarginTTM in (None, 0) and i_s is not None and not i_s.empty:
        last4 = i_s.sort_values("date", ascending=False).head(4)
        num = last4["operatingIncome"].sum()
        den = last4["revenue"].sum()
        if den not in (None, 0):
            operatingProfitMarginTTM = num / den
            backups_used.append("operatingProfitMarginTTM")
        
    o_p_m_quarter = quarterly_op_margin(i_s) # for op_margin_boost
    # ey__quarterly_df =  build_earnings_yield_df(i_s, tick_df)

    fcf_cagr = fcf_cagr_boost(cf_annual)
    # fallback to quarterly if annual missing
    if fcf_cagr is None:
        fcf_cagr = fcf_cagr_boost(c_f_s)
        if fcf_cagr is not None:
            backups_used.append("fcf_cagr_boost")

    # ── slope‑boost inputs ───────────────────────────────────────
    boosts = {
        "ebit_boost":        ebit_boost(i_s),
        "revenue_boost":     revenue_boost(i_s),
        "op_margin_boost":   op_margin_boost(o_p_m_quarter),
        "roic_boost":        roic_boost(k_m_annual),
        "fcf_cagr_boost":    fcf_cagr,
    }

    # ── consolidate key‑metrics fields that survived fallbacks ───────────────────
    key_metrics_row = {
        "evToEBITDATTM":             evToEBITDATTM,
        "freeCashFlowYieldTTM":      freeCashFlowYieldTTM,
        "evToSalesTTM":              evToSalesTTM,
        "returnOnInvestedCapitalTTM":returnOnInvestedCapitalTTM,
    }

    # ── build the final row dict -------------------------------------------------
    row: dict[str, float | None] = {
        "ticker":                         tkr,

        # ratios
        "priceToEarningsRatioTTM":        priceToEarningsRatioTTM,
        "operatingProfitMarginTTM":       operatingProfitMarginTTM,

        # key metrics
        **key_metrics_row,

        # market data / analyst
        "price":                          price,
        "epsAvgTTM":                      eps_avg,
        "enterpriseValue":                enterprise_value,

        # income‑statement TTM (from calc_ttm)
        "ebitTTM":                        i_s_ttm_calc.get("ebit"),
        "grossProfitTTM":                 i_s_ttm_calc.get("grossProfit"),
        "netIncomeTTM":                   i_s_ttm_calc.get("netIncome"),
        "weightedAverageShsOutDilTTM":    i_s_ttm_calc.get("weightedAverageShsOutDil"),

        # cash‑flow & balance‑sheet
        "operatingCashFlowTTM":           c_f_s_ttm_calc.get("operatingCashFlow"),
        "totalAssets":                    total_assets,
        
        
        # If profile didn’t return anything (None), guard against KeyError:
        "sector":                         profile.get("sector")   if isinstance(profile, dict) else None,
        "industry":                       profile.get("industry") if isinstance(profile, dict) else None,

        # all six boosts
        **boosts,

        # quarterly EPS, used later by add_earnings_yield_boost
        "eps_history":                    eps_history,

        # backup fields used for this ticker (summarised in main.py)
        "backups_used":                   ";".join(backups_used),
    }
    # print(row)
    # Stamp with Eastern timestamps
    eastern = pytz.timezone("US/Eastern")
    now_et = pd.Timestamp.now(tz=eastern)
    row["fetched_at"]       = now_et.isoformat(timespec="seconds")

    return row