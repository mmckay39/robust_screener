# metrics/hqm.py
"""
HQM (High‑Quality Momentum) score + liquidity stats.

• Uses average $‑volume and turnover filter
• NOT folded into ROBUST_SCORE
"""
import numpy as np, pandas as pd
from scipy import stats
from typing import Iterable
import os

# ---------------- helpers ----------------
def _return(series: pd.Series, days: int) -> float | None:
    try:
        return float(series.pct_change(days).iloc[-1] * 100)
    except Exception:
        return None

def _avg_dollar_vol(df: pd.DataFrame, days: int) -> float:
    tail = df.tail(days)
    #take the last price and multiply it by the average volume over the largest timeframe the user chose.
    return float((tail["close"] * tail["volume"]).mean()) 

def _turnover(df: pd.DataFrame, shares_out: float, days: int) -> float:
    tail = df.tail(days)
    return float(tail["volume"].mean() / shares_out) if shares_out else np.nan

# ----------- public API ------------------
"""Recommended default:
min_dollar_vol
- For NYSE (larger, more liquid stocks): $100,000 is reasonable for screening
- For NASDAQ (includes many small caps, but also highly liquid large caps): $50,000 - $100,000 depending on universe size
- If you want to include all stocks (including illiquid ones):
        - Set min_dollar_vol=0 or a very low number (e.g., 1,000)
        - Be aware this will include many very thinly traded stocks and increase noise.

top_n
Purpose: Limits the number of tickers for which to calculate percentile ranks and final HQM score to speed up processing and focus on investable stocks.
Recommended default:
- 2000 is a good upper bound to cover practically all investable stocks in NASDAQ/NYSE while maintaining reasonable compute time.
- You can increase this if you want every ticker scored but expect slower runs.
- If you want all companies, including worst:
    - Set top_n to something >=universe size to allow computation of HQM percentiles over the entire universe
"""

# ── metrics/hqm.py ───────────────────────────────────────────────────────────
def build_hqm(
    price_mi: pd.DataFrame,
    shares_out_map: dict[str, float],
    *,
    rank_days: Iterable[int] = (5, 10, 21, 42),
    screen_days: int | None = 42,
    top_n: int = 2_000,
    min_dollar_vol: int = 0,
) -> pd.DataFrame:
    """
    Return ticker, HQM_SCORE, DOLLAR_VOLUME, TURNOVER_RATIO.

    • Keeps rows even when TURNOVER_RATIO is NaN so HQM coverage stays high
"""    
    # ── 0️⃣  Sanity check for price history ───────────────────────────
    if price_mi.empty:
        raise ValueError("[HQM] price_mi is empty – did update_universe() fail?")

    # ── 1️⃣  Optional performance screen on last `screen_days` return ──
    tickers = price_mi.index.get_level_values("ticker").unique()
    if screen_days:
        rets = (
            price_mi
            .groupby("ticker")["close"]
            .apply(lambda s: _return(s, screen_days))
            .dropna()
        )
        tickers = rets.sort_values(ascending=False).head(top_n).index

    rows: list[dict] = []
    for t in tickers:
        frame = price_mi.xs(t, level="ticker")

        dv = _avg_dollar_vol(frame, screen_days or max(rank_days))
        if dv < min_dollar_vol:
            continue

        turns = _turnover(frame,
                          shares_out_map.get(t, np.nan),
                          screen_days or max(rank_days))

        row = {"ticker": t, "DOLLAR_VOLUME": dv, "TURNOVER_RATIO": turns}

        for d in rank_days:
            row[f"{d}D"] = _return(frame["close"], d)
        rows.append(row)

    hqm = pd.DataFrame(rows).dropna(subset=[f"{d}D" for d in rank_days])
    if hqm.empty:
        return pd.DataFrame(columns=["ticker", "HQM_SCORE",
                                     "DOLLAR_VOLUME", "TURNOVER_RATIO"])

    # ── 2️⃣  Percentiles & composite score ────────────────────────────
    pct_cols = []
    for d in rank_days:
        raw, pct = f"{d}D", f"{d}D_pct"
        pct_cols.append(pct)
        hqm[pct] = hqm[raw].rank(pct=True)

    hqm["HQM_SCORE"] = hqm[pct_cols].mean(axis=1)
    # ── include the individual window percentiles ───────────────────────
    # e.g. 3D_pct, 20D_pct, …

    return hqm[["ticker"] + pct_cols + ["HQM_SCORE", "DOLLAR_VOLUME", "TURNOVER_RATIO"]]



