from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # loads from .env file in project root
import os
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise EnvironmentError("❌ Missing FMP_API_KEY environment variable")

from core.cache import cached_get_json
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Literal, List

import pandas as pd

FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_BASE_stable = "https://financialmodelingprep.com/stable" 

FMP_DOMAINS = [
    FMP_BASE, 
    FMP_BASE_stable,   # fallback
]
# data = cached_get_json(url, force_refresh=True) #force a refresh on the data

#-----------------------------------------------------------------------------------------------------------------#
# Forward‑looking Price to Earnings (PE) helper
def fetch_analyst_estimates(
    ticker: str,
    fields: list[str],
    *,
    period: str = "annual", # all i have access to is annual
    api_key: str | None = None,
) -> dict[str, float | None]:
    """
    Return analyst‑estimate fields (e.g. epsAvg) for the most relevant fiscal
    year:
        • Jan‑Aug ➜ current calendar year
        • Sep‑Dec ➜ next calendar year

    Tries /api/v3 first (where FMP now hosts analyst data), then /stable.
    """
    result: dict[str, float | None] = {f: None for f in fields}

    # Try v3 first, then stable
    for base in (FMP_BASE, FMP_BASE_stable):          # note the order
        url = (
            f"{base}/analyst-estimates"
            f"?symbol={ticker}&period={period}"
            f"&apikey={api_key or FMP_API_KEY}"
        )
        data = cached_get_json(url)
        if isinstance(data, list) and data:
            break
    else:
        return result                             # nothing from either domain

    # Choose which fiscal year’s estimate to use
    today = datetime.now(timezone.utc)
    target_year = today.year if today.month < 9 else today.year + 1

    entry = next(
        (row for row in data if str(target_year) in row.get("date", "")),
        data[0],                                  # fallback to first row
    )

    for field in fields:
        result[field] = entry.get(field)

    return result
# print(fetch_analyst_estimates('PLTR', ['epsAvg']))

#-----------------------------------------------------------------------------------------------------------------#
def fetch_history(
    endpoint: Literal[
        "income-statement",
        "balance-sheet-statement",
        "cash-flow-statement",
        "ratios",
        "key-metrics",
    ],
    ticker: str,
    period: Literal["quarter", "annual"] = "quarter",
    limit: int = 8,
    api_key: str | None = None,
    try_fallback: bool = True,
) -> pd.DataFrame | None:
    """
    Generalised pull for any FMP endpoint.

    Returns a DataFrame sorted by 'date' descending (newest first).
    """
    domains = FMP_DOMAINS if try_fallback else FMP_DOMAINS[:1]

    for base in domains:
        url = (
            f"{base}/{endpoint}"
            f"?symbol={ticker}&period={period}&limit={limit}"
            f"&apikey={FMP_API_KEY}"
        )
        
        data = cached_get_json(url)

        if isinstance(data, list) and data:
            df = pd.DataFrame(data)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
                # ── sort by date descending so row 0 is always newest
                df = df.sort_values("date", ascending=False).reset_index(drop=True)

            return df

    return None

#-----------------------------------------------------------------------------------------------------------------#
# The ttm endpoints need their own function, i guess, since we are not specifying a quarter or annual. The url is simpler.
def fetch_ttm(
    endpoint: str,
    ticker: str,
    fields: Iterable[str] | None = None,
    api_key: str | None = None,
) -> dict[str, float | None]:
    """
    Universal fetcher for any FMP *‑ttm endpoint* (ratios‑ttm, key‑metrics‑ttm …).

    Parameters
    ----------
    endpoint : str
        The path without leading slash, e.g. `"ratios-ttm"` or `"key-metrics-ttm"`.
    ticker   : str
    fields   : iterable[str] | None
        • Pass explicit field names to filter.  
        • Pass None to return the entire latest object.
    api_key  : str | None
        If None, get_json() should inject the default key or read from env.

    Returns
    -------
    dict[field, float | None]
        { "priceToEarningsRatioTTM": 42.3, "evToEBITDATTM": None, … }
    """

    data = []
    for base in FMP_DOMAINS:
        url = f"{base}/{endpoint}?symbol={ticker}&apikey={FMP_API_KEY}"
        data = cached_get_json(url)
        if isinstance(data, list) and data:
            break

    # Ensure predictable output on error / empty response
    if not isinstance(data, list) or not data:
        requested = list(fields) if fields is not None else []
        return {field: None for field in requested}

    latest: Mapping = data[0]                # FMP returns newest object first

    # No filter → return full object
    if fields is None:
        return dict(latest)

    return {field: latest.get(field) for field in fields}

#-----------------------------------------------------------------------------------------------------------------#
from typing import Iterable, Literal
from core.cache import cached_get_json

def fetch_single(
    endpoint: str,
    ticker: str,
    fields: Iterable[str] | None = None,
    *,
    period: Literal["quarter", "annual"] = "quarter",
    limit: int = 1,
    api_key: str | None = None,
) -> dict[str, float | None]:
    """
    Universal helper for endpoints that return a single latest object.
    You can specify:
      • period = "quarter" or "annual"  (if supported by that endpoint)
      • limit  = how many rows to fetch
    
    Example:
        fetch_single("enterprise-values", "AAPL", ["enterpriseValue"],
                     period="quarter", limit=1)
    """
    data = []
    for base in FMP_DOMAINS:
        url = (
            f"{base}/{endpoint}"
            f"?symbol={ticker}"
            f"&period={period}"
            f"&limit={limit}"
            f"&apikey={api_key or FMP_API_KEY}"
        )
        data = cached_get_json(url)
        if isinstance(data, list) and data:
            break

    # If nothing came back, return None for each requested field
    if not isinstance(data, list) or not data:
        return {field: None for field in (fields or [])}

    latest = data[0]
    if fields is None:
        return dict(latest)
    return {field: latest.get(field) for field in fields}


def fetch_prices(symbols: List[str], start: str, end: str) -> pd.DataFrame:

    """pull daily bars for each symbol via individual calls.
    Returns a tidy DataFrame with date, ticker, close, volume.
    """

    records = []
    for idx, sym in enumerate(symbols, 1):
        bars = None
        for base in FMP_DOMAINS:  # try /api/v3 then /stable
            url = (
                f"{base}/historical-price-eod/full"
                f"?symbol={sym}&from={start}&to={end}&apikey={FMP_API_KEY}"
            )
            resp = cached_get_json(url)
            if isinstance(resp, list) and resp:  # success
                bars = resp
                break

        if not bars:
            print(f"⚠️  {sym}: no data returned")
            continue

        for bar in bars:
            records.append(
                {
                    "date":   bar["date"],
                    "ticker": sym,
                    "close":  bar["close"],
                    "volume": bar["volume"],
                }
            )

        # if idx % 25 == 0 or idx == len(symbols):
        #     print(f"Fetched {idx}/{len(symbols)} symbols …")

    df = pd.DataFrame(records)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df