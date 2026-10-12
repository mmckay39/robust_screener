from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # loads from .env file in project root
import os
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise EnvironmentError("❌ Missing FMP_API_KEY environment variable")

from core.cache import cached_get_json
from datetime import datetime, timezone
from typing import Iterable, Mapping, Literal

import pandas as pd

FMP_BASE_stable = "https://financialmodelingprep.com/stable"

FMP_DOMAINS = [FMP_BASE_stable]  # v3 is retired by FMP (returns [] for every endpoint)
# data = cached_get_json(url, force_refresh=True) #force a refresh on the data

#-----------------------------------------------------------------------------------------------------------------#
# Statements only change when a new filing lands (quarterly), so they can be cached for a week.
# Price-dependent endpoints (key-metrics, ratios, enterprise-values, quote) keep the 24h default.
STATEMENT_TTL = 7 * 24 * 3600
STATEMENT_ENDPOINTS = {"income-statement", "balance-sheet-statement", "cash-flow-statement"}

def _fetch_first_list(path: str, ttl: int | None = None) -> list | None:
    """
    GET `path` (e.g. "income-statement?symbol=AAPL&apikey=…") from each FMP domain
    in turn and return the first non-empty list, or None if every domain came back empty.
    `ttl` (seconds) overrides the cache default when given.
    """
    for base in FMP_DOMAINS:
        data = cached_get_json(f"{base}/{path}", **({"ttl": ttl} if ttl else {}))
        if isinstance(data, list) and data:
            return data
    return None

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
    """
    result: dict[str, float | None] = {f: None for f in fields}

    data = _fetch_first_list(
        f"analyst-estimates?symbol={ticker}&period={period}&apikey={api_key or FMP_API_KEY}"
    )
    if data is None:
        return result                             # nothing from FMP

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
    data = _fetch_first_list(
        f"{endpoint}?symbol={ticker}&period={period}&limit={limit}&apikey={FMP_API_KEY}",
        ttl=STATEMENT_TTL if endpoint in STATEMENT_ENDPOINTS else None,
    )
    if data is None:
        return None

    df = pd.DataFrame(data)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        # ── sort by date descending so row 0 is always newest
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

    return df

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

    data = _fetch_first_list(f"{endpoint}?symbol={ticker}&apikey={FMP_API_KEY}")

    # Ensure predictable output on error / empty response
    if not data:
        requested = list(fields) if fields is not None else []
        return {field: None for field in requested}

    latest: Mapping = data[0]                # FMP returns newest object first

    # No filter → return full object
    if fields is None:
        return dict(latest)

    return {field: latest.get(field) for field in fields}

#-----------------------------------------------------------------------------------------------------------------#
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
    data = _fetch_first_list(
        f"{endpoint}?symbol={ticker}&period={period}&limit={limit}&apikey={api_key or FMP_API_KEY}"
    )

    # If nothing came back, return None for each requested field
    if not data:
        return {field: None for field in (fields or [])}

    latest = data[0]
    if fields is None:
        return dict(latest)
    return {field: latest.get(field) for field in fields}
