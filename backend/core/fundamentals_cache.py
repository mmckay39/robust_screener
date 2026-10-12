# core/fundamentals_cache.py
import os, pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()
import pytz
import time
FMP_API_KEY = os.getenv("FMP_API_KEY")

from core.paths import FUNDAMENTALS_DIR
FUND_DIR = FUNDAMENTALS_DIR
FUND_DIR.mkdir(parents=True, exist_ok=True)

# ---------- cache API --------------
def load_cached(universe: str) -> pd.DataFrame | None:
    """Raw cache table with 'ticker' as a column (refresh_if_needed relies on this)."""
    path = FUND_DIR / f"{universe}_fundamentals.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "ticker" not in df.columns:
        print(f"⚠️ 'ticker' not found in {path.name}; file may be corrupt.")
        return None
    return df

def save_cached(df: pd.DataFrame, universe: str):
    path = FUND_DIR / f"{universe}_fundamentals.parquet"
    df["fetched_at"] = df["fetched_at"].astype(str)  # force string dtype
    df.to_parquet(path, index=False, compression="snappy")

TTL_DAYS = 10            # ✏️  bump to whatever you like

def refresh_if_needed(
    universe: str,
    symbols: list[str],
    build_row_fn,  # build_row_fn(ticker: str) -> dict
    *,
    ttl_days: int = TTL_DAYS
) -> pd.DataFrame:
    eastern = pytz.timezone("US/Eastern")
    now_et = pd.Timestamp.now(tz=eastern)

    cached = load_cached(universe)
    if cached is None:
        # no cache at all → seed everything
        cached = pd.DataFrame(columns=["ticker", "fetched_at"])

    cutoff = (now_et - pd.Timedelta(days=ttl_days)).tz_localize(None)
    stale: list[str] = []

    for tkr in symbols:
        row = cached.loc[cached["ticker"] == tkr]
        # 1) never seen → stale
        if row.empty:
            stale.append(tkr)
            continue

        # 2) no fetched_at column or empty value → stale
        if "fetched_at" not in row.columns or pd.isna(row.iloc[0].get("fetched_at")):
            stale.append(tkr)
            continue

        # 3) parse and compare as before
        fetched_str = row.iloc[0]["fetched_at"]
        try:
            fetched_at = pd.to_datetime(fetched_str)
        except Exception:
            stale.append(tkr)
            continue

        # normalize to ET-naive for comparison
        if fetched_at.tzinfo is None:
            fetched_at = eastern.localize(fetched_at)
        else:
            fetched_at = fetched_at.tz_convert(eastern)
        
        fetched_naive = fetched_at.tz_localize(None)

        if fetched_naive < cutoff:
            stale.append(tkr)

    # ── (re‑)fetch stale tickers ──────────────────────────────────────
    new_rows: list[dict] = []
    # print("🔄 Fetching fundamental data")
    loop = tqdm(stale, desc="🔍 Fetching fundamental data", unit="tick")
    for tkr in loop:
        try:
            start = time.perf_counter()
            data = build_row_fn(tkr)
            elapsed = time.perf_counter() - start

            now_et = pd.Timestamp.now(tz=eastern)
            data["fetched_at"]       = now_et.isoformat(timespec="seconds")
            data["last_report_date"] = now_et.date().isoformat()
            new_rows.append(data)

            loop.set_postfix_str(f"{tkr} ({elapsed:.2f}s)")
        except Exception as exc:
            loop.set_postfix_str(f"{tkr} ❌")
            print(f"⚠️  fundamentals fetch failed for {tkr}: {exc}")

    if new_rows:
        fresh_df = pd.DataFrame(new_rows)
        # drop old copies of stale then append fresh
        cached = cached[~cached["ticker"].isin(stale)]
        cached = pd.concat([cached, fresh_df], ignore_index=True)

    # drop tickers no longer in the universe, then persist
    cached = cached[cached["ticker"].isin(symbols)].copy()
    save_cached(cached, universe)
    return cached

