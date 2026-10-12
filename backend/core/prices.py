# core/prices.py
from pathlib import Path
from datetime import time as dtime
from typing import List
import os, pandas as pd, pytz, requests
from tqdm import tqdm
import time  # ✅ this one gives you perf_counter()

from core.cache import cached_get_json   # already exist
from core.paths import PRICES_DIR
from dotenv import load_dotenv
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY")

OUT_DIR = PRICES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
def save_prices(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Persist OHLCV parquet/csv and return MultiIndex (date,ticker)."""
    pq_path  = OUT_DIR / f"{universe}_ohlcv.parquet"
    csv_path = OUT_DIR / f"{universe}_ohlcv.csv"

    df = df.copy()
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    df.to_parquet(pq_path, compression="snappy", engine="pyarrow", index=False)
    df.to_csv(csv_path, index=False)
    print(f"✔️  Saved {len(df):,} rows → {pq_path}")
    return df.set_index(["date", "ticker"])

def _load_prices(universe: str) -> pd.DataFrame:
    path = OUT_DIR / f"{universe}_ohlcv.parquet"
    csv_path = OUT_DIR / f"{universe}_ohlcv.csv"

    def _empty():
        return pd.DataFrame(columns=["date","ticker","open","high","low","close","volume"])

    if not path.exists():
        # try CSV cache if present
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
                return df
            except Exception as e:
                print(f"⚠️ CSV read failed: {e}")
        return _empty()

    try:
        # IMPORTANT: pin engine to pyarrow (consistent with writer)
        df = pd.read_parquet(path, engine="pyarrow")
        # normalize index shape
        return df.reset_index() if isinstance(df.index, pd.MultiIndex) else df

    except (OSError, ValueError) as e:
        # Typical: 'Repetition level histogram size mismatch' from pyarrow
        print(f"⚠️ Parquet read failed ({e}). Quarantining and rebuilding …")
        bad = path.with_suffix(".parquet.bad")
        try:
            path.rename(bad)
            print(f"   → moved corrupt file to {bad.name}")
        except Exception:
            # if rename fails (e.g., locked), just delete
            try: path.unlink()
            except Exception: pass

        # CSV fallback if available
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
                return df
            except Exception as e2:
                print(f"⚠️ CSV fallback failed: {e2}")

        return _empty()

def _end_of_history_cutoff() -> pd.Timestamp:
    eastern = pytz.timezone("US/Eastern")
    now_et = pd.Timestamp.now(tz=eastern)
    is_businessday = now_et.dayofweek < 5
    after_close  = now_et.time() >= dtime(16, 30)

    if is_businessday and after_close:
        return now_et.normalize().tz_localize(None)  # today
    else:
        # move to previous business day
        prev_day = now_et - pd.Timedelta(days=1)
        while prev_day.weekday() >= 5:  # skip weekends
            prev_day -= pd.Timedelta(days=1)
        return prev_day.normalize().tz_localize(None)


def _fetch_missing_days(symbol: str, start: str, end: str) -> list[dict]:
    url = (
        "https://financialmodelingprep.com/stable/historical-price-eod/full"
        f"?symbol={symbol}&from={start}&to={end}&apikey={FMP_API_KEY}"
    )
    # print(f"Fetching prices for {symbol} from {start} to {end}")  # DEBUG
    bars = cached_get_json(url)
    if not bars:
            print(f"⚠️ No price data returned for {symbol}")  # DEBUG
    return [
        {"date": bar["date"], "ticker": symbol,
        "close": bar["close"], "volume": bar["volume"]}
        for bar in bars if start <= bar["date"] <= end
    ]


def update_universe(universe: str, symbols: list[str]) -> pd.DataFrame:
    """
    • Ensures OHLC data are current through _end_of_history_cutoff().
    • ALSO guarantees that every ticker in *symbols* exists in the parquet
      with the same date coverage. Missing or incomplete tickers are back-filled.
    • Never deletes rows for tickers that aren’t in *symbols*.
    """
    df_old = _load_prices(universe)                       # may be empty
    has_history = not df_old.empty

    end_date   = _end_of_history_cutoff()
    start_date = (
        df_old["date"].min().normalize()
        if has_history else
        pd.Timestamp("2022-01-01")                        # or another default
    )

    # ---------- 1️⃣ Date-based gap ----------
    if has_history:
        last_date = df_old["date"].max().normalize()
        next_date = last_date + pd.Timedelta(days=1)
        date_gaps = pd.bdate_range(next_date, end_date)
    else:
        date_gaps = pd.bdate_range(start_date, end_date)

    # ---------- 2️⃣ Symbol-based gap ----------
    if has_history:
        have_tickers = set(
            str(t).strip().upper() 
            for t in df_old["ticker"].dropna().unique()
        )
    else:
        have_tickers = set()

    # normalize all symbols as strings, uppercase, strip spaces
    symbols_norm = {str(s).strip().upper() for s in symbols if pd.notna(s)}
    missing_tickers = sorted(symbols_norm - have_tickers)

    needs_fetch = not date_gaps.empty or bool(missing_tickers)

    if not needs_fetch:
        print(f"✅ {universe} OHLC up-to-date through {end_date.date()} "
              f"({len(have_tickers)} tickers)")
        return df_old.set_index(["date", "ticker"])

    # -------------------------------------------------------------------
    new_rows: list[dict] = []

    # 2a. completely new tickers → full history
    loop = tqdm(missing_tickers, desc=f"🔄 Fetching new symbols", unit="tick")
    for sym in loop:
        start = time.perf_counter()
        new_rows += _fetch_missing_days(
            sym,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        elapsed = time.perf_counter() - start
        loop.set_postfix_str(f"{sym} ({elapsed:.2f}s)")

    # 2b. old tickers with date holes → recent gap only
    if not date_gaps.empty:
        print(
            f"🔄 {universe}: fetching {len(date_gaps)} missing days "
            f"{date_gaps[0].date()}→{date_gaps[-1].date()}"
        )

        loop = tqdm(symbols_norm, desc="🔄 Filling date gaps", unit="tick")
        for sym in loop:
            if sym in missing_tickers:
                continue

            start = time.perf_counter()
            try:
                new_rows += _fetch_missing_days(
                    sym,
                    date_gaps[0].strftime("%Y-%m-%d"),
                    date_gaps[-1].strftime("%Y-%m-%d")
                )
            except IndexError:
                print(f"⚠️ Skipped {sym} — date_gaps unexpectedly empty.")
            elapsed = time.perf_counter() - start
            loop.set_postfix_str(f"{sym} ({elapsed:.2f}s)")
    else:
        print(f"✅ No date gaps detected for {universe}.")

    # ---------- Merge & write ----------
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_new["date"] = pd.to_datetime(df_new["date"]).dt.tz_localize(None)
        df_all = (
            pd.concat([df_old, df_new], ignore_index=True)
              .drop_duplicates(subset=["date", "ticker"])
              .sort_values(["date", "ticker"])
              .reset_index(drop=True)
        )
    else:
        df_all = df_old

    out_path = OUT_DIR / f"{universe}_ohlcv"
    df_all.set_index(["date", "ticker"]).to_parquet(
        out_path.with_suffix(".parquet"),
        engine="pyarrow",
        compression="snappy"
    )
    df_all.to_csv(out_path.with_suffix(".csv"), index=False)

    print(f"✅ {universe}: added {len(new_rows)} rows; "
          f"{len(df_all['ticker'].unique())} tickers total")
    return df_all.set_index(["date", "ticker"])
