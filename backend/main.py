# ── main ─────────────────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill
from core.prices import update_universe, _load_prices
from metrics.hqm import build_hqm
import os
from core.helpers import format_sci_4sf

from metrics.run_metrics import fetch_row, build_universe_df
from metrics.value_quality import add_value_quality_metrics
from calc.rank import add_ranks, FINAL_ORDER
from core.fundamentals_cache import load_cached
from core.paths import FUNDAMENTALS_DIR, OUTPUT_DIR
# from core.fundamentals_cache import refresh_if_needed
# choose your TTL here (days). Set to 10 for your requirement.
FUND_TTL_DAYS = 10
from core.excel_export import export_to_excel

try:
    from universes import (
        download_SP500,
        download_NASDAQ,
        download_NYSE,
        download_IWM,
        download_precious_metals,
    )
except ImportError as err:  # pragma: no-cover
    sys.exit(f"❌  universes.py missing or broken – {err}")

from datetime import datetime, timezone

# e.g. “20250606_143022” for YYYYMMDD_HHMMSS
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# HQM settings ---------------------------------------
rank_days = [3, 14, 35, 70]     # <- Modify HQM Return Windows here
screen_days = 70
def _parse_args() -> str:
    parser = argparse.ArgumentParser(description="Fetch TTM & valuation metrics")
    parser.add_argument(
        "--universe",
        choices=["sp500", "nasdaq", "nyse", "iwm", "precious_metals"],
        default="sp500",
        help="Ticker universe to pull (default: sp500)",
    )
    return parser.parse_args().universe

def _download_universe(name: str) -> list[str]:
    """Return a list of symbols for the chosen universe."""
    fn_map = {
        "sp500":         download_SP500,
        "nasdaq":        download_NASDAQ,
        "nyse":          download_NYSE,
        "iwm":           download_IWM,
        "precious_metals": download_precious_metals,
    }
    return fn_map[name]()

def main() -> None:  # pragma: no-cover
    universe = _parse_args()
    symbols = _download_universe(universe)
    # symbols  = symbols[65:105]

    print(f"📥  Universe '{universe}' – {len(symbols)} symbols")

    # ── ➊ Refresh price history for HQM ────────────────────────────────
    price_mi = update_universe(universe, symbols)  # MultiIndex frame
    price_mi = price_mi.drop(columns=["open", "high", "low"], errors="ignore")
    # price_mi = _load_prices(universe)

    # load or create fundamentals dataframe
    df_fundamantals = load_cached(universe)
    
    fundamentals_path = FUNDAMENTALS_DIR / f"{universe}_fundamentals.parquet"
    print(f"📂 Fundamentals cache path → {fundamentals_path}")

    # If cache is missing/corrupt, rebuild fundamentals now and continue.
    if df_fundamantals is None:
        print("🧰 No cached fundamentals. Rebuilding now…")
        # build_universe_df uses refresh_if_needed under the hood and saves the cache
        df_fundamantals = build_universe_df(symbols, universe, price_mi)
        # ensure index=ticker for the rest of main()
        if "ticker" in df_fundamantals.columns:
            df_fundamantals = df_fundamantals.set_index("ticker")
    # print("\nDataFrame structure:")
    # print(f"Index: {df.index.name}")
    # print(f"Columns: {df.columns.tolist()}")
    # print(f"Shape: {df.shape}")
    # print(df)

    # ── ➋ Compute liquidity map & merge HQM scores ─────────────────────
    shares_map = df_fundamantals["weightedAverageShsOutDilTTM"].to_dict()       # dilution from run_metrics

    # ⛑️ Ensure all df.index tickers exist in price_mi before slicing
    tickers_in_prices = price_mi.index.get_level_values("ticker").unique()
    valid_tickers = df_fundamantals.index.intersection(tickers_in_prices)
    missing_tickers = df_fundamantals.index.difference(tickers_in_prices)

    if not missing_tickers.empty:
        print(f"⚠️ Warning: {len(missing_tickers)} tickers missing price data: {list(missing_tickers[:5])}...")

    # Safely slice only valid tickers
    price_mi = price_mi.loc[pd.IndexSlice[:, valid_tickers], :]
    
    hqm_df = build_hqm(
        price_mi,
        shares_map,
        rank_days=rank_days,
        screen_days=screen_days,
        top_n=int(len(symbols)),    # score the full universe with len(symbols)
        min_dollar_vol=1e5       # no liquidity filter for now
    )                   # returns ticker → [HQM_SCORE, DOLLAR_VOLUME, TURNOVER_RATIO]
    
    df = df_fundamantals.merge(hqm_df.set_index("ticker"), 
                            left_index=True, right_index=True, how="left")

    # print("\nPost-HQM merge columns:")
    # print(df.columns.tolist())
    # ✅ This is needed before ranking
     
    df = add_value_quality_metrics(df)
    df = add_ranks(df)  # adds VALUE_SCORE, QUALITY_SCORE, etc.

    rename_map = {f"{d}D_pct": f"{d}-day %" for d in rank_days}
    df.rename(columns=rename_map, inplace=True)    

    pct_HQM_cols = list(rename_map.values())      

    print("\n🧪 Sector check:")
    for sec, group in df.groupby("sector"):
        avg_val = group["VALUE_SCORE"].mean()
        avg_qual = group["QUALITY_SCORE"].mean()
        avg_robust = group["ROBUST_SCORE"].mean()
        print(f"{sec:<25} → V:{avg_val:.3f}, Q:{avg_qual:.3f}, R:{avg_robust:.3f}")

    # rank each ticker _within its sector_ by HQM Score
    df["sector_rank"] = (
        df.groupby("sector")["HQM_SCORE"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    # ─── Export full metrics CSV ──────────────────────────────────────────────

    # 2A) Build the full-width output frame
    full_cols = ["tkr", "sector", "sector_rank", "industry"] + pct_HQM_cols + FINAL_ORDER

    df_full = (
        df.sort_values("ROBUST_SCORE", ascending=False)
        .reset_index(names="tkr")
        .loc[:, full_cols]
    )

    # 1) Round numeric values
    # Round all float columns EXCEPT TURNOVER_RATIO to 2 decimal places
    for col in df.select_dtypes(include='float'):
        if col == "TURNOVER_RATIO":
            df[col] = df[col].round(6)  # or however many places you want
        else:
            df[col] = df[col].round(2)

    # 2) Export to Excel with styling
    out_dir = OUTPUT_DIR / universe
    excel_path = out_dir / f"{universe}_metrics_{ts}.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_excel(df_full, excel_path)

    # Ensure we keep only columns that actually exist (some tickers may miss data)
    # ─ Focused coverage report (only metrics in FINAL_ORDER) ──────────────────
    final_cols = [c for c in FINAL_ORDER if c in df.columns]

    # 1) Percent non‑null for just those columns
    cov_final = (1 - df[final_cols].isna().mean()).loc[final_cols]

    print("\nData coverage for final metrics (ordered):")
    print(cov_final.to_string(float_format="{:.1%}".format))

    # 2) Which tickers are missing each metric (<100 % coverage)?
    missing_info = {
        col: df.index[df[col].isna()].tolist()
        for col, pct in cov_final.items()
        if pct < 1.0
    }

    if missing_info:
        print("\nFields with missing data:")
        for col, tkrs in missing_info.items():
            pct = cov_final[col]
            print(f"  • {col} ({pct:.1%} coverage) → missing: {tkrs}")
    else:
        print("\n✅ All final‑metric columns are 100 % complete among selected tickers.")

    # Define minimal output columns in desired order
    # ---- build minimal preview ----
    final_cols = ["ROBUST_SCORE", "HQM_SCORE", "DOLLAR_VOLUME", "TURNOVER_RATIO", "price"]
    preview_cols = ["sector", "sector_rank"] + final_cols

    out = (
        df.sort_values("HQM_SCORE", ascending=False)
        .reset_index(names="tkr")
        .loc[:, ["tkr"] + preview_cols]
        .head(20)
        .reset_index(drop=True)
    )
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    print("\nTop‑20 by ROBUST_SCORE:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(out.to_markdown(index=False))

    # ─ Save to disk for downstream screener if you like ───────────────────────
    outpath = OUTPUT_DIR / universe / "metrics_raw.parquet"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outpath)
    print(f"\n✅  Saved raw metrics → {outpath}")

if __name__ == "__main__":
    main()