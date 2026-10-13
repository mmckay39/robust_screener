# ── main ─────────────────────────────────────────────────────────────────────
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from core.prices import update_universe
from metrics.hqm import build_hqm

from metrics.run_metrics import build_universe_df, add_earnings_yield_boost
from metrics.value_quality import add_value_quality_metrics
from calc.rank import add_ranks, FINAL_ORDER
from core.paths import OUTPUT_DIR
from core.excel_export import export_to_excel, SCI_COLS

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

from datetime import datetime

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

    # ── ➊ + ➋ Prices (HQM) and fundamentals run side by side ─────────────
    # Both are network-bound and share one FMP rate limiter (core/cache.py),
    # so they overlap while each waits on the API.
    with ThreadPoolExecutor(max_workers=1) as pool:
        fund_future = pool.submit(build_universe_df, symbols, universe)
        price_mi = update_universe(universe, symbols)  # MultiIndex frame
        df_fundamantals = fund_future.result()

    price_mi = price_mi.drop(columns=["open", "high", "low"], errors="ignore")
    # price_mi = _load_prices(universe)

    # earnings-yield boost needs both the EPS history (fundamentals) and prices
    df_fundamantals = add_earnings_yield_boost(df_fundamantals, price_mi)
    missing_fund = [s for s in symbols if s not in df_fundamantals.index]
    # ensure index=ticker for the rest of main()
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

    # 1) Round the Excel copy (the parquet below keeps full precision)
    #    percent returns & price → 2 dp; percentile ranks, scores, yields, boosts → 4 dp;
    #    SCI_COLS stay unrounded and are shown in scientific notation by the export instead
    two_dp = [*pct_HQM_cols, "price"]
    for col in df_full.select_dtypes(include="float").columns:
        if col in SCI_COLS:
            continue
        df_full[col] = df_full[col].round(2 if col in two_dp else 4)

    # 2) Export to Excel with styling
    out_dir = OUTPUT_DIR / universe
    excel_path = out_dir / f"{universe}_metrics_{ts}.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_excel(df_full, excel_path)

    # ---- terminal preview: top 20 by HQM Score ----
    preview_cols = ["sector", "sector_rank", "HQM_SCORE", "ROBUST_SCORE", "DOLLAR_VOLUME", "TURNOVER_RATIO", "price"]
    out = (
        df.sort_values("HQM_SCORE", ascending=False)
        .reset_index(names="tkr")
        .loc[:, ["tkr"] + preview_cols]
        .head(20)
        .reset_index(drop=True)
    )
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["DOLLAR_VOLUME"] = out["DOLLAR_VOLUME"].map(lambda v: f"{v:.3e}")
    out["TURNOVER_RATIO"] = out["TURNOVER_RATIO"].map(lambda v: f"{v:.3e}")
    out[["HQM_SCORE", "ROBUST_SCORE"]] = out[["HQM_SCORE", "ROBUST_SCORE"]].round(4)

    print("\nTop‑20 by HQM_SCORE:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(out.to_markdown(index=False))

    # ─ Save to disk for downstream screener if you like ───────────────────────
    outpath = OUTPUT_DIR / universe / "metrics_raw.parquet"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outpath)
    print(f"\n✅  Saved raw metrics → {outpath}")

    # ─ End-of-run summary ─────────────────────────────────────────────────
    print(f"\nData gaps: {len(missing_tickers)} without price history · {len(missing_fund)} without fundamentals")
    backups = (
        df_fundamantals.get("backups_used", pd.Series(dtype=object))
        .fillna("").astype(str)
        .str.split(";").explode()
    )
    backups = backups[backups != ""].value_counts()
    print("\nBackups used (times per field):")
    if backups.empty:
        print("  none")
    for field, n in backups.items():
        print(f"  {field:<32} {n}")

if __name__ == "__main__":
    main()