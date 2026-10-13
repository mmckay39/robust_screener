# Robust Screener

A stock screener that ranks each ticker inside its sector on value, quality, growth and momentum, then exports the results to Excel. Data comes from the [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs) (FMP) `/stable` API.

## What it does

For a chosen universe, the screener:

1. Downloads the ticker list (S&P 500, NASDAQ, NYSE, IWM holdings, or precious-metals companies).
2. Fetches daily OHLCV price history and fills any missing days.
3. Fetches fundamentals (income statement, balance sheet, cash flow, key metrics, ratios, analyst estimates) and builds trailing-twelve-month (TTM) figures.
4. Computes an HQM (High-Quality Momentum) score from price returns over several windows, plus liquidity stats.
5. Ranks every ticker by percentile within its sector and combines the ranks into pillar scores and a final `ROBUST_SCORE`.
6. Writes a formatted Excel workbook and a raw parquet file.

## Scoring

| Pillar | Built from |
|---|---|
| **Value** | Earnings yield (trailing and forward), EV/EBIT yield, EV/EBITDA yield, FCF yield, sales yield |
| **Quality** | ROIC, gross profitability (GPA), accruals ratio, operating margin |
| **Growth** | EBIT growth, revenue CAGR, FCF CAGR (trend boosts) |

- Each raw factor is converted to a 0–1 percentile within its sector.
- **Value** and **Quality** blend a static score (average of the factor percentiles) with a trend score (earnings-yield, ROIC and operating-margin boosts). The blend depends on the sector.
- The **Growth** score is a weighted average of the available growth boosts.
- `ROBUST_SCORE` weights Value, Quality and Growth. Missing pillars are dropped and the remaining weights are renormalised.
- A ticker is dropped when it is missing more than 3 value factors or more than 2 quality factors.

All weights live in `backend/calc/settings.py`. Sector-specific overrides live in `backend/calc/sector_weights.json`.

`HQM_SCORE` is computed separately from the pillars. It uses the return windows set in `backend/main.py` (`rank_days`, `screen_days`).

## Requirements

- Python 3.14 (the versions in `backend/requirements.txt` were tested on it)
- An FMP API key

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```
FMP_API_KEY=your_key_here
```

The file is git-ignored. Never commit it.

## Usage

Run from the `backend/` folder:

```bash
cd backend
python main.py --universe sp500
```

`--universe` accepts `sp500` (default), `nasdaq`, `nyse`, `iwm` or `precious_metals`.

The terminal shows:

- a progress bar for OHLC prices and another for fundamentals
- the top 20 tickers by `HQM_SCORE`
- a summary of data gaps and how many times each fallback value was used

### Output

Results go to `backend/output/<universe>/`:

- `<universe>_metrics_<timestamp>.xlsx`: the formatted workbook. Factor columns are hidden by default and grouped by colour. `DOLLAR_VOLUME` and `TURNOVER_RATIO` are shown in scientific notation but stay numeric.
- `metrics_raw.parquet`: the full-precision data frame.

## Caching and rate limits

Everything is cached on disk so reruns are cheap.

| Data | Location | Refresh |
|---|---|---|
| Raw FMP responses | `backend/data/cache/fmp/` | 24 hours (statements: 7 days) |
| Price history | `backend/data/prices/` | Fills only the missing days each run |
| Fundamentals | `backend/data/fundamentals/` | 10 days |

FMP calls are capped at **275 requests per minute** by a shared limiter in `backend/core/cache.py`. Each stage uses 4 worker threads, so the total in-flight requests stay well under 300.

To force a fresh fundamentals pull for a universe, move `backend/data/fundamentals/<universe>_fundamentals.parquet` out of the way and rerun. Cached FMP responses are reused, so this costs only the calls whose cache entries have expired.

## Project layout

```
backend/
├── main.py                 # entry point: runs the pipeline and writes output
├── universes.py            # ticker lists for each universe
├── requirements.txt
├── core/
│   ├── cache.py            # disk cache + rate limiter for FMP calls
│   ├── fetchers.py         # FMP endpoint wrappers
│   ├── fundamentals_cache.py  # fundamentals build and refresh
│   ├── prices.py           # OHLC download and gap fill
│   ├── excel_export.py     # workbook formatting
│   └── paths.py            # project-anchored folders
├── calc/
│   ├── ttm.py              # TTM and yield calculations
│   ├── fallbacks.py        # annual-data fallbacks when TTM is missing
│   ├── boosts.py           # trend boosts (earnings yield, ROIC, margins, growth)
│   ├── stats.py            # CAGR and slope helpers
│   ├── rank.py             # percentile ranks and pillar scores
│   ├── settings.py         # global weights and sector overrides
│   └── sector_weights.json
└── metrics/
    ├── run_metrics.py      # builds one fundamentals row per ticker
    ├── hqm.py              # HQM score and liquidity stats
    └── value_quality.py    # value and quality factor columns
```

## Notes

- FMP's legacy `/api/v3` endpoints are retired and return an empty list. This project uses `/stable` only.
- Some tickers have incomplete data. When TTM figures are missing, the screener falls back to the latest annual figures and records each fallback in a `backups_used` column.
