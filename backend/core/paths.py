# core/paths.py
"""Project-anchored locations, so output does not depend on the working directory."""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
CACHE_DIR = DATA_DIR / "cache" / "fmp"        # raw FMP JSON, keyed by URL hash
PRICES_DIR = DATA_DIR / "prices"              # OHLCV history per universe
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"  # fundamentals per universe
OUTPUT_DIR = BACKEND_DIR / "output"           # reports: output/<universe>/
