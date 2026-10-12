# --------------------- CONFIGURABLE WEIGHTS --------------------- #

# Weights to combine static & trend for the Value pillar:
VAL_STATIC_W = 0.67      # e.g. 70% weight on the static composite
VAL_TREND_W  = 0.33      # e.g. 30% weight on the “earnings_yield_boost_pct”

# Weights to combine static & trend for the Quality pillar:
QUAL_STATIC_W = 0.5
QUAL_ROIC_W   = 0.25
QUAL_OPM_W    = 0.25

# Weights for the Growth pillar (all trend‐based):
GR_EBIT_W = 0.4
GR_REV_W  = 0.3
GR_FCF_W  = 0.3

# Final composite weights (Value vs. Quality vs. Growth):
FINAL_VAL_W   = 0.4
FINAL_QUAL_W  = 0.4
FINAL_GR_W    = 0.2

# -------------------  Sector‑specific overrides  ------------------- #
from pathlib import Path
import json

# JSON lives in the same folder as this file
_SECTOR_CFG_PATH = Path(__file__).with_name("sector_weights.json")

try:
    with open(_SECTOR_CFG_PATH, "r", encoding="utf-8") as fp:
        _SECTOR_CFG = json.load(fp)
except FileNotFoundError:
    _SECTOR_CFG = {}            # falls back to global defaults

# Helper for “inherit” chains in the JSON
def _inherit(sector: str, key: str, default):
    info = _SECTOR_CFG.get(sector, {})
    if "inherit" in info:
        return _inherit(info["inherit"], key, default)
    return info.get(key, default)

# --------- public helpers the rest of the code will import ---------
def sector_value_split(sector: str) -> tuple[float, float]:
    """
    Return (static_w, trend_w) for the Value pillar of *sector*.
    Falls back to global VAL_STATIC_W / VAL_TREND_W defaults.
    """
    return tuple(_inherit(sector, "value_split", (VAL_STATIC_W, VAL_TREND_W)))

def sector_quality_split(sector: str) -> tuple[float, float]:
    """
    Return (static_w, trend_w) for the Quality pillar of *sector*.
    Defaults to QUAL_STATIC_W vs (1‑QUAL_STATIC_W) if not overridden.
    """
    default = (QUAL_STATIC_W, 1 - QUAL_STATIC_W)
    return tuple(_inherit(sector, "quality_split", default))

def sector_pillar_weights(sector: str) -> tuple[float, float, float]:
    """
    Return (Value, Quality, Growth) final pillar weights for *sector*.
    Defaults to FINAL_VAL_W / FINAL_QUAL_W / FINAL_GR_W.
    """
    return tuple(_inherit(
        sector,
        "pillar_weights",
        (FINAL_VAL_W, FINAL_QUAL_W, FINAL_GR_W),
    ))

