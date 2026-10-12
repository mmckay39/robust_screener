# core/cache.py

import os
import time
import json
import hashlib
import pathlib
import requests

# ------------------------------------------------------------ #
# 1. Configuration:                                          #
# ------------------------------------------------------------ #

# Where on disk we store cached JSON files
_CACHE_DIR = pathlib.Path(".cache_fmp")  
_CACHE_DIR.mkdir(exist_ok=True)

# How many seconds a cached file remains “fresh” before we re-download it
# (e.g., 3600 = 1 hour; set to a larger value if you don’t need intraday accuracy)
DEFAULT_TTL = 86400
# 3600        # 1 hour
# 10_800      # 3 hours
# 86_400      # 1 day
# 2_592_000   # 30 days (≈ 1 month)

# Ensure we don’t exceed ~290 requests/minute → sleep between calls
MAX_REQUESTS_PER_MINUTE = 299  
_SLEEP_SEC = 60.0 / MAX_REQUESTS_PER_MINUTE 
# _SLEEP_SEC = 60.0 / 1200
# _SLEEP_SEC = 0


# ------------------------------------------------------------ #
# 2. Helpers to map URL → cache‐filename                       #
# ------------------------------------------------------------ #

def _url_to_cache_path(url: str) -> pathlib.Path:
    """
    Turn a URL string into a safe filename, by hashing the URL.
    e.g. 'https://...&limit=12' → '.cache_fmp/abcdef123456.json'
    """
    # Use a stable hash (SHA-256) so filenames are uniform length
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    # You could add an extension that reflects endpoint, but hex→.json is fine.
    return _CACHE_DIR / f"{h}.json"


# ------------------------------------------------------------ #
# 3. The main cached_get_json() function                        #
# ------------------------------------------------------------ #

def cached_get_json(
    url: str,
    ttl: int = DEFAULT_TTL,
    force_refresh: bool = False
) -> dict | list | None:
    """
    1) Compute cache_path from URL.
    2) If file exists && (now - mtime) < ttl && not force_refresh:
         → load JSON from disk, return it.
    3) Otherwise:
         → sleep (_SLEEP_SEC) to respect rate limit,
         → perform `requests.get(url).json()`,
         → save JSON to disk under cache_path,
         → return JSON.
    Returns:
      • list or dict (whatever was in the response)
      • None if the HTTP request failed or parsed JSON was invalid.
    """
    cache_path = _url_to_cache_path(url)

    # 2. If cached and not stale, return that
    if cache_path.exists() and not force_refresh:
        mtime = cache_path.stat().st_mtime
        age   = time.time() - mtime
        if age < ttl:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                # If reading/decoding fails, fall through and re-download
                pass

    # 3. Need to fetch fresh  → only now do we sleep
    #time.sleep(_SLEEP_SEC)  # simple rate-limit throttle

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        # On any network or JSON parse error, return None
        return None

    # Attempt to write the JSON to disk for future use
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        # If we can’t write to disk (e.g. permission issue), ignore silently
        pass

    return data

