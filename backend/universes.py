import requests, pandas as pd, re
from io import StringIO
from core.cache import cached_get_json

from dotenv import load_dotenv
load_dotenv()  # loads from .env file in project root
import os
FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise EnvironmentError("❌ Missing FMP_API_KEY environment variable")

# --- Precious‑metals universe (FMP stock‑screener) --------------------------
def download_precious_metals():
    INDUSTRIES = [
        "Gold",
        "Silver",
        "Other Precious Metals/Minerals",
    ]
    EXCHANGES = "NYSE,NASDAQ,AMEX"
    REGEX_CLEAN = re.compile(r"^[A-Z]{1,5}$")   # same 1‑4‑letter filter you use elsewhere
    
    tickers = []
    
    for industry in INDUSTRIES:
        url = (
            "https://financialmodelingprep.com/stable/company-screener"
            f"?industry={industry.replace(' ', '%20')}"
            f"&exchange={EXCHANGES}&limit=5000&apikey={FMP_API_KEY}"
        )
        tickers.extend([entry["symbol"] for entry in cached_get_json(url)])
    
    clean_precious_metals = (
        pd.Series(tickers)
          .dropna()
          .str.upper()
          .drop_duplicates()
          .loc[lambda s: s.str.match(REGEX_CLEAN)]
          .tolist()
    )
    
    # print(f"✅ Precious‑metals tickers found: {len(clean_precious_metals)}")
    # print(clean_precious_metals[:20])  # peek
    return clean_precious_metals
    # ----------------------------------------------------------------------------

# # Download list of SP500stocks from wikipedia
# def download_SP500():

#     sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    
#     sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
    
#     symbols_list = sp500['Symbol'].unique().tolist()
    
#     # Filter out anything that isnt only all caps letters
#     clean_sp500 = [s for s in symbols_list if re.fullmatch(r'[A-Z]+', s)] #clean symbols using regex

#     return clean_sp500

# Download list of SP500 stocks from Wikipedia
def download_SP500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}   # <-- added header to avoid 403

    # Fetch the page with requests (instead of letting pd.read_html hit it directly)
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    # Parse the tables from the HTML text
    sp500 = pd.read_html(StringIO(resp.text))[0]

    sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-")
    symbols_list = sp500["Symbol"].unique().tolist()

    # Filter out anything that isn’t only all caps letters
    clean_sp500 = [s for s in symbols_list if re.fullmatch(r"[A-Z]+", s)]

    return clean_sp500

    # 2) ----------  NASDAQ ----------
def download_NASDAQ():
    nasdaq_url = "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    nasdaq_txt = requests.get(nasdaq_url, timeout=10).text
    
    # read; skip the one‑line footer with creation time
    nasdaq_df = pd.read_csv(
        StringIO(nasdaq_txt),
        sep='|',
        skipfooter=1,
        engine='python'
    )
    
    ## --------- hard filters ----------
    nasdaq_df = nasdaq_df[
        (nasdaq_df['ETF'] == 'N') & # Drop ETFs / ETNs / CEFs (ETF == 'N') - only want operating companies, not funds.
        (nasdaq_df['Test Issue'] == 'N') & # Drop test issues (Test Issue == 'N') - “ZVZZT”‑style tickers used for NASDAQ system tests.
        (nasdaq_df['Financial Status'] == 'N') & # Drop deficiency/ bankruptcy flags (Financial Status == 'N') - Codes D (deficient) or E (delinquent) often precede delistings.
        (nasdaq_df['NextShares'] == 'N') & # Exclude NextShares (NextShares == 'N') -	An obscure exch‑traded mutual‑fund wrapper — no reason to trade them.
        (nasdaq_df['Market Category'].isin(['Q', 'G']))  # keep Global Select/Global
    ]
    
    # text‑based exclusions
    nasdaq_df = nasdaq_df[
        ~nasdaq_df['Security Name'].str.contains('ADR', na=False) & # Exclude ADRs (search "ADR" in Security Name) - Foreign receipts behave differently; you can screen them separately.
        ~nasdaq_df['Security Name'].str.contains('Acquisition', na=False)  # simple SPAC catch
    ]
    
    # symbol‑shape filter (drops units/rights/wts/penny suffixes)
    #nasdaq_df = nasdaq_df[nasdaq_df['Symbol'].str.match(r'^[A-Z]{1,5}$')] # Symbol regex: only 1‑4 all‑caps letters	- Kicks out SPAC/units/rights ('.U', '.W', etc.).
    
    # build the clean list
    clean_nasdaq = nasdaq_df['Symbol'].unique().tolist()

    return clean_nasdaq

# 3) ----------  NYSE -------------------------------------------------------------------------------------------------
def download_NYSE():
    other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    other_txt = requests.get(other_url, timeout=10).text
    
    other_df = pd.read_csv(
        StringIO(other_txt),
        sep='|',
        skipfooter=1,        # strip the one‑line “File Creation Time” footer
        engine='python'
    )
    
    # --------  NYSE core filters  --------
    nyse_df = other_df[
        (other_df['Exchange'] == 'N') &      # Big Board only
        (other_df['ETF'] == 'N') & # Drop ETFs / ETNs / CEFs (ETF == 'N') - only want operating companies, not funds.
        (other_df['Test Issue'] == 'N') & # Drop test issues (Test Issue == 'N') - “ZVZZT”‑style tickers used for NYSE system tests.
        ~other_df['Security Name'].str.contains('ADR', na=False) # Exclude ADRs (search "ADR" in Security Name) - Foreign receipts behave differently; you can screen them separately.
    ]
    
    nyse_df = nyse_df[~nyse_df['Security Name'].str.contains('Acquisition', na=False)] # simple SPAC catch
    
    # regex drops SPAC units/warrants/rights + (optionally) class shares w/ dot
    symbol_regex = r'^[A-Z]{1,5}$'          # keep ≥1 and ≤4 capital letters only. 
    #symbol_regex = r'^[A-Z]{1,4}([.-][A-Z])?$' # If you do want class‑share tickers like BRK.B, change the regex to allow . or -:
    nyse_df = nyse_df[nyse_df['ACT Symbol'].str.match(symbol_regex)]
    
    clean_nyse = nyse_df['ACT Symbol'].unique().tolist()

    return clean_nyse

# 3) ----------  IWM  -------------
def download_IWM():
    iwm_url = (
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
        "latest-holdings.csv"
    )
    iwm_text = requests.get(iwm_url, timeout=10).text
    iwm_df   = pd.read_csv(StringIO(iwm_text), skiprows=9)  # rows 0‑8 are headers
    
    # ------------- IWM filters -------------
    iwm_df = iwm_df[
        (iwm_df['Weight (%)'] > 0) & # Weight > 0 - drops footer/header noise that sometimes parses as NaNs
        ~iwm_df['Name'].str.contains('Cash',     na=False) & # No cash placeholders - removes money‑market sweep lines
        ~iwm_df['Name'].str.contains('Treasury', na=False) & # No treasury placeholders - removes money‑market sweep lines
        ~iwm_df['Name'].str.contains('ADR',      na=False) & # No ADRs -  want only U.S.‑listed operating companies.
        ~iwm_df['Name'].str.contains(r' LP\b',   na=False) # No LPs / MLPs - avoids pass‑through tax headaches
    ]
    
    ticker_regex = r'^[A-Z]{1,5}$' # kicks out units/warrants (*.U, .W, etc.)
    iwm_df = iwm_df[iwm_df['Ticker'].str.match(ticker_regex, na=False)]
    
    clean_iwm = iwm_df['Ticker'].unique().tolist()

    return clean_iwm