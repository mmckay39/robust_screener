Section 1: Exchange Universes
A. Stocks Exchanges
Use these three major U.S. exchanges separately. Rank tickers within each universe for clean percentile scores and sector balance:
NASDAQ: Tech-heavy, growth, volatile Short/mid-term momentum
S&P 500: Large-cap blend, stable companies Long-term trend investing
NYSE: Mixed bag: industrials, banks, dividend payers Balanced strategies

B. ETF-Based Universes
Use ETFs to define curated stock pools:
QQQ: NASDAQ-100 tech/growth High-momentum, explosive movers
SPY: S&P 500 Stable, broad-market stocks
IWM: Russell 2000 small-caps Volatility hunting, risky momentum
VUG: Large-cap growth Growth-only screening
VTV: Large-cap value Value + stability blend
XLF/XLE/XLK: Sector-specific ETFs Rotational or theme-based momentum


Section 2: Filters
A. Volume Filters (Liquidity Gatekeepers)
To avoid false momentum signals from illiquid stocks, apply:
1. Share Volume Filter
avg_20d_volume > 100,000-500,000 shares/day

2. Dollar Volume Filter (recommended for ETFs & small-caps)
avg_20d_dollar_volume > $10 million/day
Dollar volume = share volume × price. More robust for filtering out penny stocks and thin movers

3. Fundamental Filters (Quality/Value)
A. Metrics
- Free Cash Flow: Positive FCF or top 50% percentile
- Debt-to-Equity: Prefer D/E < 1.0
- Return on Equity (ROE): Aim for ROE > 10%
- Earnings Growth: EPS growth > 10% year-over-year
- P/E or EV/EBITDA: Filter out highly overvalued stocks
- Price-to-Earnings Ratio'
- Forward Price-to-Earnings Ratio'
- Price-to-Book Ratio
- Price-to-Sales Ratio
- EV/GP:
- PEG Ratio: 
- FCF Yield:
- Net Debt/EBITDA: Measures leverage more accurately than D/E. FMP API, Alpha Vantage (totalDebt/ebitda)
- Shareholder Yield	(Dividend Yield + Buyback Yield) → Total cash returned to shareholders.	Calculate from yfinance/SEC filings
- Altman Z-Score	Predicts bankruptcy risk (avoid value traps).	FMP API (or calculate manually)






B. Providers
- Financial Modeling Prep
- AlphaVantage
- yFinance


4. Technical Filters
• Above moving averages (e.g., 50-day or 200-day)
• Trend strength (optional): ADX > 20 or 25
• Relative strength vs SPY or sector ETFs: RS > 1.0
• Recent breakout: Price at 20-day high or 3% above last close

5. Price and Market Cap Filters
• Minimum price: Avoid stocks under $10
• Market cap range: For example, $500M to $50B
• Beta: Filter based on volatility (e.g., 0.8 to 2.0)

6. Sentiment & Institutional Filters
• Analyst upgrades or downgrades
• Institutional ownership percentage
• Insider buying/selling activity

7. Event-Based Filters
• Upcoming or recent earnings reports
• Earnings beats or positive revisions
• Mergers and acquisitions (M&A) activity
• Dividend increases

8. HQM Score 
- Multiple Timeframes
- Single screener
- HQM Acceleration

Features:
Script Launches
I choose my timeframe
Weights auto adjust
It generates a list of stocks
Everything should have weight adjusters on it

Section 3: Timeframe Analysis
3-5 timeframes is optimal. Less is too fragile, more is overfitting

1. Holding Periods
A. Short Term (1-3 months)
- Use: 5d, 10d, 21d
- Maybe 63 days if you want something anchoring

B. Medium Term (3-12 months)
- Use: 10d, 21d, 63d, 126d
- Good balance of short-term and trend

C. Long Term (1 Year - 2 Years)
- Use: 21d, 63d, 126d, 252d
- Drop 5d and 10d — too noisy

2. Weighting the timeframes
HQM Timeframe Weights by Holding Period

Short-term (1–3 months):
5d: 30%
10d: 35%
21d: 25%
63d: 10%

Medium-term (3–12 months):
10d: 15%
21d: 25%
63d: 35%
126d: 25%

Long-term (12–24 months):
21d: 10%
63d: 25%
126d: 30%
252d: 35%

