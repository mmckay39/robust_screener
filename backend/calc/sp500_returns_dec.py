import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# SETTINGS
# -----------------------------------------
TICKER = "^GSPC"
YEARS  = [2020, 2021, 2022, 2023, 2024]

# -----------------------------------------
# FUNCTION: Get cumulative December returns
# -----------------------------------------
def december_returns(year):
    start = f"{year}-12-01"
    end   = f"{year}-12-31"

    # Explicit auto_adjust=False so Adj Close is available
    data = yf.download(TICKER, start=start, end=end, auto_adjust=False, progress=False)

    if data.empty:
        raise ValueError(f"No data found for December {year}")

    # Use Adjusted Close
    prices = data["Adj Close"].copy()

    # Map index → day of month
    days = prices.index.day
    prices.index = days

    # If multiple rows for same day (rare), take last
    prices = prices.groupby(prices.index).last().sort_index()

    # Cumulative return vs Dec 1
    returns = prices / prices.iloc[0] - 1
    return returns

# -----------------------------------------
# PLOT
# -----------------------------------------
plt.figure(figsize=(10, 6))

for year in YEARS:
    r = december_returns(year)
    plt.plot(r.index, r.values, marker="o", label=str(year))

plt.title("S&P 500 December Cumulative Returns (2020–2024)")
plt.xlabel("Day of December")
plt.ylabel("Cumulative Return vs Dec 1")
plt.grid(True)
plt.legend(title="Year")
plt.xticks(range(1, 32))
plt.tight_layout()

plt.show()
