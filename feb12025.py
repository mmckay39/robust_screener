from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------- Step 1: Download Data for Multiple Tickers ---------------------
tickers = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]
dfs = []
for ticker in tickers:
    data = yf.download(ticker, period="2y", interval="1d")
    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data["Ticker"] = ticker  # Label each row with its ticker
    dfs.append(data)
    
# Concatenate data for all tickers and sort by date
df = pd.concat(dfs)
df.sort_index(inplace=True)  # Ensure data is time-sorted

# --------------------- Step 2: Compute Common Technical Features ---------------------
price_col = "Close"

# Compute Log Returns, Future Returns and Volatility BEFORE feature engineering:
df["Log_Returns"] = np.log(df[price_col] / df[price_col].shift(1))
# Future_Returns: forward return over next 5 days
df["Future_Returns"] = df[price_col].shift(-5) / df[price_col] - 1
# 10-day rolling volatility
df["Volatility"] = df[price_col].pct_change().rolling(10).std()

# Relative Strength Index (RSI)
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["RSI"] = compute_rsi(df[price_col])

# MACD Histogram (Momentum Indicator)
df["MACD"] = df[price_col].ewm(span=12).mean() - df[price_col].ewm(span=26).mean()
df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

# Rate of Change (ROC) over 10 days
df["ROC"] = df[price_col].pct_change(10) * 100

# Average True Range (ATR)
df["High-Low"] = df["High"] - df["Low"]
df["High-Close"] = abs(df["High"] - df[price_col].shift(1))
df["Low-Close"] = abs(df["Low"] - df[price_col].shift(1))
df["TR"] = df[["High-Low", "High-Close", "Low-Close"]].max(axis=1)
df["ATR"] = df["TR"].rolling(14).mean()

# Bollinger Band Width
rolling_mean = df[price_col].rolling(window=20).mean()
rolling_std = df[price_col].rolling(window=20).std()
df["Bollinger_Width"] = (rolling_std / rolling_mean) * 100

# On-Balance Volume (OBV)
df["OBV"] = (np.sign(df[price_col].diff()) * df["Volume"]).cumsum()

# Additional Trend Features: cumulative returns and trend strength
df["Cumulative_Returns_5D"] = df[price_col].pct_change(5).rolling(5).sum()
df["Cumulative_Returns_10D"] = df[price_col].pct_change(10).rolling(10).sum()
df["Cumulative_Returns_20D"] = df[price_col].pct_change(20).rolling(20).sum()
df["Rate_of_Change"] = df[price_col].pct_change(5)
df["Trend_Strength"] = df["RSI"] - 50  # Center RSI around 0

# Drop rows with NaN values from rolling calculations
df.dropna(inplace=True)

# --------------------- Step 3: Preprocessing ---------------------
# Define our enhanced feature set.
features = [
    "RSI", "MACD_Hist", "ROC",
    "Cumulative_Returns_5D", "Cumulative_Returns_10D", "Cumulative_Returns_20D",
    "Rate_of_Change", "Trend_Strength",
    "Volatility", "ATR", "Bollinger_Width", "OBV"
]

# Replace any remaining NaNs and Infs
df.fillna(method="ffill", inplace=True)
df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
# Add a small noise term to prevent issues with covariance matrices
X_scaled += np.random.normal(0, 1e-6, X_scaled.shape)

# --------------------- Step 4: Train HSMM with n_components=3 ---------------------
hsmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=500, random_state=42)
hsmm.fit(X_scaled)
df["HSMM_State"] = hsmm.predict(X_scaled)

# --------------------- Step 5: Analyze State Durations ---------------------
df["State_Shift"] = df["HSMM_State"] != df["HSMM_State"].shift(1)
df["State_Duration"] = df.groupby(df["State_Shift"].cumsum()).cumcount() + 1

# --------------------- Step 6: Evaluate and Label States ---------------------
# Compute average future returns and volatility per state
state_summary = df.groupby("HSMM_State").agg({"Future_Returns": "mean", "Volatility": "mean"})
# Sort states by average Future_Returns (lowest → highest)
sorted_states = state_summary.sort_values(by="Future_Returns", ascending=True).index
state_returns = df.groupby("HSMM_State")["Future_Returns"].mean()

# For 3 states, assign:
# - Lowest returns → "Bearish"
# - Middle → "Sideways"
# - Highest → "Bullish"
state_mapping = {
    sorted_states[0]: "Bearish",
    sorted_states[1]: "Sideways",
    sorted_states[2]: "Bullish"
}
df["Market_Regime"] = df["HSMM_State"].map(state_mapping)

# --------------------- Step 7: Output Analysis ---------------------
print("Assigned Market Regimes (Average Future Returns per HSMM State):")
print(state_returns.sort_values())

print("\nUpdated Average Duration in Each Market Regime:")
print(df.groupby("Market_Regime")["State_Duration"].mean())

print("\nUpdated Average Future Returns per Market Regime:")
print(df.groupby("Market_Regime")["Future_Returns"].mean())

# --------------------- Step 8: Plot Market States ---------------------
# Compute HSMM state probabilities
state_probs = hsmm.predict_proba(X_scaled)
for i in range(3):
    df[f"Prob_State_{i}"] = state_probs[:, i]

# --- New: Plot Uptrend Probability Over Time ---
# Identify which state is mapped to "Bullish"
uptrend_state = [s for s, regime in state_mapping.items() if regime == "Bullish"][0]
df["Uptrend_Prob"] = df[f"Prob_State_{uptrend_state}"]

plt.figure(figsize=(14, 6))
plt.plot(df.index, df["Uptrend_Prob"], color="green", linewidth=2, label="Uptrend Probability")
plt.axhline(0.7, color="red", linestyle="--", linewidth=1, label="High Confidence Threshold (0.7)")
plt.title("Uptrend Probability Over Time (HSMM, n_components=3)")
plt.xlabel("Date")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.show()

# --- Create a multi-panel figure with Price segmentation, RSI, and HSMM probabilities ---
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Panel 1: Price segmentation plot
ax1 = axs[0]
ax1.set_title("Price Segments Colored by HSMM State (n_components=3)")
ax1.set_ylabel("Price")
grouped = df.groupby((df["HSMM_State"] != df["HSMM_State"].shift()).cumsum())
for name, group in grouped:
    state = group["HSMM_State"].iloc[0]
    label = state_mapping.get(state, str(state))
    ax1.plot(group.index, group[price_col], color=plt.cm.coolwarm(state / 3.), linewidth=2)
    mid_idx = group.index[len(group) // 2]
    duration = group["State_Duration"].iloc[-1]
    ax1.annotate(f"{label}\n({duration} days)", xy=(mid_idx, group[price_col].iloc[len(group)//2]),
                 fontsize=8, color="black", ha="center", va="bottom")
for t in df.index[df["State_Shift"]]:
    ax1.axvline(x=t, color="gray", linestyle="--", linewidth=0.5)

# Panel 2: RSI indicator plot
ax2 = axs[1]
ax2.plot(df.index, df["RSI"], color="purple", label="RSI")
ax2.axhline(70, color="red", linestyle="--", linewidth=1, label="Overbought (70)")
ax2.axhline(30, color="green", linestyle="--", linewidth=1, label="Oversold (30)")
ax2.set_title("RSI (Relative Strength Index)")
ax2.set_ylabel("RSI")
ax2.legend()

# Panel 3: HSMM state probabilities plot
ax3 = axs[2]
colors = plt.cm.viridis(np.linspace(0, 1, 3))
for i in range(3):
    ax3.plot(df.index, df[f"Prob_State_{i}"], color=colors[i], label=f"Prob State {i}")
ax3.set_title("HSMM State Probabilities (n_components=3)")
ax3.set_ylabel("Probability")
ax3.set_xlabel("Date")
ax3.legend(loc="upper left")

plt.tight_layout()
import pyPortfolioOpt as ppo

plt.show()
