import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Configuration ─────────────────────────────────────────────────────────────

COMPANIES = {
    "TCS":      "TCS.NS",
    "Infosys":  "INFY.NS",
    "Wipro":    "WIPRO.NS",
}

START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ── Step 1: Download Data ──────────────────────────────────────────────────────

print("Downloading stock data...")

raw = {}
for name, ticker in COMPANIES.items():
    print(f"  Fetching {name} ({ticker})...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    close = df["Close"]
    # yfinance 0.2+ may return a DataFrame with a MultiIndex column — flatten it
    if hasattr(close, "squeeze"):
        close = close.squeeze()
    raw[name] = close
    print(f"  {name}: {len(df)} trading days")

# ── Step 2: Align to Common Timeline ──────────────────────────────────────────

# Combine into one DataFrame — this automatically aligns dates
# Any date missing for one company (e.g. holiday) becomes NaN
combined = pd.DataFrame(raw)

# Drop rows where ANY company has missing data
combined.dropna(inplace=True)

print(f"\nAligned dataset: {len(combined)} common trading days")
print(f"Date range: {combined.index[0].date()} to {combined.index[-1].date()}")

# ── Step 3: Normalize ─────────────────────────────────────────────────────────

# Min-max normalization: scales each stock to range [0, 1]
# This is important because TCS, Infosys, Wipro trade at very different price levels
normalized = (combined - combined.min()) / (combined.max() - combined.min())

# ── Step 4: Save to CSV ───────────────────────────────────────────────────────

combined.save    = combined.to_csv("data/raw_prices.csv")
normalized.to_csv("data/normalized_prices.csv")
print("\nSaved: data/raw_prices.csv")
print("Saved: data/normalized_prices.csv")

# ── Step 5: Plot ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Raw prices
for name in COMPANIES:
    axes[0].plot(combined.index, combined[name], label=name)
axes[0].set_title("Raw Stock Prices (INR)")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Price (INR)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Normalized prices
for name in COMPANIES:
    axes[1].plot(normalized.index, normalized[name], label=name)
axes[1].set_title("Normalized Stock Prices [0–1]")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Normalized Price")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/task1_time_series.png", dpi=150)
plt.show()
print("\nPlot saved: plots/task1_time_series.png")

print("\nTask 1 complete!")
