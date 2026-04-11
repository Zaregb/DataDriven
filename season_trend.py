"""
Seasonality & Trend Exploration
--------------------------------
This script performs a full exploratory analysis of trend and seasonality
on the cleaned demand dataset.

The goal is to:
- Understand long-term demand evolution (trend)
- Detect possible annual seasonality
- Identify structural breaks (e.g. COVID)
- Prepare informed decisions for forecasting models
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORT LIBRARIES
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import STL

# Improve default visualization aesthetics
sns.set(style="whitegrid")

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

# Load the cleaned dataset
df = pd.read_csv("clean_demand.csv", parse_dates=["date"])

# Ensure data is sorted chronologically
df = df.sort_values("date")

# ──────────────────────────────────────────────────────────────────────────────
# 3. BUILD A GLOBAL MONTHLY TIME SERIES
# ──────────────────────────────────────────────────────────────────────────────
# We aggregate all SKUs and locations to study overall demand structure.
# This avoids noise and allows us to see macro-level patterns first.

ts = (
    df.groupby("date")["[SumDemand]"]
      .sum()
      .asfreq("MS")  # Monthly start frequency
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. GLOBAL TIME SERIES PLOT (RAW DATA)
# ──────────────────────────────────────────────────────────────────────────────
# Purpose:
# - Identify long-term trend
# - Detect volatility
# - Visualize structural breaks (e.g. COVID period)

plt.figure(figsize=(12, 5))
plt.plot(ts, label="Monthly demand")
plt.title("Global Monthly Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Total Demand")
plt.legend()
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 5. TREND DETECTION USING MOVING AVERAGE
# ──────────────────────────────────────────────────────────────────────────────
# A 12-month rolling mean smooths short-term noise
# and reveals the underlying long-term trend.

ts_ma_12 = ts.rolling(window=12).mean()

plt.figure(figsize=(12, 5))
plt.plot(ts, alpha=0.4, label="Raw demand")
plt.plot(ts_ma_12, linewidth=2, label="12-month moving average")
plt.title("Trend Estimation Using 12-Month Moving Average")
plt.xlabel("Date")
plt.ylabel("Total Demand")
plt.legend()
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 6. MONTHLY SEASONALITY (BOXPLOT)
# ──────────────────────────────────────────────────────────────────────────────
# Purpose:
# - Compare distributions for each calendar month
# - Detect recurring high or low demand months
# - Strong differences indicate seasonality

plt.figure(figsize=(10, 5))
sns.boxplot(
    x="MonthNumber",
    y="[SumDemand]",
    data=df
)
plt.title("Monthly Demand Distribution (Seasonality)")
plt.xlabel("Month")
plt.ylabel("Demand")
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 7. YEAR × MONTH HEATMAP
# ──────────────────────────────────────────────────────────────────────────────
# Purpose:
# - Visualize seasonal patterns across multiple years
# - Identify repeated annual structures
# - Highlight exceptional periods (e.g. COVID)

pivot_table = df.pivot_table(
    index="YEAR",
    columns="MonthNumber",
    values="[SumDemand]",
    aggfunc="sum"
)

plt.figure(figsize=(12, 6))
sns.heatmap(
    pivot_table,
    cmap="YlOrRd",
    linewidths=0.5
)
plt.title("Demand Heatmap (Year × Month)")
plt.xlabel("Month")
plt.ylabel("Year")
plt.tight_layout()
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 8. STL TIME SERIES DECOMPOSITION
# ──────────────────────────────────────────────────────────────────────────────
# Purpose:
# - Statistically separate:
#     • Trend component
#     • Seasonal component
#     • Residual (noise)
# - Confirm visually what was observed in previous plots

# Remove missing values (required for STL)
ts_clean = ts.dropna()

stl = STL(ts_clean, period=12)
stl_result = stl.fit()

stl_result.plot()
plt.suptitle("STL Decomposition of Monthly Demand", y=1.02)
plt.tight_layout()
plt.show()
