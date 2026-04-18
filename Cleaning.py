import pandas as pd
import numpy as np

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df1 = pd.read_excel('data-driven_Demand_1.xlsx', sheet_name='Export')
#df2 = pd.read_excel('data-driven_Demand_2.xslsx', sheet_name='Export')
asp  = pd.read_excel('data-driven_Demand_1.xlsx', sheet_name='Sheet1')

df = pd.concat([df1], ignore_index=True)
#df = pd.concat([df1, df2], ignore_index=True)

# ── 2. DROP USELESS COLUMNS & FUTURE ROWS ────────────────────────────────────
df.drop(columns=['Day', '[SumFCF_LAG0]', '[SumSTAT_LAG0]'], inplace=True)
df = df[df['YEAR'] <= 2025].reset_index(drop=True)

# ── 3. FILL NaN IN [SumDemand] WITH 0 ────────────────────────────────────────
# Justification: absence of a record in a transactional system = no order placed
df['[SumDemand]'] = df['[SumDemand]'].fillna(0)

# ── 4. ADD DATE COLUMN ────────────────────────────────────────────────────────
df['date'] = pd.to_datetime(
    df['YEAR'].astype(str) + '-' + df['MonthNumber'].astype(str) + '-01'
)

# ── 5. ADD COVID FLAG ─────────────────────────────────────────────────────────
# Business disruption period: Q2 2020 → Q4 2021
df['covid_flag'] = (
    ((df['YEAR'] == 2020) & (df['QuarterNo'] >= 2)) |
    (df['YEAR'] == 2021)
).astype(int)

# ── 6. MERGE ASP + COMPUTE DEMAND VALUE ──────────────────────────────────────
df = df.merge(asp, on='Product_ID', how='left')
df['demand_value'] = df['[SumDemand]'] * df['ASP ($)']

# ── 7. OUTLIER DETECTION (Z-SCORE PER SKU×COUNTRY, EXCLUDING COVID) ──────────
# Justification: compute baseline mean/std on non-COVID data only,
# then apply to full series. Flag if |z| > 3.
df['zscore']       = np.nan
df['outlier_flag'] = 0

for (sku, country), idx in df.groupby(['Product_ID', 'Country']).groups.items():
    group         = df.loc[idx]
    normal_idx    = group[group['covid_flag'] == 0].index
    normal_demand = df.loc[normal_idx, '[SumDemand]']

    if len(normal_demand) < 6 or normal_demand.std() == 0:
        continue

    mu    = normal_demand.mean()
    sigma = normal_demand.std()

    df.loc[idx, 'zscore'] = (df.loc[idx, '[SumDemand]'] - mu) / sigma

    outlier_idx = normal_idx[df.loc[normal_idx, 'zscore'].abs() > 3]
    df.loc[outlier_idx, 'outlier_flag'] = 1

# ── 8. SORT & EXPORT ──────────────────────────────────────────────────────────
df.sort_values(['Product_ID', 'Country', 'date'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv('clean_demand.csv', index=False)

print("Done. Shape:", df.shape)
print("Columns:", list(df.columns))
print("COVID rows flagged:", df['covid_flag'].sum())
print("Outliers flagged:  ", df['outlier_flag'].sum())
