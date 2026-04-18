import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ==============================================================
    # Step 1 - Load data
    # ==============================================================
    print("Step 1: Loading data...")

    pca_demand = pd.read_csv(
        os.path.join(script_dir, "pca_demand.csv"),
        index_col=0
    )

    df_clean = pd.read_csv(
        os.path.join(script_dir, "clean_demand.csv"),
        low_memory=False
    )

    df_clean.columns = df_clean.columns.str.replace('[\\[\\]]', '', regex=True)

    df_clean['SumDemand'] = pd.to_numeric(df_clean['SumDemand'], errors='coerce').fillna(0)
    df_clean['ASP ($)'] = pd.to_numeric(df_clean['ASP ($)'], errors='coerce')

    # ==============================================================
    # Step 2 - Aggregate per SKU-Month
    # ==============================================================
    sku_month_df = (
        df_clean
        .groupby(['Product_ID', 'YEAR', 'MonthNumber'])
        .agg({'SumDemand': 'sum', 'outlier_flag': 'max'})
        .reset_index()
        .sort_values(['Product_ID', 'YEAR', 'MonthNumber'])
    )

    asp_per_sku = df_clean.groupby('Product_ID')['ASP ($)'].mean()

    # ==============================================================
    # Step 3 - Feature engineering (7 features)
    # ==============================================================
    print("Step 3: Computing features...")

    skus = pca_demand.index

    features = pd.DataFrame(
        index=skus,
        columns=[
            'Mean_Demand',
            'CV',
            'Trend_Slope',
            'Seasonality_Strength',
            'ASP',
            'Mean_Demand_Value',
            'Sparsity'
        ],
        dtype=float
    )

    for sku in skus:
        full_series = pca_demand.loc[sku].values
        sku_data = sku_month_df[sku_month_df['Product_ID'] == sku].reset_index(drop=True)
        non_outlier = sku_data[sku_data['outlier_flag'] != 1]

        if len(non_outlier) > 0:
            mean_demand = non_outlier['SumDemand'].mean()
            std_demand = non_outlier['SumDemand'].std(ddof=1) if len(non_outlier) > 1 else 0
            cv = std_demand / mean_demand if mean_demand != 0 else 0

            if len(non_outlier) > 1:
                slope, _ = np.polyfit(non_outlier.index, non_outlier['SumDemand'], 1)
            else:
                slope = 0
        else:
            mean_demand = cv = slope = 0

        if len(full_series) % 12 == 0 and np.std(full_series) != 0:
            reshaped = full_series.reshape(-1, 12)
            seasonality = np.std(reshaped.mean(axis=0)) / np.std(full_series)
        else:
            seasonality = 0

        asp = asp_per_sku.get(sku, 0)
        sparsity = np.mean(full_series == 0)

        features.loc[sku] = [
            mean_demand,
            cv,
            slope,
            seasonality,
            asp,
            mean_demand * asp,
            sparsity
        ]

    # ==============================================================
    # Step 4 - Scaling
    # ==============================================================
    print("Step 4: Scaling...")
    X_scaled = StandardScaler().fit_transform(features)

    # ==============================================================
    # Step 5 - PCA
    # ==============================================================
    print("Step 5: PCA...")
    pca = PCA(n_components=7)
    Z = pca.fit_transform(X_scaled)

    Z_df = pd.DataFrame(
        Z,
        index=features.index,
        columns=[f"PC{i+1}" for i in range(7)]
    )

    L = pca.explained_variance_ratio_

    # ==============================================================
    # Step 6 - Scree plot (annotated PCs)
    # ==============================================================
    plt.figure(figsize=(7, 5))
    x = np.arange(1, 8)

    plt.bar(x, L, alpha=0.7, label="Variance expliquée")
    plt.plot(x, np.cumsum(L), marker='o', label="Variance cumulée")

    for i, val in enumerate(L):
        plt.text(i + 1, val + 0.01, f"PC{i+1}", ha='center')

    plt.axhline(0.8, linestyle='--', color='gray', label='80 %')
    plt.axhline(0.9, linestyle='--', color='black', label='90 %')

    plt.xlabel("Composantes principales")
    plt.ylabel("Variance expliquée")
    plt.title("Scree Plot – PCA")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==============================================================
    # Step 7 - 6 scatter plots PC1–PC4
    # ==============================================================
    pc_pairs = [
        ('PC1', 'PC2'),
        ('PC1', 'PC3'),
        ('PC1', 'PC4'),
        ('PC2', 'PC3'),
        ('PC2', 'PC4'),
        ('PC3', 'PC4')
    ]

    plt.figure(figsize=(14, 8))

    for i, (pcx, pcy) in enumerate(pc_pairs, 1):
        plt.subplot(2, 3, i)
        plt.scatter(Z_df[pcx], Z_df[pcy], alpha=0.7)
        for sku in Z_df.index:
            plt.annotate(
                sku,
                (Z_df.loc[sku, pcx], Z_df.loc[sku, pcy]),
                fontsize=7,
                alpha=0.5
            )
        plt.xlabel(pcx)
        plt.ylabel(pcy)
        plt.title(f"{pcx} vs {pcy}")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ==============================================================
    # Step 8 - LOADINGS (INTERPRÉTATION PCA)
    # ==============================================================
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features.columns,
        columns=[f"PC{i+1}" for i in range(7)]
    )

    print("\n=== PCA LOADINGS ===")
    print(loadings.round(3))

    # --- Heatmap ---
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("PCA Loadings – Features vs PCs")
    plt.tight_layout()
    plt.show()

    # --- Barplots par PC ---
    for pc in loadings.columns[:4]:
        plt.figure(figsize=(6, 4))
        loadings[pc].sort_values().plot(kind='barh')
        plt.axvline(0, color='black')
        plt.title(f"Contribution des features – {pc}")
        plt.tight_layout()
        plt.show()

    # ==============================================================
    # Step 9 - Save outputs
    # ==============================================================
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    features.to_csv(os.path.join(output_dir, "sku_features.csv"))
    Z_df.to_csv(os.path.join(output_dir, "sku_pca_scores.csv"))
    loadings.to_csv(os.path.join(output_dir, "sku_pca_loadings.csv"))

    print("✅ PCA completed and interpreted successfully.")


if __name__ == "__main__":
    main()