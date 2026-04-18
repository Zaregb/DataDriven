import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1 - Load data
    print("Step 1: Loading data...")
    pca_demand_path = os.path.join(script_dir, "pca_demand.csv")
    pca_demand = pd.read_csv(pca_demand_path, index_col=0) # Index is Product_ID
    X_raw = pca_demand.fillna(0).values # 94 x 108
    print(f"X_raw shape: {X_raw.shape}")
    
    clean_demand_path = os.path.join(script_dir, "clean_demand.csv")
    df_clean = pd.read_csv(clean_demand_path, sep=';', low_memory=False)
    df_clean['[SumDemand]'] = pd.to_numeric(df_clean['[SumDemand]'], errors='coerce').fillna(0)
    df_clean['ASP ($)'] = pd.to_numeric(df_clean['ASP ($)'], errors='coerce')
    
    # Step 2 - Aggregate clean_demand per SKU-Month
    print("Step 2: Aggregating clean_demand...")
    sku_month_df = df_clean.groupby(['Product_ID', 'YEAR', 'MonthNumber']).agg({
        '[SumDemand]': 'sum',
        'outlier_flag': 'max' # 1 if any country had an outlier for this SKU-month
    }).reset_index()
    
    # Sort to ensure chronological order: YEAR then MonthNumber
    sku_month_df = sku_month_df.sort_values(['Product_ID', 'YEAR', 'MonthNumber'])
    
    # We also need overall ASP per SKU
    asp_per_sku = df_clean.groupby('Product_ID')['ASP ($)'].mean()
    
    # Arrays to hold our 7 features
    skus = pca_demand.index
    n_skus = len(skus)
    features = pd.DataFrame(index=skus, columns=[
        'Mean_Demand', 'CV', 'Trend_Slope', 'Seasonality_Strength', 
        'ASP', 'Mean_Demand_Value', 'Sparsity'
    ], dtype=float)
    
    # Step 3 - Compute the 7 features per SKU
    print("Step 3: Computing features per SKU...")
    for sku in skus:
        # Get full series from pca_demand (which was already pivoted and sorted chronologically YYYY_MM)
        full_series = pca_demand.loc[sku].values
        
        # Get SKU specific aggregated data
        sku_data = sku_month_df[sku_month_df['Product_ID'] == sku].copy()
        sku_data = sku_data.sort_values(['YEAR', 'MonthNumber']).reset_index(drop=True)
        
        # 1-3. We need non-outlier months
        non_outlier_mask = sku_data['outlier_flag'] != 1
        non_outlier_data = sku_data[non_outlier_mask]
        
        if len(non_outlier_data) > 0:
            mean_demand = non_outlier_data['[SumDemand]'].mean()
            std_demand = non_outlier_data['[SumDemand]'].std(ddof=1) if len(non_outlier_data) > 1 else 0
            cv = (std_demand / mean_demand) if mean_demand != 0 else 0
            
            # Trend slope on non-outlier months
            # time_index is the chronological integer index (0 to N-1) from the full series
            time_indices = non_outlier_data.index.values
            demand_vals = non_outlier_data['[SumDemand]'].values
            if len(time_indices) > 1:
                slope, _ = np.polyfit(time_indices, demand_vals, deg=1)
            else:
                slope = 0
        else:
            mean_demand = 0
            cv = 0
            slope = 0
            
        # 4. Seasonality strength
        # reshape full series (108 months) -> (n_years, 12). 108 / 12 = 9
        if len(full_series) % 12 == 0:
            n_years = len(full_series) // 12
            reshaped = full_series.reshape(n_years, 12)
            monthly_means = reshaped.mean(axis=0) # mean per calendar month
            std_monthly = np.std(monthly_means, ddof=1) if n_years > 1 else np.std(monthly_means)
            std_full = np.std(full_series, ddof=1) if len(full_series) > 1 else np.std(full_series)
            
            seasonality = (std_monthly / std_full) if std_full != 0 else 0
        else:
            seasonality = 0
            
        # 5. ASP
        asp = asp_per_sku.get(sku, 0)
        
        # 6. Mean demand value
        mean_demand_val = mean_demand * asp
        
        # 7. Sparsity
        sparsity = np.sum(full_series == 0) / len(full_series)
        
        features.loc[sku, 'Mean_Demand'] = mean_demand                   #PC1
        features.loc[sku, 'CV'] = cv                                     #PC2
        features.loc[sku, 'Trend_Slope'] = slope                         #PC3
        features.loc[sku, 'Seasonality_Strength'] = seasonality          #PC4
        features.loc[sku, 'ASP'] = asp                                   #PC5
        features.loc[sku, 'Mean_Demand_Value'] = mean_demand_val         #PC6
        features.loc[sku, 'Sparsity'] = sparsity                         #PC7
        
    print(f"Features shape generated: {features.shape}")
    print("Sample of features:")
    print(features.head())
    
    # Step 4 - Autoscaling
    print("Step 4: Autoscaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Step 5 - Apply PCA
    print("Step 5: Applying PCA...")
    pca = PCA(n_components=7)
    pca.fit(X_scaled)
    
    L = pca.explained_variance_ratio_
    A = pca.components_.T
    Z = X_scaled @ A
    
    # Create DataFrame for Z
    columns = [f"PC{i+1}" for i in range(7)]
    Z_df = pd.DataFrame(Z, index=features.index, columns=columns)
    
    # Step 6 & 7 - Plots
    print("Step 6 & 7: Generating Scree plot and Score scatter plot...")
    plt.figure(figsize=(14, 6))
    
    # Scree Plot
    plt.subplot(1, 2, 1)
    x_components = np.arange(1, 8)
    cumulative_variance = np.cumsum(L)
    
    plt.bar(x_components, L, alpha=0.7, label='Marginal', color='skyblue')
    plt.plot(x_components, cumulative_variance, marker='o', color='red', label='Cumulative')
    
    plt.axhline(y=0.80, color='gray', linestyle='--', alpha=0.7, label='80% Variance')
    plt.axhline(y=0.90, color='black', linestyle='--', alpha=0.7, label='90% Variance')
    
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.ylim(0, 1.05)
    plt.legend()
    
    # Score Scatter Plot
    plt.subplot(1, 2, 2)
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.7, color='steelblue')
    
    for i, sku in enumerate(features.index):
        plt.annotate(sku, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.6)
        
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Score Scatter Plot (PC1 vs PC2)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Step 8 - Save outputs
    print("Step 8: Saving outputs...")
    features_out = os.path.join(script_dir, 'sku_features.csv')
    scores_out = os.path.join(script_dir, 'sku_pca_scores.csv')
    
    features.to_csv(features_out)
    Z_df.to_csv(scores_out)
    print(f"Saved: {features_out}\nSaved: {scores_out}")
    print("Done!")

if __name__ == "__main__":
    main()
