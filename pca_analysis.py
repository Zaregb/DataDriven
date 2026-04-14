import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    # ==========================================
    # Step 1 — Load the data
    # ==========================================
    print("Loading data...")
    # Load pca_demand.csv, using the first column (Product_ID) as the index
    df = pd.read_csv('pca_demand.csv', index_col=0)
    
    # Extract numpy array and sku names
    X = df.values
    sku_names = df.index.values
    
    print(f"Expected shape: (94, 108)")
    print(f"Actual data shape: {X.shape}")
    
    # ==========================================
    # Step 3 — Row-wise scaling
    # ==========================================
    # We want to scale each SKU (row) across all 108 months (axis=1)
    # Using keepdims=True ensures the summary arrays remain (94, 1) so they broadcast correctly
    print("Performing row-wise scaling...")
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    
    # Standardize row-wise to remove volume differences
    # To prevent division by zero for SKUs with constant demand, replace 0 std with 1
    X_std[X_std == 0] = 1
    X_scaled = (X - X_mean) / X_std
    
    # ==========================================
    # Step 4 — Apply PCA
    # ==========================================
    print("Applying PCA...")
    n_components = 94
    pca = PCA(n_components=n_components)
    
    # Fit the PCA on the row-scaled matrix
    pca.fit(X_scaled)
    
    # Extract variance, loadings, and scores (following the style in exercise3_scaling_sol.py)
    L = pca.explained_variance_ratio_ # Explained variance ratio
    A = pca.components_.T             # Loadings
    Z = X_scaled @ A                  # Scores
    
    # ==========================================
    # Step 5 — Scree plot
    # ==========================================
    print("Generating Scree Plot...")
    plt.figure(figsize=(10, 6))
    components = np.arange(1, n_components + 1)
    
    # Plot marginal and cumulative variance
    plt.scatter(components, L, s=15, color='k', label='Marginal explained variance')
    plt.plot(components, np.cumsum(L), color='r', marker='o', markersize=3, label='Cumulative explained variance')
    
    # Add horizontal dashed lines at 80% and 90%
    plt.axhline(y=0.80, color='gray', linestyle='--', label='80% threshold')
    plt.axhline(y=0.90, color='gray', linestyle=':', label='90% threshold')
    
    plt.xlabel('Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot of PCA on Row-Scaled Demand')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ==========================================
    # Step 6 — Score scatter plot
    # ==========================================
    print("Generating Scores Scatter Plot (PC1 vs PC2)...")
    plt.figure(figsize=(10, 8))
    
    # Plot PC1 vs PC2 scores, one point per SKU
    plt.scatter(Z[:, 0], Z[:, 1], c='k', alpha=0.5)
    
    # Label each point with its SKU name
    for i, sku in enumerate(sku_names):
        plt.annotate(sku, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.7, 
                     xytext=(2, 2), textcoords='offset points')
                     
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA Scores on SKUs')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
