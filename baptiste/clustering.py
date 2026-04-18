import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import os


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scores_path = os.path.join(script_dir, "outputs", "sku_pca_scores.csv")

    # ==============================================================
    # Step 1 - Load PCA scores
    # ==============================================================
    print("Step 1: Loading PCA scores...")

    Z_df = pd.read_csv(scores_path, index_col=0)

    # We cluster only on the first PCs (as in the notebook)
    X = Z_df[['PC1', 'PC2', 'PC3', 'PC4']].values

    print("Shape of clustering matrix:", X.shape)

    # ==============================================================
    # Step 2 - Scale (good practice, even if PCA already scaled)
    # ==============================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==============================================================
    # Step 3 - Davies–Bouldin analysis (k = 2 to 10)
    # ==============================================================
    print("Step 3: Computing Davies–Bouldin index...")

    k_range = range(2, 11)
    db_scores = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init='random',
            n_init=20,
            random_state=42
        )
        labels = kmeans.fit_predict(X_scaled)
        db = davies_bouldin_score(X_scaled, labels)
        db_scores.append(db)
        print(f"k = {k}, DB index = {db:.3f}")

    # ==============================================================
    # Step 4 - Plot DB index vs k
    # ==============================================================
    plt.figure(figsize=(6, 4))
    plt.plot(k_range, db_scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Davies–Bouldin index")
    plt.title("Cluster selection using Davies–Bouldin index")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Best number of clusters
    best_k = k_range[np.argmin(db_scores)]
    print(f"\n✅ Optimal number of clusters according to DB index: k = {best_k}")

    # ==============================================================
    # Step 5 - Final KMeans with optimal k
    # ==============================================================
    print("Step 5: Fitting KMeans with optimal k...")

    kmeans = KMeans(
        n_clusters=best_k,
        init='random',
        n_init=20,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)
    Z_df['Cluster'] = labels

    # ==============================================================
    # Step 6 - Visualisation of clusters
    # ==============================================================
    plt.figure(figsize=(12, 5))

    # PC1 vs PC2
    plt.subplot(1, 2, 1)
    plt.scatter(Z_df['PC1'], Z_df['PC2'], c=labels, cmap='tab10')
    for sku in Z_df.index:
        plt.annotate(
            sku,
            (Z_df.loc[sku, 'PC1'], Z_df.loc[sku, 'PC2']),
            fontsize=8,
            alpha=0.6
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters in PC1–PC2 space")
    plt.grid(alpha=0.3)

    # PC1 vs PC3
    plt.subplot(1, 2, 2)
    plt.scatter(Z_df['PC1'], Z_df['PC3'], c=labels, cmap='tab10')
    for sku in Z_df.index:
        plt.annotate(
            sku,
            (Z_df.loc[sku, 'PC1'], Z_df.loc[sku, 'PC3']),
            fontsize=8,
            alpha=0.6
        )
    plt.xlabel("PC1")
    plt.ylabel("PC3")
    plt.title("Clusters in PC1–PC3 space")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ==============================================================
    # Step 7 - Save results
    # ==============================================================
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    Z_df.to_csv(os.path.join(output_dir, "sku_pca_clusters.csv"))

    print("\n✅ Clustering analysis completed.")
    print("Clusters saved to: outputs/sku_pca_clusters.csv")


if __name__ == "__main__":
    main()