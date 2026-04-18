import pandas as pd
import os

def create_pca_dataset():
    print("Reading clean_demand.csv...")

    # Path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'clean_demand.csv')

    df = pd.read_csv(input_path, low_memory=False)

    # ---- FIX IMPORTANT ----
    # Remove brackets from column names if present
    df.columns = df.columns.str.replace('[\\[\\]]', '', regex=True)

    # Safety check
    required_cols = {'Product_ID', 'YEAR', 'MonthNumber', 'SumDemand'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Pivoting data...")

    # Ensure numeric demand
    df['SumDemand'] = pd.to_numeric(df['SumDemand'], errors='coerce').fillna(0)

    pivot_df = df.pivot_table(
        index='Product_ID',
        columns=['YEAR', 'MonthNumber'],
        values='SumDemand',
        aggfunc='sum'
    ).fillna(0)

    # Flatten columns: YYYY_MM
    pivot_df.columns = [
        f"{year}_{int(month):02d}"
        for year, month in pivot_df.columns
    ]

    # Save
    output_path = os.path.join(script_dir, 'pca_demand.csv')
    pivot_df.to_csv(output_path)

    print(f"✅ Success! Created {output_path}")
    print("Shape:", pivot_df.shape)

if __name__ == "__main__":
    create_pca_dataset()