import pandas as pd

def create_pca_dataset():
    # Read the data
    print("Reading clean_demand.csv...")
    df = pd.read_csv('clean_demand.csv', sep=';', low_memory=False)
    
    # Create the pivot table
    # Rows: Product_ID (SKUs)
    # Columns: YEAR and MonthNumber
    # Values: [SumDemand]
    print("Pivoting data...")
    # Convert [SumDemand] to numeric to prevent sum aggregation errors
    df['[SumDemand]'] = pd.to_numeric(df['[SumDemand]'], errors='coerce')
    
    pivot_df = df.pivot_table(
        index='Product_ID', 
        columns=['YEAR', 'MonthNumber'], 
        values='[SumDemand]', 
        aggfunc='sum'
    ).fillna(0)
    
    # Flatten the column names from MultiIndex (YEAR, MonthNumber) to just 'YYYY_MM'
    pivot_df.columns = [f"{year}_{month:02d}" for year, month in pivot_df.columns]
    
    # Save the transformed dataset
    output_filename = 'pca_demand.csv'
    pivot_df.to_csv(output_filename)
    
    print(f"Success! Created {output_filename} with shape {pivot_df.shape}")
    print(f"Expected shape is roughly (94, 48) - 94 SKUs and 48 months.")

if __name__ == "__main__":
    create_pca_dataset()
