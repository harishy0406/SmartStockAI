import pandas as pd
import os

def load_data():
    """Load datasets from data folder."""
    try:
        products = pd.read_csv("data/product_details.csv")
        inventory = pd.read_csv("data/inventory_data.csv")
        anomalies = pd.read_csv("data/anomalies.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")
    
    # Standardize column names (remove spaces, lowercase)
    products.columns = [col.strip().replace(" ", "_") for col in products.columns]
    inventory.columns = [col.strip().replace(" ", "_") for col in inventory.columns]
    anomalies.columns = [col.strip().replace(" ", "_") for col in anomalies.columns]

    # Ensure necessary columns exist
    required_cols = ['Product_ID', 'Product_Name', 'Brand', 'Category', 'Cost_Price', 'Selling_Price', 'Reorder_Level']
    for col in required_cols:
        if col not in products.columns:
            raise ValueError(f"Missing column in product_details.csv: {col}")

    return products, inventory, anomalies

def preprocess_data():
    """Merge datasets and create features for ML."""
    products, inventory, anomalies = load_data()

    # Merge inventory with product details
    df = inventory.merge(products, on='Product_ID', how='left')

    # Check merge result
    if df.isnull().sum().sum() > 0:
        print("⚠️ Warning: There are missing values after merge")

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Create basic features
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    # Safe Stock_to_Reorder calculation
    if 'Reorder_Level' in df.columns:
        df['Stock_to_Reorder'] = df['Stock_Level'] / df['Reorder_Level']
    else:
        df['Stock_to_Reorder'] = 0  # fallback if missing

    return df, anomalies

if __name__ == "__main__":
    df, anomalies = preprocess_data()
    print("Preprocessing completed :")
    print(df.head())
