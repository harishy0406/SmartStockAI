import pandas as pd # type: ignore
import random
import numpy as np # type: ignore
from datetime import datetime, timedelta
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# -----------------------------
# 1. Product Details (same for all warehouses)
# -----------------------------
products = [
    ("P001", "Tata Salt 1kg", "Grocery", "Tata", 20, 25, 30, 730),
    ("P002", "Colgate Toothpaste 100g", "Personal Care", "Colgate", 40, 60, 25, 720),
    ("P003", "Fortune Sunflower Oil 1L", "Grocery", "Adani Wilmar", 120, 150, 40, 365),
    ("P004", "Lux Soap Bar", "Personal Care", "Hindustan Unilever", 25, 40, 20, 730),
    ("P005", "Britannia Marie Biscuits", "Snacks", "Britannia", 15, 25, 30, 180),
    ("P006", "Thums Up Bottle 500ml", "Beverage", "Coca-Cola India", 20, 35, 40, 180),
    ("P007", "Dabur Amla Hair Oil 200ml", "Personal Care", "Dabur", 70, 110, 30, 540),
    ("P008", "Maggi Noodles 2-min 70g", "Snacks", "Nestle India", 25, 40, 35, 150)
]

warehouses = ["W001", "W002", "W003"]  # 3 warehouses

product_df = pd.DataFrame(products, columns=[
    "Product_ID", "Product_Name", "Category", "Brand",
    "Cost_Price", "Selling_Price", "Reorder_Level", "Shelf_Life_Days"
])
product_df.to_csv("product_details.csv", index=False)
print("✅ product_details.csv created")

# -----------------------------
# 2. Inventory Data (multi-warehouse)
# -----------------------------
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 6, 30)
inventory_data = []

current_date = start_date
while current_date <= end_date:
    month_factor = 1.0
    if current_date.month in [3, 4, 5]:
        month_factor = 1.3
    elif current_date.month in [1, 2]:
        month_factor = 0.9

    for wid in warehouses:
        for pid, pname, category, brand, cost, sell, reorder, shelf in products:
            base_stock = random.randint(80, 300)  # varied per warehouse
            base_sales = int(np.clip(np.random.normal(20, 10), 5, 50))
            sales = int(base_sales * month_factor)

            # Random anomalies
            if random.random() < 0.05:  # spike
                sales += random.randint(20, 50)
            if random.random() < 0.03:  # drop
                sales -= random.randint(10, 30)

            stock_level = max(base_stock - sales, 10)
            inventory_data.append([
                current_date.strftime("%Y-%m-%d"), wid, pid, pname, category, brand,
                stock_level, sales, reorder
            ])
    current_date += timedelta(days=1)

inventory_df = pd.DataFrame(inventory_data, columns=[
    "Date", "Warehouse_ID", "Product_ID", "Product_Name", "Category", "Brand",
    "Stock_Level", "Sales", "Reorder_Level"
])
inventory_df.to_csv("inventory_data.csv", index=False)
print("✅ inventory_data.csv created")

# -----------------------------
# 3. Anomalies Data (multi-warehouse)
# -----------------------------
summary = inventory_df.groupby(["Warehouse_ID","Product_ID"])["Sales"].agg(["mean","std"]).reset_index()
anomalies = []

for index, row in inventory_df.iterrows():
    wid = row["Warehouse_ID"]
    pid = row["Product_ID"]
    sales = row["Sales"]
    stock = row["Stock_Level"]
    reorder = row["Reorder_Level"]
    
    mean = summary.loc[(summary["Warehouse_ID"]==wid) & (summary["Product_ID"]==pid), "mean"].values[0]
    std = summary.loc[(summary["Warehouse_ID"]==wid) & (summary["Product_ID"]==pid), "std"].values[0]

    if sales > mean + 2*std or stock < reorder:
        anomaly_type = "High sales spike" if sales > mean + 2*std else "Sudden drop in stock"
        anomalies.append([
            row["Date"], wid, pid, row["Product_Name"], row["Brand"],
            anomaly_type, stock, sales, "High", "No"
        ])

anomaly_df = pd.DataFrame(anomalies, columns=[
    "Date", "Warehouse_ID", "Product_ID", "Product_Name", "Brand",
    "Anomaly_Type", "Stock_Level", "Sales", "Severity", "Resolved"
])
anomaly_df.to_csv("anomalies.csv", index=False)
print("✅ anomalies.csv created")
