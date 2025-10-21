# train_prophet.py
import os
import sys
import joblib
import pandas as pd
from prophet import Prophet

# Add scripts folder to path for preprocess
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)
from preprocess import preprocess_data

# Load data
df, _ = preprocess_data()
df.columns = df.columns.str.strip()

# Automatically pick Product_ID with enough rows
product_counts = df['Product_ID'].value_counts()
eligible_products = product_counts[product_counts >= 2].index.tolist()
if not eligible_products:
    raise ValueError("No Product_ID has enough rows for training.")

product_id = eligible_products[0]
df_product = df[df['Product_ID']==product_id][['Date','Sales']].copy()
df_product.rename(columns={'Date':'ds','Sales':'y'}, inplace=True)

# Train Prophet
model = Prophet()
model.fit(df_product)
print(f"Prophet model trained for Product_ID={product_id}")

# Save model
model_dir = os.path.join(project_dir, "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "forecast_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")
