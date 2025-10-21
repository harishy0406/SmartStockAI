# predict_api.py
import os
import sys
import joblib
import pandas as pd

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)
from preprocess import preprocess_data

# Load models
model_dir = os.path.join(project_dir, "models")
forecast_model = joblib.load(os.path.join(model_dir,"forecast_model.pkl"))
anomaly_model = joblib.load(os.path.join(model_dir,"anomaly_detector.pkl"))

# Forecast function
def forecast_stock(df_input):
    df_input = df_input.copy()
    df_input.rename(columns={'Date':'ds','Sales':'y'}, inplace=True)
    if df_input.shape[0] < 2:
        raise ValueError("Not enough data to forecast.")
    forecast = forecast_model.predict(df_input)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]

# Anomaly detection function
def detect_anomaly(df_input):
    df_input = df_input.copy()
    features = ['Stock_Level','Sales','Stock_to_Reorder','Day_of_Week','Month']
    df_features = df_input[features]
    preds = anomaly_model.predict(df_features)
    df_input['Anomaly_Pred'] = [1 if x==-1 else 0 for x in preds]
    return df_input

# Test
if __name__ == "__main__":
    df, _ = preprocess_data()
    df.columns = df.columns.str.strip()
    product_counts = df['Product_ID'].value_counts()
    eligible_products = product_counts[product_counts >= 2].index.tolist()
    product_id = eligible_products[0]
    df_product = df[df['Product_ID']==product_id]

    forecast_df = forecast_stock(df_product)
    anomaly_df = detect_anomaly(df_product)

    print("Forecast sample:")
    print(forecast_df.head())
    print("Anomaly detection sample:")
    print(anomaly_df.head())
