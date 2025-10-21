# train_anomaly.py
import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

# Add scripts folder to path for preprocess
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)
from preprocess import preprocess_data

# Load data
df, _ = preprocess_data()
df.columns = df.columns.str.strip()

# Features for anomaly detection
features = ['Stock_Level', 'Sales', 'Stock_to_Reorder', 'Day_of_Week', 'Month']
missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

X = df[features]

# Train Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X)
print(f"Isolation Forest trained on {X.shape[0]} rows")

# Save model
model_dir = os.path.join(project_dir, "models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "anomaly_detector.pkl")
joblib.dump(iso, model_path)
print(f"Model saved at: {model_path}")