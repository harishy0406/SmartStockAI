# smartstock_vision_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from ultralytics import YOLO
from PIL import Image
import time
import os

# ==============================
# 📁 PATH SETUP
# ==============================
data_dir = "data"
forecast_file = os.path.join(data_dir, "forecast_precomputed.csv")
anomaly_file = os.path.join(data_dir, "anomalies_precomputed.csv")
product_file = os.path.join(data_dir, "product_details.csv")

# ==============================
# 🎛️ SIDEBAR: MODE SELECTION
# ==============================
st.sidebar.title("SmartStock AI System")
mode = st.sidebar.radio(
    "Select Mode:",
    ["📊 Analytics Dashboard", "🖼️ Shelf Image Detection", "🎥 Live Shelf Monitoring"]
)

# ==============================
# MODE 1: ANALYTICS DASHBOARD
# ==============================
if mode == "📊 Analytics Dashboard":
    st.title("📈 SmartStock AI: Dynamic Inventory Dashboard")

    # --- Load CSVs ---
    forecast_df = pd.read_csv(forecast_file)
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    anomaly_df = pd.read_csv(anomaly_file)
    anomaly_df['Date'] = pd.to_datetime(anomaly_df['Date'])
    product_df = pd.read_csv(product_file)

    # --- Sidebar Filters ---
    categories = product_df['Category'].unique()
    selected_category = st.sidebar.selectbox("Select Category", categories)
    products_in_cat = product_df[product_df['Category'] == selected_category]['Product_ID'].unique()
    selected_product = st.sidebar.selectbox("Select Product_ID", products_in_cat)
    multi_products = st.sidebar.multiselect("Compare Multiple Products", products_in_cat, default=[selected_product])

    # Date range filter
    min_date = forecast_df['ds'].min()
    max_date = pd.to_datetime("2026-03-31")
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
    )

    # --- Helper Function ---
    def add_fluctuations(df, col, scale=0.1):
        fluctuation = df[col] * (scale * np.random.randn(len(df)))
        return (df[col] + fluctuation).clip(lower=0)

    # --- Generate dynamic anomalies ---
    dynamic_anomaly_df = anomaly_df.copy()
    future_forecast = forecast_df.copy()
    future_forecast = future_forecast[future_forecast['Product_ID'].isin(multi_products)]
    future_forecast = future_forecast[
        (future_forecast['ds'] > anomaly_df['Date'].max()) &
        (future_forecast['ds'] <= pd.to_datetime("2026-03-31"))
    ]

    future_anomalies_list = []
    for _, row in future_forecast.iterrows():
        anomaly_flag = np.random.choice([0, 1], p=[0.9, 0.1])
        stock_level = max(row['yhat'] - np.random.randint(0, 10), 0)
        future_anomalies_list.append({
            "Date": row['ds'], "Product_ID": row['Product_ID'],
            "Stock_Level": stock_level, "Stock_to_Reorder": stock_level + 10,
            "Anomaly_Pred": anomaly_flag
        })
    future_anomaly_df = pd.DataFrame(future_anomalies_list)
    dynamic_anomaly_df = pd.concat([dynamic_anomaly_df, future_anomaly_df], ignore_index=True)

    # --- Add fluctuations ---
    forecast_df['yhat'] = add_fluctuations(forecast_df, 'yhat', scale=0.15)
    forecast_df['yhat_lower'] = add_fluctuations(forecast_df, 'yhat_lower', scale=0.1)
    forecast_df['yhat_upper'] = add_fluctuations(forecast_df, 'yhat_upper', scale=0.1)

    # --- Filter Data ---
    forecast_data = forecast_df[forecast_df['Product_ID'].isin(multi_products)]
    forecast_data = forecast_data[
        (forecast_data['ds'] >= pd.to_datetime(start_date)) &
        (forecast_data['ds'] <= pd.to_datetime(end_date))
    ]

    anomaly_data = dynamic_anomaly_df[dynamic_anomaly_df['Product_ID'].isin(multi_products)]
    anomaly_data = anomaly_data[
        (anomaly_data['Date'] >= pd.to_datetime(start_date)) &
        (anomaly_data['Date'] <= pd.to_datetime(end_date))
    ]

    # --- Stock Alerts ---
    st.subheader("🚨 Stock Alerts")
    stock_alerts = anomaly_data.copy()
    stock_alerts['Stock_Status'] = np.where(
        stock_alerts['Stock_Level'] < stock_alerts['Stock_to_Reorder'], 'Low', 'OK'
    )
    stock_alerts_display = stock_alerts[['Product_ID', 'Stock_Level', 'Stock_to_Reorder', 'Stock_Status']]
    st.dataframe(stock_alerts_display.style.applymap(
        lambda x: 'color: red;' if x == 'Low' else 'color: green;', subset=['Stock_Status'])
    )

    # --- Forecast Chart ---
    st.subheader("📈 Sales Forecast")
    fig_forecast = px.line(
        forecast_data, x='ds', y='yhat', color='Product_ID',
        title='Predicted Sales Over Time'
    )
    st.plotly_chart(fig_forecast)

    # --- Anomaly Detection Chart ---
    st.subheader("📊 Anomaly Detection")
    fig_anomaly = px.scatter(
        anomaly_data, x='Date', y='Stock_Level', color='Anomaly_Pred',
        color_continuous_scale=['blue', 'red']
    )
    st.plotly_chart(fig_anomaly)

    # --- Historical Trend ---
    st.subheader("📉 Historical Trend Comparison")
    for pid in multi_products:
        subset = anomaly_data[anomaly_data['Product_ID'] == pid]
        fig_trend = px.line(subset, x='Date', y='Stock_Level', title=f'Stock Trend - {pid}')
        st.plotly_chart(fig_trend)

# ==============================
# MODE 2: IMAGE UPLOAD DETECTION
# ==============================
elif mode == "🖼️ Shelf Image Detection":
    st.title("🧠 SmartShelf - Image Detection Mode")
    uploaded_image = st.file_uploader("Upload a shelf image (3x3 grid)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        frame = np.array(image)
        st.image(frame, caption="Uploaded Shelf", use_column_width=True)

        model = YOLO("yolov8n.pt")
        results = model(frame, stream=True)
        ROWS, COLS = 3, 3
        frame_height, frame_width, _ = frame.shape
        rack_h, rack_w = frame_height // ROWS, frame_width // COLS
        label_matrix = [["Empty" for _ in range(COLS)] for _ in range(ROWS)]

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if conf > 0.5:
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    c, r_ = min(cx // rack_w, COLS-1), min(cy // rack_h, ROWS-1)
                    label_matrix[r_][c] = label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Detected Items", use_column_width=True)
        st.subheader("📋 Shelf Grid Summary")
        df = pd.DataFrame(label_matrix, columns=[f"Col {i+1}" for i in range(COLS)])
        df.index = [f"Rack {i+1}" for i in range(ROWS)]
        st.dataframe(df)

# ==============================
# MODE 3: LIVE WEBCAM MONITORING
# ==============================
elif mode == "🎥 Live Shelf Monitoring":
    st.title("📹 Real-Time Shelf Monitoring (3x3 Grid)")
    model = YOLO("yolov8n.pt")

    ROWS, COLS = 3, 3
    cap = cv2.VideoCapture(1)  # Change to 0 or 1 based on your webcam index

    stframe = st.empty()
    delay = st.slider("Frame Delay (seconds)", 0.1, 2.0, 0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Camera not detected or disconnected.")
            break

        frame_height, frame_width, _ = frame.shape
        rack_h, rack_w = frame_height // ROWS, frame_width // COLS
        label_matrix = [["Empty" for _ in range(COLS)] for _ in range(ROWS)]

        results = model(frame, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if conf > 0.5:
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    c, r_ = min(cx // rack_w, COLS-1), min(cy // rack_h, ROWS-1)
                    label_matrix[r_][c] = label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Draw grid
        for r in range(1, ROWS):
            cv2.line(frame, (0, r*rack_h), (frame_width, r*rack_h), (255,255,0), 2)
        for c in range(1, COLS):
            cv2.line(frame, (c*rack_w, 0), (c*rack_w, frame_height), (255,255,0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        time.sleep(delay)

    cap.release()
