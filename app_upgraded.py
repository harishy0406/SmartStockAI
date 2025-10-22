import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# --- Paths ---
data_dir = "data"
forecast_file = os.path.join(data_dir, "forecast_precomputed.csv")
anomaly_file = os.path.join(data_dir, "anomalies_precomputed.csv")
product_file = os.path.join(data_dir, "product_details.csv")

# --- Load CSVs ---
forecast_df = pd.read_csv(forecast_file)
forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

anomaly_df = pd.read_csv(anomaly_file)
anomaly_df['Date'] = pd.to_datetime(anomaly_df['Date'])

product_df = pd.read_csv(product_file)

# --- Sidebar: Filters ---
st.sidebar.title("Filters")
categories = product_df['Category'].unique()
selected_category = st.sidebar.selectbox("Select Category", categories)

products_in_cat = product_df[product_df['Category'] == selected_category]['Product_ID'].unique()
selected_product = st.sidebar.selectbox("Select Product_ID", products_in_cat)

multi_products = st.sidebar.multiselect(
    "Compare multiple products (optional)", products_in_cat, default=[selected_product]
)

# Date range filter
min_date = forecast_df['ds'].min()
max_date = pd.to_datetime("2026-03-31")  # Extend to March 2026
start_date, end_date = st.sidebar.date_input(
    "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
)

st.title("SmartStock AI Dynamic Dashboard")
st.write(f"Category: {selected_category} | Selected Products: {multi_products}")

# --- Helper function to add realistic fluctuations ---
def add_fluctuations(df, col, scale=0.1):
    fluctuation = df[col] * (scale * np.random.randn(len(df)))
    return (df[col] + fluctuation).clip(lower=0)

# --- Generate dynamic anomalies for future forecast ---
dynamic_anomaly_df = anomaly_df.copy()

future_forecast = forecast_df.copy()
future_forecast = future_forecast[future_forecast['Product_ID'].isin(multi_products)]
future_forecast = future_forecast[(future_forecast['ds'] > anomaly_df['Date'].max()) &
                                  (future_forecast['ds'] <= pd.to_datetime("2026-03-31"))]

future_anomalies_list = []
for _, row in future_forecast.iterrows():
    anomaly_flag = np.random.choice([0, 1], p=[0.9, 0.1])
    stock_level = max(row['yhat'] - np.random.randint(0,10), 0)
    future_anomalies_list.append({
        "Date": row['ds'], "Product_ID": row['Product_ID'],
        "Stock_Level": stock_level,
        "Stock_to_Reorder": stock_level+10,
        "Anomaly_Pred": anomaly_flag
    })

future_anomaly_df = pd.DataFrame(future_anomalies_list)
dynamic_anomaly_df = pd.concat([dynamic_anomaly_df, future_anomaly_df], ignore_index=True)

# Add realistic fluctuations to forecast
forecast_df['yhat'] = add_fluctuations(forecast_df, 'yhat', scale=0.15)
forecast_df['yhat_lower'] = add_fluctuations(forecast_df, 'yhat_lower', scale=0.1)
forecast_df['yhat_upper'] = add_fluctuations(forecast_df, 'yhat_upper', scale=0.1)

# Add weekly seasonality to stock
dynamic_anomaly_df['Day_of_Week'] = dynamic_anomaly_df['Date'].dt.dayofweek
dynamic_anomaly_df['Stock_Level'] += dynamic_anomaly_df['Day_of_Week'] * 2

# --- Filter data ---
forecast_data = forecast_df[forecast_df['Product_ID'].isin(multi_products)]
forecast_data = forecast_data[(forecast_data['ds'] >= pd.to_datetime(start_date)) &
                              (forecast_data['ds'] <= pd.to_datetime(end_date))]

anomaly_data = dynamic_anomaly_df[dynamic_anomaly_df['Product_ID'].isin(multi_products)]
anomaly_data = anomaly_data[(anomaly_data['Date'] >= pd.to_datetime(start_date)) &
                            (anomaly_data['Date'] <= pd.to_datetime(end_date))]

# --- Stock Alerts ---
st.subheader("Stock Alerts")
stock_alerts = anomaly_data.copy()
stock_alerts['Stock_Status'] = np.where(
    stock_alerts['Stock_Level'] < stock_alerts['Stock_to_Reorder'], 'Low', 'OK'
)
stock_alerts_display = stock_alerts[['Product_ID','Stock_Level','Stock_to_Reorder','Stock_Status']]
st.dataframe(
    stock_alerts_display.style.applymap(
        lambda x: 'color: red;' if x=='Low' else 'color: green;', subset=['Stock_Status']
    )
)

# --- Forecast Data Table ---
st.subheader("Forecast Data Table")
st.dataframe(forecast_data.sort_values(by=['ds','Product_ID']).reset_index(drop=True))

# --- Anomaly Data Table ---
st.subheader("Anomaly Data Table")
st.dataframe(anomaly_data.sort_values(by=['Date','Product_ID']).reset_index(drop=True))

# --- Predictive Restocking Suggestions ---
st.subheader("Restocking Recommendations")
forecast_sum = forecast_data.groupby('Product_ID')['yhat'].sum().reset_index()
stock_sum = anomaly_data.groupby('Product_ID')['Stock_Level'].mean().reset_index()
restock_df = pd.merge(forecast_sum, stock_sum, on='Product_ID')
restock_df['Suggested_Order'] = restock_df['yhat'] - restock_df['Stock_Level']
restock_df['Suggested_Order'] = restock_df['Suggested_Order'].apply(lambda x: max(x,0))
st.dataframe(restock_df[['Product_ID','Stock_Level','yhat','Suggested_Order']])

# --- Forecast Chart ---
st.subheader("Sales Forecast")
fig_forecast = px.line(forecast_data, x='ds', y='yhat', color='Product_ID',
                       labels={'ds':'Date','yhat':'Predicted Sales'},
                       title='Predicted Sales Over Time')
for pid in multi_products:
    subset = forecast_data[forecast_data['Product_ID']==pid]
    fig_forecast.add_scatter(
        x=subset['ds'], y=subset['yhat_upper'], mode='lines', name=f'{pid} Upper Bound',
        line=dict(dash='dot', color='green')
    )
    fig_forecast.add_scatter(
        x=subset['ds'], y=subset['yhat_lower'], mode='lines', name=f'{pid} Lower Bound',
        line=dict(dash='dot', color='red')
    )
for pid in multi_products:
    anomalies_dates = anomaly_data[(anomaly_data['Product_ID']==pid) & (anomaly_data['Anomaly_Pred']==1)]['Date']
    fig_forecast.add_scatter(
        x=anomalies_dates,
        y=[forecast_data['yhat'].max()/2]*len(anomalies_dates),
        mode='markers', name=f'{pid} Anomaly', marker=dict(color='red', size=10)
    )
st.plotly_chart(fig_forecast)

# --- Anomaly Detection Chart ---
st.subheader("Anomaly Detection")
fig_anomaly = px.scatter(
    anomaly_data, x='Date', y='Stock_Level', color='Anomaly_Pred',
    color_continuous_scale=['blue','red'],
    labels={'Stock_Level':'Stock Level','Date':'Date','Anomaly_Pred':'Anomaly'},
    title='Stock Level and Anomalies'
)
st.plotly_chart(fig_anomaly)

# --- Historical Trend Comparison ---
st.subheader("Historical Trend Comparison")
for pid in multi_products:
    subset = anomaly_data[anomaly_data['Product_ID']==pid]
    fig_trend = px.line(subset, x='Date', y='Stock_Level', title=f'Stock Trend - Product {pid}')
    st.plotly_chart(fig_trend)

# --- Anomaly Frequency Heatmap ---
st.subheader("Anomaly Frequency Heatmap")
if 'Warehouse_ID' not in anomaly_data.columns:
    anomaly_data['Warehouse_ID'] = np.random.randint(1,4,size=len(anomaly_data))  # simulate 3 warehouses
heatmap_data = anomaly_data.groupby(['Product_ID','Warehouse_ID'])['Anomaly_Pred'].sum().reset_index()
fig_heatmap = px.density_heatmap(
    heatmap_data, x='Warehouse_ID', y='Product_ID', z='Anomaly_Pred',
    color_continuous_scale='Reds', title='Anomaly Frequency per Product & Warehouse'
)
st.plotly_chart(fig_heatmap)

# --- Download Buttons ---
st.download_button("Download Forecast CSV", forecast_data.to_csv(index=False).encode('utf-8'),
                   file_name="forecast_filtered.csv", mime='text/csv')
st.download_button("Download Anomalies CSV", anomaly_data.to_csv(index=False).encode('utf-8'),
                   file_name="anomalies_filtered.csv", mime='text/csv')



# --- Shelf Image Upload & YOLO Detection ---
st.subheader("Shelf Monitoring Live")
st.sidebar.subheader("Shelf Monitoring Live")
if st.sidebar.button("Live"):
    st.write("Live monitoring started...")

st.subheader("Shelf Monitoring Image")
st.sidebar.subheader("Shelf Monitoring Image")
uploaded_file = st.sidebar.file_uploader("Upload shelf image", type=["jpg","png","jpeg"])

ROWS, COLS = 3, 3  # 3x3 grid

if uploaded_file:
    from PIL import Image
    import cv2
    from ultralytics import YOLO
    import numpy as np

    # Load YOLO model
    model = YOLO('yolov8n.pt')  # lightweight YOLOv8

    # Open image
    image = Image.open(uploaded_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(frame)

    # Create empty grid
    frame_height, frame_width, _ = frame.shape
    rack_h = frame_height // ROWS
    rack_w = frame_width // COLS
    grid_labels = np.full((ROWS, COLS), "", dtype=object)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Determine which grid cell the object belongs to
            cx, cy = (x1+x2)//2, (y1+y2)//2
            col_idx = min(cx // rack_w, COLS-1)
            row_idx = min(cy // rack_h, ROWS-1)
            
            # Store label
            grid_labels[row_idx, col_idx] = label

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Draw grid lines
    for r in range(1, ROWS):
        cv2.line(frame, (0, r*rack_h), (frame_width, r*rack_h), (255,255,0), 2)
    for c in range(1, COLS):
        cv2.line(frame, (c*rack_w, 0), (c*rack_w, frame_height), (255,255,0), 2)

    # Convert frame back to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, caption="Shelf Detection", use_column_width=True)

    # Show detected items per grid
    st.subheader("Detected Items Grid (3x3)")
    for r in range(ROWS):
        row_items = [grid_labels[r,c] if grid_labels[r,c] != "" else "Empty" for c in range(COLS)]
        st.write(" | ".join(row_items))
