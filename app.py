import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# --- Paths ---
data_dir = "data"
forecast_file = os.path.join(data_dir, "forecast_pre.csv")
anomaly_file = os.path.join(data_dir, "anomalies_pre.csv")
product_file = os.path.join(data_dir, "product_details.csv")  # For categories

# --- Load CSVs ---
forecast_df = pd.read_csv(forecast_file)
anomaly_df = pd.read_csv(anomaly_file)
product_df = pd.read_csv(product_file)

# --- Sidebar: Category & Product Selection ---
st.sidebar.title("Filters")
categories = product_df['Category'].unique()
selected_category = st.sidebar.selectbox("Select Category", categories)

# Filter products by category
products_in_cat = product_df[product_df['Category'] == selected_category]['Product_ID'].unique()
selected_product = st.sidebar.selectbox("Select Product_ID", products_in_cat)

# Optional: Multi-product selection
multi_products = st.sidebar.multiselect("Compare multiple products (optional)", products_in_cat, default=[selected_product])

# Date range filter
min_date = pd.to_datetime(forecast_df['ds']).min()
max_date = pd.to_datetime(forecast_df['ds']).max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

st.title("SmartStock AI Dashboard")
st.write(f"Category: {selected_category} | Selected Products: {multi_products}")

# --- Filter data ---
forecast_data = forecast_df[forecast_df['Product_ID'].isin(multi_products)]
forecast_data = forecast_data[(pd.to_datetime(forecast_data['ds']) >= pd.to_datetime(start_date)) &
                              (pd.to_datetime(forecast_data['ds']) <= pd.to_datetime(end_date))]

anomaly_data = anomaly_df[anomaly_df['Product_ID'].isin(multi_products)]
anomaly_data = anomaly_data[(pd.to_datetime(anomaly_data['Date']) >= pd.to_datetime(start_date)) &
                            (pd.to_datetime(anomaly_data['Date']) <= pd.to_datetime(end_date))]

# --- Stock Alerts ---
st.subheader("Stock Alerts")
stock_alerts = anomaly_data.copy()
stock_alerts['Stock_Status'] = np.where(stock_alerts['Stock_Level'] < stock_alerts['Reorder_Level'], 'Low', 'OK')
stock_alerts_display = stock_alerts[['Product_ID', 'Stock_Level', 'Reorder_Level', 'Stock_Status']]
st.dataframe(stock_alerts_display.style.applymap(lambda x: 'color: red;' if x=='Low' else 'color: green;', subset=['Stock_Status']))

# --- Predictive Restocking Suggestions ---
st.subheader("Restocking Recommendations")
forecast_sum = forecast_data.groupby('Product_ID')['yhat'].sum().reset_index()
stock_sum = anomaly_data.groupby('Product_ID')['Stock_Level'].mean().reset_index()
restock_df = pd.merge(forecast_sum, stock_sum, on='Product_ID')
restock_df['Suggested_Order'] = restock_df['yhat'] - restock_df['Stock_Level']
restock_df['Suggested_Order'] = restock_df['Suggested_Order'].apply(lambda x: max(x, 0))
st.dataframe(restock_df[['Product_ID', 'Stock_Level', 'yhat', 'Suggested_Order']])

# --- Forecast Chart with Confidence Interval and anomalies ---
st.subheader("Sales Forecast")
fig_forecast = px.line(forecast_data, x='ds', y='yhat', color='Product_ID',
                       labels={'ds':'Date','yhat':'Predicted Sales'},
                       title='Predicted Sales Over Time')
# Add upper/lower bounds
for pid in multi_products:
    subset = forecast_data[forecast_data['Product_ID']==pid]
    fig_forecast.add_scatter(x=subset['ds'], y=subset['yhat_upper'], mode='lines', name=f'{pid} Upper Bound', line=dict(dash='dot', color='green'))
    fig_forecast.add_scatter(x=subset['ds'], y=subset['yhat_lower'], mode='lines', name=f'{pid} Lower Bound', line=dict(dash='dot', color='red'))

# Highlight anomalies in red
for pid in multi_products:
    anomalies_dates = anomaly_data[(anomaly_data['Product_ID']==pid) & (anomaly_data['Anomaly_Pred']==1)]['Date']
    fig_forecast.add_scatter(x=anomalies_dates,
                             y=[forecast_data['yhat'].max()/2]*len(anomalies_dates),
                             mode='markers', name=f'{pid} Anomaly', marker=dict(color='red', size=10))

st.plotly_chart(fig_forecast)

st.subheader("Forecast Data Table")
st.dataframe(forecast_data.sort_values(by=['ds','Product_ID']).reset_index(drop=True))

# --- Add: Anomaly Data Table ---
st.subheader("Anomaly Data Table")
st.dataframe(anomaly_data.sort_values(by=['Date','Product_ID']).reset_index(drop=True))

# --- Anomaly Detection Chart ---
st.subheader("Anomaly Detection")
fig_anomaly = px.scatter(anomaly_data, x='Date', y='Stock_Level', color='Anomaly_Pred',
                         color_continuous_scale=['blue','red'],
                         labels={'Stock_Level':'Stock Level','Date':'Date','Anomaly_Pred':'Anomaly'},
                         title='Stock Level and Anomalies')
st.plotly_chart(fig_anomaly)

# --- Historical Trend Comparison ---
st.subheader("Historical Trend Comparison")
for pid in multi_products:
    subset = anomaly_data[anomaly_data['Product_ID']==pid]
    fig_trend = px.line(subset, x='Date', y='Stock_Level', title=f'Stock Trend - Product {pid}')
    st.plotly_chart(fig_trend)

# --- Anomaly Frequency Heatmap ---
st.subheader("Anomaly Frequency Heatmap")
heatmap_data = anomaly_data.groupby(['Product_ID','Warehouse_ID'])['Anomaly_Pred'].sum().reset_index()
fig_heatmap = px.density_heatmap(heatmap_data, x='Warehouse_ID', y='Product_ID', z='Anomaly_Pred',
                                 color_continuous_scale='Reds', title='Anomaly Frequency per Product & Warehouse')
st.plotly_chart(fig_heatmap)

# --- Download Buttons ---
st.download_button("Download Forecast CSV", forecast_data.to_csv(index=False).encode('utf-8'),
                   file_name="forecast_filtered.csv", mime='text/csv')
st.download_button("Download Anomalies CSV", anomaly_data.to_csv(index=False).encode('utf-8'),
                   file_name="anomalies_filtered.csv", mime='text/csv')
