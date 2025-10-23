# SmartStock AI: Predictive Inventory Management System

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.40-orange?logo=streamlit)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object--Detection-green?logo=opencv)](https://github.com/ultralytics/ultralytics)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue?logo=pandas)](https://pandas.pydata.org/)

---

## 🚀 Project Overview

**SmartStock AI** is an intelligent inventory management system that uses **machine learning** and **computer vision** to help warehouses predict, monitor, and optimize their stock levels.  
It brings together **real-time stock insights**, **AI-based forecasting**, and **object detection** for a smarter and more automated inventory workflow.

### 🔍 Key Features

- 📦 **Predictive Inventory Forecasting** – Anticipates stock needs based on past trends.  
- 🧠 **AI-Powered Object Detection** – Uses **YOLOv8** to identify items in uploaded shelf images.  
- 📊 **Interactive Dashboard** – Built with Streamlit for seamless visualization and control.  
- 📷 **Image-Based Shelf Scanning** – Upload shelf images to automatically detect missing items.  
- 🔄 **Modular Design** – Combines dynamic stock prediction and visual analysis in one app.  
- ⚡ **Real-Time Insights** – Detect anomalies, track stock levels, and visualize performance instantly.  

---

## 📂 Folder Structure

```bash
SMARTSTOCKAI/
├── data/
│   ├── anomalies.csv
│   ├── forecast_pre.csv
│   ├── inventory_data.csv
│   └── product_details.csv
├── forecasts/
├── images/
├── models/
│   ├── anomaly_detector.pkl
│   └── forecast_model.pkl
├── scripts/
│   ├── __pycache__/
│   ├── generate_sample_data.py
│   ├── predict_api.py
│   ├── preprocess.py
│   ├── train_anomaly.py
│   └── train_prophet.py
├── app.py
├── README.md
├── shelf_image.py
├── shelf_monitor.py
├── smartstock_vision_app.py
├── SmartStockAI.ipynb
├── yolov8n.pt
├─ requirements.txt
└─ README.md
````

---

## 🧠 Tech Stack

* **Language:** Python 3.10+
* **Frontend:** Streamlit (interactive web app)
* **ML & Forecasting:** Scikit-learn, NumPy, Pandas
* **Computer Vision:** OpenCV, Ultralytics YOLOv8
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Environment:** Virtualenv / Conda

---

## ⚙️ Installation & Setup

1️⃣ **Clone the Repository**

```bash
git clone https://github.com/harishy0406/SmartStockAI.git
cd SmartStockAI
```

2️⃣ **Create Virtual Environment**

```bash
python -m venv venv
# Activate it
venv\Scripts\activate       # (Windows)
source venv/bin/activate    # (Linux/Mac)
```

3️⃣ **Install Required Libraries**

```bash
pip install -r requirements.txt
```

4️⃣ **Run the Streamlit App**

* For **SmartStockAI Dashboard**

```bash
streamlit run app.py
```

* For **Shelf Image Detection**

```bash
streamlit run app_upload.py
```

---

## 📸 Example Outputs

### 🖼️ 1. SmartStock AI Dynamic Dashboard

<img width="1919" height="801" alt="image" src="https://github.com/user-attachments/assets/96480886-8e19-4130-ab0f-a364472048df" />


### 🤖 2. Shelf Detection Results

<img width="1828" height="742" alt="image" src="https://github.com/user-attachments/assets/576f13a2-0590-424e-b061-ba73c522b048" />


### 📈 3. Forecast Visualization

<img width="1665" height="824" alt="image" src="https://github.com/user-attachments/assets/9b396bae-4316-4a11-a946-3041fa331907" />

<img width="1721" height="515" alt="image" src="https://github.com/user-attachments/assets/28aeace7-9924-40d0-90e8-9fc48d83ef97" />

<img width="816" height="811" alt="image" src="https://github.com/user-attachments/assets/2c00cd61-206b-477f-aa4e-e62b478509b3" />

---

## 🌟 Future Enhancements

* 🔔 **Automated Low-Stock Alerts** via Email or SMS
* ☁️ **Cloud Dashboard** for multi-warehouse integration
* 📹 **Live Shelf Monitoring** using IP cameras
* 🤖 **Vision Transformer Integration (ViT)** for improved recognition
* 🧾 **ERP API Integration** for seamless business workflows

---

## 🤝 Contribution

Want to make **SmartStock AI** even smarter?
Fork the repo, improve features, and create a pull request!
Let’s redefine warehouse intelligence — together 🚀

---
