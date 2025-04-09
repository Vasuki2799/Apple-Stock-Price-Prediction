# 📈 Apple Stock Price Prediction using Deep Learning

Welcome to the **Apple Stock Price Prediction** project!  
This project explores and compares multiple deep learning models including **RNN**, **LSTM**, **GRU**, and **Transformer** to predict stock prices using historical data. It also features an interactive **Streamlit web app** for predictions and model comparisons.

---

<img width="1440" alt="Screenshot 2025-04-09 at 9 59 59 PM" src="https://github.com/user-attachments/assets/b6d6988f-8418-4166-9126-88128f3366c3" />

## 🔍 Project Overview

This project aims to:
- Forecast Apple Inc. stock closing prices using time series data.
- Compare the performance of different deep learning models.
- Build a Streamlit dashboard for real-time predictions and model evaluation.

---

## 📌 Features of This Project

- 📊 Upload your own `.csv` file to make predictions.
- 📉 Predict stock prices using 4 advanced models.
- 📋 View a side-by-side comparison of model performances.
- 🎨 Stylish and user-friendly UI with color themes and dynamic content.

---

## 🛠 Technologies & Tools Used

| Category          | Tools/Libraries                                       |
|------------------|--------------------------------------------------------|
| **Languages**     | Python                                                 |
| **IDE**           | Google Colab, Visual Studio Code                      |
| **Libraries**     | TensorFlow/Keras, NumPy, Pandas, Scikit-learn, Matplotlib |
| **Visualization** | Streamlit                                             |
| **Model Types**   | RNN, LSTM, GRU, Transformer                           |

---

## 🔄 How This Project Was Developed

### 1. Data Preprocessing
- Used Apple historical stock data.
- Normalized data using `MinMaxScaler`.
- Created sequences for time series prediction (60 time steps).

### 2. Model Training (in Colab)
- Trained 4 separate deep learning models:
  - RNN
  - LSTM
  - GRU
  - Transformer
- Saved models in `.keras` format.

### 3. Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### 4. Streamlit Application
- Interactive UI for uploading CSV and making predictions.
- Visualization of last 60 days and predicted stock price.
- Comparison section with evaluation metrics and observations.

---

## 🧪 Sample Model Comparison Table

| Model       | MSE     | RMSE   | MAE    | R² Score |
|-------------|---------|--------|--------|----------|
| RNN         | 0.0023  | 0.048  | 0.035  | 0.92     |
| LSTM        | 0.0018  | 0.042  | 0.031  | 0.94     |
| GRU         | 0.0015  | 0.039  | 0.028  | 0.95     |
| Transformer | 0.0012  | 0.035  | 0.025  | 0.97     |

---

## 🎬 Demo Video
https://www.linkedin.com/posts/vasuki27_datascience-deeplearning-stockprediction-activity-7315775758323310592-knVV?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFWofHABP5vZ1q4SVksdeQ_qxpl9ilnOKXM

## 👩‍💻 Developed By

Name: VASUKI ARUL

Batch Code: DS-C-WD-E-B29

LinkedIn: https://www.linkedin.com/in/vasuki27/

