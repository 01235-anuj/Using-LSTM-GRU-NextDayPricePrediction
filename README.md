# 📈 Next Day Stock Price Prediction using LSTM and GRU

This project compares **LSTM** and **GRU** deep learning models to predict the **next day's closing stock price of Barclays (BCS)** using historical data fetched via the `yfinance` API.

---

## 📌 Problem Statement

Accurately forecasting stock prices is a critical task in the financial industry. Traditional methods often struggle with time-series dependencies, noise, and volatility. This project aims to tackle this by applying **Recurrent Neural Network (RNN)** based models — specifically **LSTM** and **GRU** — which are well-suited for sequence prediction problems.

The goal is to predict the **next day's closing price** using past closing prices and compare which architecture performs better in terms of accuracy and generalization.

---

## 📊 Data Source

We use **Yahoo Finance** to download daily historical stock data for **Barclays PLC (BCS)** using the `yfinance` Python package.

Key attributes:
- **Ticker**: `BCS`
- **Features used**: `Close` price only
- **Timeframe**: Last several years of daily data
- **Data Preprocessing**:
  - Normalization using MinMaxScaler
  - Sliding window technique to create sequences of past prices

---

## 🧠 Models Used

Two RNN-based architectures are implemented and compared:

### 🔷 LSTM (Long Short-Term Memory)
- Capable of learning long-term dependencies
- Avoids vanishing gradient problems
- Widely used in time-series forecasting

### 🔷 GRU (Gated Recurrent Unit)
- Simplified version of LSTM with fewer parameters
- Faster training time
- Often performs comparably to LSTM with similar accuracy

Both models are built using **TensorFlow/Keras**, and trained using the same dataset, same sequence lengths, and same epochs for fair comparison.

---

## ⚙️ Model Architecture

- **Input Layer**: Sliding window sequences (e.g., 60 timesteps)
- **Hidden Layers**:
  - LSTM or GRU with 50–100 units
  - Dropout for regularization
- **Dense Output Layer**: 1 neuron (predicting next day's closing price)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

---

## 📈 Evaluation Metrics

We evaluate and compare both models based on:

- **Training and Validation Loss Curves**
- **RMSE (Root Mean Squared Error)** — lower is better
- **Visual Comparison** of:
  - Predicted vs Actual closing prices
  - Time-series plots for clarity

### 📷 Predicted vs Actual Closing Price

<img width="593" height="318" alt="image" src="https://github.com/user-attachments/assets/d2facade-ab18-48ae-a26a-702c5a02676d" />


---


## 📁 Project Structure
```Stock-Price-Prediction-LSTM-GRU/
├── notebook/
│ ├── A_Comparison_of_LSTM_and_GRU_Models_for_Predicting_Next_Day_Barclays_Stock_Price.ipynb
│ └── closing_price_comparison.png
├── README.md
├── requirements.txt
└── .gitignore```

