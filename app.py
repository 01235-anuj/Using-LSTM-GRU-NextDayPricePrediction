import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Barclays Stock Predictor")

st.title("📈 Barclays Stock Price Prediction using LSTM & GRU")
st.write("This app compares LSTM and GRU models to predict Barclays' next-day stock price.")

@st.cache_data
def load_data():
    data = yf.download("BARC.L", start="2010-01-01", end="2023-12-31")
    return data

data = load_data()
st.subheader("Historical Stock Price")
st.line_chart(data['Close'])

try:
    lstm_model = load_model('lstm_model.h5')
    gru_model = load_model('gru_model.h5')
except:
    st.error("Model files not found. Please upload 'lstm_model.h5' and 'gru_model.h5'")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

X_test = []
for i in range(60, len(scaled_data)):
    X_test.append(scaled_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

if st.button("Predict Next Day Closing Price"):
    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred)
    gru_pred = scaler.inverse_transform(gru_pred)
    st.write(f"🔮 LSTM Prediction: £{lstm_pred[-1][0]:.2f}")
    st.write(f"🔮 GRU Prediction: £{gru_pred[-1][0]:.2f}")
    st.line_chart({
        'LSTM Prediction': lstm_pred.flatten()[-100:], 
        'GRU Prediction': gru_pred.flatten()[-100:], 
        'Actual': data['Close'].values[-100:]
    })