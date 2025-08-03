
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("📈 Barclays Stock Price Prediction: LSTM vs GRU")
st.markdown("Upload stock price data and get **next day's prediction** using trained deep learning models.")

# Load models and scaler
with open('lstm_model.pkl', 'rb') as f:
    lstm_model = pickle.load(f)

with open('gru_model.pkl', 'rb') as f:
    gru_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# File upload
uploaded_file = st.file_uploader("Upload CSV File with Historical Stock Data", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", data.tail())

    # Preprocessing (assume 'Close' column exists)
    close_prices = data['Close'].values.reshape(-1, 1)
    scaled_close = scaler.transform(close_prices)

    if len(scaled_close) >= 60:
        last_60 = scaled_close[-60:].reshape(1, 60, 1)

        lstm_pred = lstm_model.predict(last_60)
        gru_pred = gru_model.predict(last_60)

        lstm_price = scaler.inverse_transform(lstm_pred)[0][0]
        gru_price = scaler.inverse_transform(gru_pred)[0][0]

        st.subheader("📊 Predicted Next Day Closing Price")
        st.write(f"**LSTM**: ₹{lstm_price:.2f}")
        st.write(f"**GRU**: ₹{gru_price:.2f}")
    else:
        st.error("❗ Need at least 60 rows of data to make a prediction.")
