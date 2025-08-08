
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load your saved model predictions and data
lstm_pred = joblib.load('lstm_predictions.pkl')
gru_pred = joblib.load('gru_predictions.pkl')
data = pd.read_csv('stock_data.csv')  # Ensure this has 'Date' and 'Close' columns

# Convert 'Date' to datetime and set as index for proper time series plotting
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Title and description
st.title("LSTM vs GRU Stock Price Prediction")
st.write("Visual comparison of LSTM and GRU models vs Actual stock prices")

# Extract last 100 data points
lstm_series = lstm_pred.flatten()[-100:]
gru_series = gru_pred.flatten()[-100:]
actual_series = data['Close'].values[-100:]

# Prepare DataFrame for visualization
chart_data = pd.DataFrame({
    'LSTM Prediction': lstm_series,
    'GRU Prediction': gru_series,
    'Actual': actual_series
}, index=data.index[-100:])

# Show line chart
st.line_chart(chart_data)

# Optional: display table
if st.checkbox("Show Prediction Table"):
    st.dataframe(chart_data)
