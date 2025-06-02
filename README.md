# Stock-Price-Prediction-using-LSTM-and-Yahoo-Finance

# 📈 Stock Price Prediction using LSTM and Yahoo Finance

This project is a machine learning pipeline for predicting **daily stock closing prices** using an LSTM (Long Short-Term Memory) model trained on historical data from [Yahoo Finance](https://finance.yahoo.com/).

Built in Python using:
- `TensorFlow/Keras` for modeling
- `yfinance` for live stock data
- `scikit-learn` for preprocessing

---

## 📊 Features

- Predicts today's stock **closing price** using recent historical data
- Uses features like: `Open`, `High`, `Low`, `Volume`, and `Previous Close`
- Supports scaling, sequence creation, and live predictions
- Easy to adapt for multiple stocks (e.g., AAPL, MSFT, TSLA)
- Logs predictions to a CSV file for tracking performance

---

## 🧠 Model

- **Model type**: Stacked LSTM
- **Input**: Last `n_steps` days of stock data
- **Output**: Predicted Close price
- **Loss function**: Mean Squared Error (MSE)

---

## 📦 Installation

Install all required libraries using:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib



