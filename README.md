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


MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

