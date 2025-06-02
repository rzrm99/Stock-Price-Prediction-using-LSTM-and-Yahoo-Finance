import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# --- Configuration ---
ticker = 'NFLX'
n_steps = 30
epochs = 50
batch_size = 8

# --- Step 1: Download Historical Data ---
df = yf.download(ticker, start='2018-01-01')[['Open', 'High', 'Low', 'Close', 'Volume']]

# --- Step 2: Add Previous Close Feature ---
df['Prev_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)  # Remove first row with NaN

# Rearrange columns: [Open, High, Low, Volume, Prev_Close, Close]
df = df[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Close']]

# --- Step 3: Scale Features ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

# --- Step 4: Create Sequences for LSTM ---
def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps, :-1])  # all features except Close
        y.append(data[i + n_steps - 1, -1])  # Close value
    return np.array(X), np.array(y)

X, y = lstm_split(scaled_df.values, n_steps)
split_index = int(len(X) * 0.8)
X_train, y_train = X[:split_index], y[:split_index]

# --- Step 5: Build Model ---
model = Sequential()
model.add(Input(shape=(n_steps, X.shape[2])))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# --- Step 6: Train Model ---
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

# --- Step 7: Predict Today’s Close ---
# Download recent data
end_date = datetime.today()
start_date = end_date - timedelta(days=60)
latest = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))[['Open', 'High', 'Low', 'Close', 'Volume']]

# Add previous close
latest['Prev_Close'] = latest['Close'].shift(1)
latest.dropna(inplace=True)

# Use only last `n_steps` rows
if len(latest) < n_steps:
    raise ValueError(f"Not enough recent data to form prediction window. Required: {n_steps}, Got: {len(latest)}")

latest = latest[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Close']]
latest_scaled = scaler.transform(latest)
X_input = latest_scaled[-n_steps:, :-1].reshape(1, n_steps, -1)

# Predict and inverse transform
pred_scaled = model.predict(X_input)[0, 0]
dummy = np.zeros((1, scaled_df.shape[1]))
dummy[0, -1] = pred_scaled
pred_actual = scaler.inverse_transform(dummy)[0, -1]

# --- Step 8: Output Results ---
print(f"\n📈 Improved Predicted Closing Price for {ticker} Today: ${pred_actual:.2f}")
print(f"🔧 Scaled prediction: {pred_scaled}")
print(f"📉 Inverse transformed prediction (actual): {pred_actual:.2f}")

print("\n📊 Latest data used:")
print(latest.tail())

