import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Fetch data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Add technical indicators
def add_technical_indicators(data):
    # Simple Moving Average (SMA)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Fill NA values after rolling calculations
    data.fillna(method='bfill', inplace=True)
    return data

# Data preprocessing
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create datasets
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i + time_step, :])
        y.append(data[i + time_step, 0])  # Predict 'Close'
    return np.array(X), np.array(y)

# Plot predictions
def plot_predictions(true_data, predicted_data):
    plt.figure(figsize=(10, 6))
    plt.plot(true_data, label='Actual Price')
    plt.plot(predicted_data, label='Predicted Price')
    plt.legend()
    plt.show()

# Main script
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = "2025-02-10"
    
    
    # Fetch and preprocess data
    data = fetch_stock_data(ticker, start_date, end_date)
    data = add_technical_indicators(data[['Close', 'Volume']])
    data = data.dropna()  # Drop rows with NaN values
    
    # Preprocess data
    scaled_data, scaler = preprocess_data(data.values)
    
    # Prepare training and testing datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    time_step = 60
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=20, verbose=1)
    
    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.hstack((predictions, np.zeros((predictions.shape[0], X_test.shape[2] - 1))))
    )[:, 0]
    
    # Prepare true data
    true_data = scaler.inverse_transform(test_data)[time_step:, 0]
    
    # Plot predictions vs actual
    plot_predictions(true_data, predictions)
