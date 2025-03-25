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
    return data[['Close', 'Volume']]

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
        y.append(data[i + time_step, 0])  # Predict the 'Close' price
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
    ticker = "AAPL"  # Apple stock as an example
    start_date = "2015-01-01"
    end_date = "2025-01-06"
    
    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)
    data = data.dropna()

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
        LSTM(50, return_sequences=True, input_shape=(time_step, 2)),  # Adjusted for 2 features
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
        np.hstack((predictions, np.zeros((predictions.shape[0], 1))))  # Add dummy volume column for inverse scaling
    )[:, 0]  # Extract the scaled 'Close' column
    
    # Prepare true data
    true_data = scaler.inverse_transform(test_data)[time_step:, 0]
    
    # Plot predictions vs actual
    plot_predictions(true_data, predictions)
