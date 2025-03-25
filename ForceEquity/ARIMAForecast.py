import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the dataset
file_path = "ForecastData-2025-2026.csv"
data = pd.read_csv(file_path)

# Convert the 'CB Month' column to datetime
data['CB Month'] = pd.to_datetime(data['CB Month'], format='%Y-%m')

# Clean the cost columns (remove $ and commas, convert to float)
data['Decom-Cost'] = data['Decom-Cost'].replace({'\$': '', ',': ''}, regex=True).astype(float)
data['DA-UHC-Cost'] = data['DA-UHC-Cost'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Set the 'CB Month' as the index
data.set_index('CB Month', inplace=True)

def arima_forecast(series, train_period, forecast_horizon):
    # Train-test split
    train = series[:train_period]

    # Fit the ARIMA model
    model = ARIMA(train, order=(1, 1, 1))
    fit = model.fit()

    # Forecast
    forecast = fit.forecast(steps=forecast_horizon)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(series, label='Actual Data')
    plt.plot(pd.date_range(start=train.index[-1] + pd.DateOffset(1), periods=forecast_horizon, freq='MS'),
             forecast, label='Forecast', color='red')
    plt.axvline(x=train.index[-1], color='gray', linestyle='--', label='Training/Test Split')
    plt.title("ARIMA Forecasting")
    plt.legend()
    plt.show()

    return forecast

# Forecasting for Decom-Cost
decom_cost_series = data['DA-UHC-Cost']
forecast_horizon = 10  # Forecasting for 10 months
train_period = len(decom_cost_series) - forecast_horizon  # Use the first 14 months for training

# Run ARIMA forecasting
arima_forecast_decom_cost = arima_forecast(
    decom_cost_series, train_period=train_period, forecast_horizon=forecast_horizon
)

# Print the forecasted values
print(arima_forecast_decom_cost)

