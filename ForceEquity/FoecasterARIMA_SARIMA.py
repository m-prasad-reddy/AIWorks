import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Sample data (replace with your actual data)
data = {
    "BDPaaS Compute": [4230.00, 4516.70, 5088.34, 5088.34, 5088.34, 5088.34, 5088.34, 5088.34, 5088.34, 5088.34, 4924.20, 5088.34, 5088.34, 5088.34],
    "BDPaaS Storage": [1651.99, 1707.06, 2078.16, 2078.16, 2078.16, 2078.16, 2078.16, 2078.16, 2078.16, 2220.94, 2175.87, 2248.40, 2535.06, 2535.06],
    "ECE Elastic": [5612.55, 5612.55, 9179.15, 9179.15, 9179.15, 9179.15, 9179.15, 10043.07, 10043.07, 10043.07, 10043.07, 10043.07, 10043.07, 10043.07],
    "HCC Kubernetes_UoM 2023": [651.44, 624.17, 887.89, 1478.11, 2054.27, 1896.02, 1771.51, 1712.31, 1638.57, 1981.20, 3077.28, 2870.58, 1676.31, 1370.94],
    "RedHat LINUX 8 - ODI": [1171.82, 400.51, 428.78, 454.24, 436.36, 456.52, 445.22, 456.94, 478.67, 423.81, 356.27, 406.37, 347.47, 348.11],
    "Storage - Object": [9002.05, 8470.15, 6232.28, 6984.80, 5833.05, 6446.64, 6038.73, 6494.72, 6319.38, 7535.30, 8122.64, 7888.75, 7898.70, 8014.76]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Function to forecast using ARIMA
def forecast_arima(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit.forecast(steps=24)  # Forecast for 2 years

# Function to forecast using SARIMA
def forecast_sarima(series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit.forecast(steps=24)  # Forecast for 2 years

# Example usage
for bill_name, values in df.items():
    print(f"Forecast for {bill_name} using ARIMA:")
    arima_forecast = forecast_arima(values, order=(1, 1, 1))  # Example order
    print(arima_forecast)

    #print(f"Forecast for {bill_name} using SARIMA:")
    #sarima_forecast = forecast_sarima(values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Example seasonal order
    #print(sarima_forecast)
