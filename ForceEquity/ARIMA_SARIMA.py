import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error

 

# Data provided by the user

data = {

    'CB Month': ['2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12',

                 '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12'],

    'Decom-Cost': [93677.19, 107438.39, 89886.18, 73632.50, 73791.22, 49439.28, 47960.91, 44319.42, 41760.47, 38913.68, 37489.41, 37682.63,

                   27152.28, 20668.86, 19014.36, 19747.82, 18983.25, 19515.63, 19022.00, 20669.11, 20983.13, 20450.30, 15896.29, 15804.12],

    'DA-UHC-Cost': [14045.36, 13359.64, 13679.54, 13202.69, 12983.84, 12784.44, 12939.10, 12950.79, 12746.13, 13091.63, 12948.51, 13292.89,

                    17426.48, 17441.67, 17290.24, 17294.07, 17424.21, 17956.04, 17952.13, 18110.01, 18186.28, 18280.53, 18251.44, 18213.68]

}

 

# Convert data to DataFrame

df = pd.DataFrame(data)

df['CB Month'] = pd.to_datetime(df['CB Month'])

df.set_index('CB Month', inplace=True)

 

# Function to fit ARIMA model and forecast

def arima_forecast(series, order=(1,1,1), steps=22):

    model = ARIMA(series, order=order)

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)

    return forecast

 

# Function to fit SARIMA model and forecast

def sarima_forecast(series, order=(1,1,1), seasonal_order=(1,1,1,12), steps=22):

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)

    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=steps)

    return forecast

 

# Forecasting for Decom-Cost

decom_cost_arima_forecast = arima_forecast(df['Decom-Cost'])

decom_cost_sarima_forecast = sarima_forecast(df['Decom-Cost'])

 

# Forecasting for DA-UHC-Cost

da_uhc_cost_arima_forecast = arima_forecast(df['DA-UHC-Cost'])

da_uhc_cost_sarima_forecast = sarima_forecast(df['DA-UHC-Cost'])

 

# Error analysis function

def error_analysis(actual_series, forecast_series):

    mse = mean_squared_error(actual_series[:len(forecast_series)], forecast_series)

    mae = mean_absolute_error(actual_series[:len(forecast_series)], forecast_series)

    return mse, mae

 

# Error analysis for Decom-Cost

decom_cost_arima_mse, decom_cost_arima_mae = error_analysis(df['Decom-Cost'], decom_cost_arima_forecast)

decom_cost_sarima_mse, decom_cost_sarima_mae = error_analysis(df['Decom-Cost'], decom_cost_sarima_forecast)

 

# Error analysis for DA-UHC-Cost

da_uhc_cost_arima_mse, da_uhc_cost_arima_mae = error_analysis(df['DA-UHC-Cost'], da_uhc_cost_arima_forecast)

da_uhc_cost_sarima_mse, da_uhc_cost_sarima_mae = error_analysis(df['DA-UHC-Cost'], da_uhc_cost_sarima_forecast)

 

# Print the results

print("Decom-Cost ARIMA Forecast:")

print(decom_cost_arima_forecast)

print("\nDecom-Cost SARIMA Forecast:")

print(decom_cost_sarima_forecast)

print("\nDA-UHC-Cost ARIMA Forecast:")

print(da_uhc_cost_arima_forecast)

print("\nDA-UHC-Cost SARIMA Forecast:")

print(da_uhc_cost_sarima_forecast)

 

print("\nError Analysis for Decom-Cost:")

print(f"ARIMA MSE: {decom_cost_arima_mse}, MAE: {decom_cost_arima_mae}")

print(f"SARIMA MSE: {decom_cost_sarima_mse}, MAE: {decom_cost_sarima_mae}")

 

print("\nError Analysis for DA-UHC-Cost:")

print(f"ARIMA MSE: {da_uhc_cost_arima_mse}, MAE: {da_uhc_cost_arima_mae}")

print(f"SARIMA MSE: {da_uhc_cost_sarima_mse}, MAE: {da_uhc_cost_sarima_mae}")

 

# Plotting the forecasts

plt.figure(figsize=(14,7))

 

plt.subplot(2,1,1)

plt.plot(df['Decom-Cost'], label='Actual Decom-Cost')

plt.plot(decom_cost_arima_forecast.index[-22:], decom_cost_arima_forecast.values[-22:], label='ARIMA Forecast')

plt.plot(decom_cost_sarima_forecast.index[-22:], decom_cost_sarima_forecast.values[-22:], label='SARIMA Forecast')

plt.title('Decom-Cost Forecast')

plt.legend()

 

plt.subplot(2,1,2)

plt.plot(df['DA-UHC-Cost'], label='Actual DA-UHC-Cost')

plt.plot(da_uhc_cost_arima_forecast.index[-22:], da_uhc_cost_arima_forecast.values[-22:], label='ARIMA Forecast')

plt.plot(da_uhc_cost_sarima_forecast.index[-22:], da_uhc_cost_sarima_forecast.values[-22:], label='SARIMA Forecast')

plt.title('DA-UHC-Cost Forecast')

plt.legend()

 

plt.tight_layout()

plt.show()

 