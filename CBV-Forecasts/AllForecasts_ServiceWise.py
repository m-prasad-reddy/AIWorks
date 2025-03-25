import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Load the CSV data
df = pd.read_csv('chargeback-costs-timeseries-data.csv')  # Replace 'your_file.csv' with your actual file name

# Convert the 'ChargeBack_Month' column to datetime format if it's not already
df['ChargeBack_Month'] = pd.to_datetime(df['ChargeBack_Month'])

# Ensure the data is sorted by month
df = df.sort_values('ChargeBack_Month')

# Function to filter services by BU_CODE
def get_services_by_bu(bu_code):
    return df[df['BU_CODE'] == bu_code]['Service_Name'].unique()

# Method to forecast using ARIMA
def forecast_arima(service_name, bu_code):
    service_data = df[(df['Service_Name'] == service_name) & (df['BU_CODE'] == bu_code)].set_index('ChargeBack_Month')
    
    if len(service_data) < 12 or 'Monthly_cost' not in service_data.columns:
        print(f"Insufficient data for ARIMA on service: {service_name} with BU_CODE: {bu_code}")
        return None

    # Fit ARIMA model
    model = ARIMA(service_data['Monthly_cost'], order=(2, 1, 2))
    model_fit = model.fit()

    # Forecast for 24 months
    forecast_2_10 = model_fit.forecast(steps=12)  # 2+10 forecast
    forecast_4_8 = model_fit.forecast(steps=12)   # 4+8 forecast (you can adjust the logic if needed)

    # Create a DataFrame for the results
    forecast_index = pd.date_range(start=service_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
    forecast_df = pd.DataFrame({
        '2+10 Forecast (ARIMA)': forecast_2_10,
        '4+8 Forecast (ARIMA)': forecast_4_8
    }, index=forecast_index)

    return forecast_df

# Method to forecast using SARIMA (ignore non-suitable services)
def forecast_sarima(service_name, bu_code):
    service_data = df[(df['Service_Name'] == service_name) & (df['BU_CODE'] == bu_code)].set_index('ChargeBack_Month')
    
    if len(service_data) < 12 or 'Monthly_cost' not in service_data.columns:
        print(f"Insufficient data for SARIMA on service: {service_name} with BU_CODE: {bu_code}")
        return None

    # Fit SARIMA model; you may need to adjust the order for your data
    model = SARIMAX(service_data['Monthly_cost'], order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Forecast for 24 months
    forecast_2_10 = model_fit.forecast(steps=12)  # 2+10 forecast
    forecast_4_8 = model_fit.forecast(steps=12)   # 4+8 forecast (you can adjust the logic if needed)

    # Create a DataFrame for the results
    forecast_index = pd.date_range(start=service_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
    forecast_df = pd.DataFrame({
        '2+10 Forecast (SARIMA)': forecast_2_10,
        '4+8 Forecast (SARIMA)': forecast_4_8
    }, index=forecast_index)

    return forecast_df

# Method to generate forecasts for all services under a BU_CODE
def generate_forecasts_for_bu(bu_code):
    services = get_services_by_bu(bu_code)
    all_forecasts_arima = []
    all_forecasts_sarima = []

    if len(services) == 0:
        print(f"No services found for BU_CODE: {bu_code}")
        return

    for service in services:
        print(f"Generating forecasts for service: {service} under BU_CODE: {bu_code}")

        # ARIMA Forecast
        arima_forecast_df = forecast_arima(service, bu_code)
        if arima_forecast_df is not None:
            arima_forecast_df['Service_Name'] = service
            all_forecasts_arima.append(arima_forecast_df)

        # SARIMA Forecast
        sarima_forecast_df = forecast_sarima(service, bu_code)
        if sarima_forecast_df is not None:
            sarima_forecast_df['Service_Name'] = service
            all_forecasts_sarima.append(sarima_forecast_df)

    # Combine all forecasts into a single DataFrame
    if all_forecasts_arima:
        combined_forecasts_arima = pd.concat(all_forecasts_arima)
        print(f"ARIMA forecasts for BU_CODE: {bu_code}")
        print(combined_forecasts_arima)
        combined_forecasts_arima.to_csv(f'forecasted_values_arima_{bu_code}.csv', index=True)
        print(f"ARIMA forecasts saved to 'forecasted_values_arima_{bu_code}.csv'")

    if all_forecasts_sarima:
        combined_forecasts_sarima = pd.concat(all_forecasts_sarima)
        print(f"SARIMA forecasts for BU_CODE: {bu_code}")
        print(combined_forecasts_sarima)
        combined_forecasts_sarima.to_csv(f'forecasted_values_sarima_{bu_code}.csv', index=True)
        print(f"SARIMA forecasts saved to 'forecasted_values_sarima_{bu_code}.csv'")

# Method to plot the forecasts
def plot_forecasts(bu_code):
    services = get_services_by_bu(bu_code)

    if len(services) == 0:
        print(f"No services found for BU_CODE: {bu_code}")
        return

    for service in services:
        arima_forecast_df = forecast_arima(service, bu_code)
        sarima_forecast_df = forecast_sarima(service, bu_code)

        plt.figure(figsize=(12, 6))
        plt.plot(df[(df['Service_Name'] == service) & (df['BU_CODE'] == bu_code)].set_index('ChargeBack_Month')['Monthly_cost'], label='Historical Costs', color='blue')

        if arima_forecast_df is not None:
            plt.plot(arima_forecast_df.index, arima_forecast_df['2+10 Forecast (ARIMA)'], label='2+10 Forecast (ARIMA)', linestyle='--', color='orange')
            plt.plot(arima_forecast_df.index, arima_forecast_df['4+8 Forecast (ARIMA)'], label='4+8 Forecast (ARIMA)', linestyle='--', color='red')

        if sarima_forecast_df is not None:
            plt.plot(sarima_forecast_df.index, sarima_forecast_df['2+10 Forecast (SARIMA)'], label='2+10 Forecast (SARIMA)', linestyle=':', color='green')
            plt.plot(sarima_forecast_df.index, sarima_forecast_df['4+8 Forecast (SARIMA)'], label='4+8 Forecast (SARIMA)', linestyle=':', color='purple')

        plt.title(f'Forecast for {service} with BU_CODE: {bu_code}')
        plt.xlabel('Month')
        plt.ylabel('Monthly Cost')
        plt.legend()
        plt.show()

# Example usage
bu_code_input = 'AIDE_0075121'  # Replace with actual BU_CODE
generate_forecasts_for_bu(bu_code_input)
plot_forecasts(bu_code_input)