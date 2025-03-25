import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Load the CSV data
df = pd.read_csv('chargeback-costs-timeseries-data.csv')  # Replace 'your_file.csv' with your actual file name

# Convert the 'ChargeBack_Month' column to datetime format if it's not already
df['ChargeBack_Month'] = pd.to_datetime(df['ChargeBack_Month'])

# Ensure the data is sorted by month
df = df.sort_values('ChargeBack_Month')

# Function to get services by BU_CODE
def get_services_by_bu(bu_code):
    return df[df['BU_CODE'] == bu_code]['Service_Name'].unique()

# Function to forecast for a specific service
def forecast_service(service_name, bu_code):
    service_data = df[(df['Service_Name'] == service_name) & (df['BU_CODE'] == bu_code)].set_index('ChargeBack_Month')

    if 'Monthly_cost' not in service_data.columns:
        print(f"'Monthly_cost' column not found for service: {service_name} with BU_CODE: {bu_code}")
        return None

    if len(service_data) < 12:
        print(f"Not enough data for service: {service_name} with BU_CODE: {bu_code}")
        return None

    # Fit ARIMA model (you can adjust the order as needed)
    model = ARIMA(service_data['Monthly_cost'], order=(2, 1, 2))
    model_fit = model.fit()

    # Forecast for 24 months (2+10 and 4+8)
    forecast_2_10 = model_fit.forecast(steps=12)  # 2+10 forecast
    forecast_4_8 = model_fit.forecast(steps=12)   # 4+8 forecast (you can adjust the logic for different forecasts if needed)
    print(f"2 + 10 Forecast for {service_name} with BU_CODE: {bu_code}")
    print(forecast_2_10)
    print(f"4 + 8 Forecast for {service_name} with BU_CODE: {bu_code}")
    print(forecast_4_8)
    # Combine forecasts into a DataFrame
    forecast_index = pd.date_range(start=service_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
    forecast_df = pd.DataFrame({
        '2+10 Forecast': forecast_2_10,
        '4+8 Forecast': forecast_4_8
    }, index=forecast_index)

    return forecast_df

# Main function to generate forecasts for all services under a BU_CODE
def generate_forecasts_for_bu(bu_code):
    services = get_services_by_bu(bu_code)
    all_forecasts = []

    if len(services) == 0:
        print(f"No services found for BU_CODE: {bu_code}")
        return

    for service in services:
        print(f"Generating forecast for service: {service} under BU_CODE: {bu_code}")
        forecast_df = forecast_service(service, bu_code)
        if forecast_df is not None:
            forecast_df['Service_Name'] = service  # Add service name for identification
            all_forecasts.append(forecast_df)

    # Combine all forecasts into a single DataFrame
    if all_forecasts:
        combined_forecasts = pd.concat(all_forecasts)
        #combined_forecasts.to_csv(f'forecasted_values_{bu_code}.csv', index=True)
        print(f"Forecasts saved to 'forecasted_values_{bu_code}.csv'")
    else:
        print("No forecasts generated.")

# Example usage
bu_code_input = 'AIDE_0075121'  # Replace with actual BU_CODE
generate_forecasts_for_bu(bu_code_input)
