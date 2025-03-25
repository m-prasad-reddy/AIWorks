import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Method to forecast service-wise along with BU_CODE
def forecast_service_bu_wise(service_name, bu_code):
    # Filter data based on both Service_Name and BU_CODE
    service_bu_data = df[(df['Service_Name'] == service_name) & (df['BU_CODE'] == bu_code)].set_index('ChargeBack_Month')
    
    # Check if there's enough data
    if len(service_bu_data) < 12:
        print(f"Not enough data for service: {service_name} with BU_CODE: {bu_code}")
        return

    # Fit ARIMA model (you can adjust the order as needed)
    model = ARIMA(service_bu_data['Monthly_cost'], order=(2, 1, 2))  # ARIMA(2,1,2)
    model_fit = model.fit()

    # Forecast for 10 months
    forecast_10 = model_fit.forecast(steps=20)
    
    # Forecast for 8 months
    forecast_8 = model_fit.forecast(steps=16)

    # Plotting the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(service_bu_data['Monthly_cost'], label='Historical Costs', color='blue')
    plt.plot(forecast_10.index, forecast_10, label='10-Month Forecast', color='orange')
    plt.plot(forecast_8.index, forecast_8, label='8-Month Forecast', color='green')
    plt.title(f'Forecast for {service_name} with BU_CODE: {bu_code}')
    plt.xlabel('Month')
    plt.ylabel('Monthly_Cost')
    plt.legend()
    plt.show()
    #print(f"2 + 10 Forecast \t {bu_code}\t{service_name}")
    #print(forecast_10)
    #print(f"4 + 8 Forecast \t {bu_code}\t{service_name}")
    print(forecast_8)
    

# Method to forecast BU_CODE wise data (if needed separately)
def forecast_bu_code_wise(bu_code):
    bu_data = df[df['BU_CODE'] == bu_code].set_index('ChargeBack_Month')
    
    # Check if there's enough data
    if len(bu_data) < 12:
        print(f"Not enough data for BU_CODE: {bu_code}")
        return

    # Fit ARIMA model (you can adjust the order as needed)
    model = ARIMA(bu_data['Monthly_cost'], order=(2, 1, 2))  # ARIMA(2,1,2)
    model_fit = model.fit()

    # Forecast for 10 months
    forecast_10 = model_fit.forecast(steps=10)
    
    # Forecast for 8 months
    forecast_8 = model_fit.forecast(steps=8)

    # Plotting the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(bu_data['Monthly_cost'], label='Historical Costs', color='blue')
    plt.plot(forecast_10.index, forecast_10, label='10-Month Forecast', color='orange')
    plt.plot(forecast_8.index, forecast_8, label='8-Month Forecast', color='green')
    plt.title(f'Forecast for BU_CODE: {bu_code}')
    plt.xlabel('Month')
    plt.ylabel('Monthly Cost')
    plt.legend()
    plt.show()

# Example usage for specific service and BU_CODE
#forecast_service_bu_wise('Your_Service_Name', 'Your_BU_Code')  # Replace with actual service name and BU_CODE
#forecast_bu_code_wise('Your_BU_Code')  # Replace with actual BU_CODE if needed
#forecast_service_bu_wise('MySQL Database', 'AIDE_0075121')  # Replace with actual service name and BU_CODE
#forecast_bu_code_wise('AIDE_0075121')  # Replace with actual BU_CODE if needed
#print(get_services_by_bu('AIDE_0075121'))
forecasted_bu='UHGWM110-006878'
forecasted_service='ODI CPU'
#for service in get_services_by_bu(forecasted_bu):
    #print(f"Forecasting for service: {service}")   
    #forecast_service_bu_wise(service, forecasted_bu)

forecast_service_bu_wise(forecasted_service, forecasted_bu)
