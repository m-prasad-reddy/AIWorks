import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Sample data (replace with your actual data)
data = {
    "BDPaaS Compute": [4230.00, 4516.70, 5088.34, 5088.34, 5088.34, 5088.34, 5088.34, 5088.34, 5088.34, 5088.34, 4924.20, 5088.34, 5088.34, 5088.34],
    "BDPaaS Storage": [1651.99, 1707.06, 2078.16, 2078.16, 2078.16, 2078.16, 2078.16, 2078.16, 2078.16, 2220.94, 2175.87, 2248.40, 2535.06, 2535.06],
    "ECE Elastic": [5612.55, 5612.55, 9179.15, 9179.15, 9179.15, 9179.15, 9179.15, 10043.07, 10043.07, 10043.07, 10043.07, 10043.07, 10043.07, 10043.07],
    "HCC Kubernetes_UoM 2023": [651.44, 624.17, 887.89, 1478.11, 2054.27, 1896.02, 1771.51, 1712.31, 1638.57, 1981.20, 3077.28, 2870.58, 1676.31, 1370.94],
    "RedHat LINUX 8 - ODI": [1171.82, 400.51, 428.78, 454.24, 436.36, 456.52, 445.22, 456.94, 478.67, 423.81, 356.27, 406.37, 347.47, 348.11],
    "Storage - Object": [9002.05, 8470.15, 6232.28, 6984.80, 5833.05, 6446.64, 6038.73, 6494.72, 6319.38, 7535.30, 8122.64, 7888.75, 7898.70, 8014.76]
}

# Function to forecast using the Holt-Winters method
def forecast(data, method='2+10'):
    forecasts = {}
    for bill_name, values in data.items():
        # Create a DataFrame
        df = pd.DataFrame(values, columns=['Values'])
        
        # Apply the Holt-Winters method
        model = ExponentialSmoothing(df['Values'], seasonal='add', seasonal_periods=12).fit()
        
        # Forecast for 2025 and 2026
        if method == '2+10':
            forecasted_values = model.forecast(24)  # 2 years
        elif method == '8+4':
            forecasted_values = model.forecast(12)  # 1 year
        else:
            raise ValueError("Unknown method. Please use '2+10' or '8+4'.")
        
        forecasts[bill_name] = forecasted_values.tolist()
    
    return forecasts

# Generate forecasts
forecast_2_10 = forecast(data, method='2+10')
forecast_8_4 = forecast(data, method='8+4')

# Display results
print("Forecast using the 2+10 method:")
for bill, values in forecast_2_10.items():
    print(f"{bill}: {values}")

print("\nForecast using the 8+4 method:")
for bill, values in forecast_8_4.items():
    print(f"{bill}: {values}")
