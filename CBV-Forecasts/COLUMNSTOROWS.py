import pandas as pd

# Load the CSV data
df = pd.read_csv('ROW_COLUMN_DATA.csv')  # Replace 'your_file.csv' with your actual file name

# Specify the columns to transpose (monthly costs)
monthly_columns = [
    '2023-01', '2023-02', '2023-03', '2023-04', '2023-05',
    '2023-06', '2023-07', '2023-08', '2023-09', '2023-10',
    '2023-11', '2023-12', '2024-01', '2024-02', '2024-03',
    '2024-04', '2024-05', '2024-06', '2024-07', '2024-08',
    '2024-09', '2024-10', '2024-11', '2024-12'
]

# Melt the DataFrame to transpose the monthly columns
df_melted = df.melt(id_vars=['ASK_ID', 'ServiceName'], 
                     value_vars=monthly_columns, 
                     var_name='Monthly Time Series', 
                     value_name='Monthly Cost')

# Display the transposed DataFrame
print(df_melted)

# Optionally, save the transposed DataFrame to a new CSV file
df_melted.to_csv('transposed_data.csv', index=False)  # Adjust the file name as needed


# Melt the DataFrame to transpose the monthly columns
df_melted = df.melt(id_vars=['ASK_ID', 'ServiceName'], 
                     value_vars=monthly_columns, 
                     var_name='Monthly Time Series', 
                     value_name='Monthly Cost')

# Fill blank values in the Monthly Cost column with $0.00
df_melted['Monthly Cost'] = df_melted['Monthly Cost'].fillna('0.00')

# Sort the melted DataFrame by ASK_ID, ServiceName, Monthly Time Series
df_sorted = df_melted.sort_values(by=['ASK_ID', 'ServiceName', 'Monthly Time Series'])

# Display the sorted DataFrame
print(df_sorted)

# Optionally, save the sorted transposed DataFrame to a new CSV file
df_sorted.to_csv('sorted_transposed_data.csv', index=False)  # Adjust the file name as needed