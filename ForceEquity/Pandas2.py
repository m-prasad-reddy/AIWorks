import pandas as pd

# Create the initial DataFrame
data = {
    'Bill Name': ['BDPaaS Compute', 'BDPaaS Storage', 'Compute - UBUNTU', 'DBaaS MS SQL - SQL 2019', 'ECE Elastic', 
                  'EOL - RedHat LINUX 6 - ODI', 'EOL - RedHat LINUX 7 - ODI', 'Github Enterprise Cloud', 
                  'HCC Kubernetes_UoM 2023', 'MySQL 8 ODI_CPU', 'MySQL 8.0_CPU', 'NAS Nearstore', 'ODI CPU', 
                  'OpenShift Enterprise', 'Optum Functions Platform', 'OSFI - Kubernetes', 'OSFI - Origins', 
                  'Public Cloud - GCP Usage', 'RedHat LINUX 6 - ODI', 'RedHat LINUX 7', 'RedHat LINUX 7 - ODI', 
                  'RedHat LINUX 8 - ODI', 'Search Analytics', 'Shared Storage', 'SQL 2016 UCI_CPU', 
                  'Standard Performance NAS', 'Standard Performance Storage', 'Storage - Object', 'TSM'],
    '2023-11': [4230.0, 1651.99, 0.0, 36.05, 5612.55, 0.0, 0.0, 0.04, 651.44, 501.31, 0.0, 1465.3, 2965.91, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 18382.54, 1171.82, 6.7, 0.0, 0.0, 0.0, 7648.6, 9002.05, 6.58],
    '2023-12': [4516.7, 1707.06, 0.0, 36.05, 5612.55, 0.0, 0.0, 0.0, 624.17, 18.72, 0.0, 1484.73, 2116.8, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 18301.14, 400.51, 16.75, 0.0, 0.0, 0.0, 7663.6, 8470.15, 6.58],
    '2024-01': [5088.34, 2078.16, 0.0, 39.86, 9179.15, 0.0, 2523.65, 0.0, 887.89, 28.27, 0.0, 1088.44, 3333.12, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7800.75, 428.78, 19.26, 0.0, 0.0, 5846.44, 0.0, 6232.28, 4.37],
    '2024-02': [5088.34, 2078.16, 0.0, 39.86, 9179.15, 0.0, 1428.77, 0.0, 1478.11, 28.04, 0.0, 1104.86, 1794.24, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2578.2, 454.24, 19.26, 0.0, 0.0, 5853.84, 0.0, 6984.8, 0.65],
    '2024-03': [5088.34, 2078.16, 0.0, 39.86, 9179.15, 0.0, 1245.44, 0.0, 2054.27, 0.0, 0.0, 1066.46, 1467.84, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1631.6, 436.36, 12.84, 0.0, 0.0, 6170.56, 0.0, 5833.05, 0.65], 
    '2024-04': [5088.34, 2078.16, 0.0, 39.86, 9179.15, 0.0, 1331.46, 0.0, 1896.02, 0.0, 0.0, 1014.75, 1569.22, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1739.12, 456.52, 32.1, 0.0, 0.0, 6170.56, 0.0, 6446.64, 0.0], 
    '2024-05': [5088.34, 2078.16, 0.0, 39.86, 9179.15, 0.0, 1278.46, 14.94, 1771.51, 0.0, 0.0, 978.77, 1506.56, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1672.87, 445.22, 19.26, 0.0, 0.0, 6295.62, 0.0, 6038.73, 0.0], 
    '2024-06': [5088.34, 2078.16, 0.0, 39.86, 10043.07, 0.0, 1333.25, 7.4, 1712.31, 0.0, 0.0, 989.64, 1571.33, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1741.36, 456.94, 19.26, 0.0, 0.0, 5896.02, 0.0, 6494.72, 0.0], 
    '2024-07': [5088.34, 2078.16, 0.0, 39.86, 10043.07, 0.0, 1247.23, 7.55, 1638.57, 0.0, 0.0, 1061.22, 1429.38, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1633.84, 478.67, 12.84, 0.0, 0.0, 5896.02, 0.0, 6319.38, 0.0], 
    '2024-08': [5088.34, 2220.94, 0.0, 39.86, 10043.07, 0.0, 1333.25, 21.27, 1981.2, 0.0, 0.0, 1001.69, 1476.1, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1741.36, 423.81, 12.84, 0.0, 0.0, 5859.02, 0.0, 7535.3, 1.07],
    '2024-09': [4924.2,2175.87,0.0,39.86,10043.07,0.0,952.32,17.62,3077.28,0.0,0.0,953.4,1095.17,0.0,0.0,0.0,0.0,
                0.0,0.0,0.0,1243.84,356.27,12.84,0.0,0.0,6155.02,0.0,8122.64,0.0], 
    '2024-10': [5088.34, 2248.4, 0.0, 39.86, 10043.07, 0.0, 884.48, 15.61, 2870.58, 0.0, 0.0, 1033.39, 1019.01, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1159.04, 406.37, 12.84, 0.0, 0.0, 6021.08, 0.0, 7888.75, 0.0], 
    '2024-11': [5088.34, 2535.06, 0.0, 39.86, 10043.07, 0.0, 0.0, 31.99, 1676.31, 0.0, 0.0, 1052.24, 137.98, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 347.47, 12.84, 0.0, 0.0, 5283.3, 0.0, 7898.7, 0.54], 
    '2024-12': [5088.34, 2535.06, 0.0, 39.86, 10043.07, 0.0, 0.0, 69.38, 1370.94, 0.0, 0.0, 1031.1, 138.24, 
                0.0, 0.0, 0.0, 0.0, 0.0, 5.78, 0.0, 0.0, 348.11, 12.84, 0.0, 0.0, 5320.3, 0.0, 8014.76, 0.0]
    
}

df = pd.DataFrame(data)

# Transform the data to long format
df_long = df.melt(id_vars='Bill Name', var_name='Month', value_name='Chargeback Amount')

# Convert 'Month' column to datetime
df_long['Month'] = pd.to_datetime(df_long['Month'], format='%Y-%m')

# Set index to 'Month' for time series analysis
df_long = df_long.set_index('Month')

# Display the transformed DataFrame
print(df_long)

# Create time series DataFrames for each bill name
time_series_dict = {bill_name: group for bill_name, group in df_long.groupby('Bill Name')}

# Example: Access the time series DataFrame for 'BDPaaS Compute'
bdpaas_compute_df = time_series_dict['BDPaaS Compute']
print(bdpaas_compute_df)
