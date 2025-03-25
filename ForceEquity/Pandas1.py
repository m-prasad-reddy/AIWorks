import pandas as pd
import numpy as np

# Creating a series
data1 = [1,2,3,4,5]
s = pd.Series(data1)
#print(s)

# Creating a dataframe
data2 = [['Alex',10],['Bob',12],['Clarke',13]]
df1 = pd.DataFrame(data2,columns=['Name','Age'])
#print(df1)

# Creating a dataframe from dictionary
data3 = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df2 = pd.DataFrame(data3)
#print(df2)

#print(df2['Age'].min()) 
#print(df2['Name'])

df = pd.read_csv('ForJson_CBV-2023-2024.csv', encoding='latin1')
# print(df.info())  # Summary of the DataFrame
# print(df.describe())  # Statistical summary
#print(df['2023-11'].dtype)
#print(df.iloc[0:5])  # First 5 rows
#print(df.loc[0:10, ['Bill Name', '2023-11']])  # First 10 rows of CBV and 2023 columns
#print(df[['Bill Name','2023-11']])
#print(df[df['Bill Name']=='BDPaaS Storage'])
#df.replace(to_replace=np.nan, value=0) # this wont replace the NaN values in the dataframe
#print(df[['Bill Name','2023-11']].replace(to_replace=np.nan, value=0))
df.iloc[:, 1:] = df.iloc[:, 1:].replace(to_replace=np.nan, value=0) # this will replace the NaN values in the dataframe except column ['Bill Name]
df.iloc[:, 1:] = df.iloc[:, 1:].replace(r'[\$,]', '', regex=True).astype(float) 
print(df)
df.to_csv('output.csv', encoding='utf-8', index=False)

