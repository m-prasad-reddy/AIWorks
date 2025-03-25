import sqlsrvapp as sqs
import pandas as pd

# SERVER = 'localhost'
# DATABASE = 'adventureworks'
# USERNAME = 'sa'
# PASSWORD = 'S0rry!43'
# connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}'

engine = sqs.mssql_engine()

SQL_QUERY = """
SELECT 
TOP 5 c.CustomerID, 
c.CompanyName, 
COUNT(soh.SalesOrderID) AS OrderCount 
FROM 
SalesLT.Customer AS c 
LEFT OUTER JOIN SalesLT.SalesOrderHeader AS soh ON c.CustomerID = soh.CustomerID 
GROUP BY 
c.CustomerID, 
c.CompanyName 
ORDER BY 
OrderCount DESC;
"""

# cursor = conn.cursor()
# cursor.execute(SQL_QUERY)

# records = cursor.fetchall()
# for r in records:
#     print(f"{r.CustomerID}\t{r.OrderCount}\t{r.CompanyName}")

df = pd.read_sql(SQL_QUERY, engine)
print(df)