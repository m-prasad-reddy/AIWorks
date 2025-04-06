import pyodbc

# Config vibes
USERNAME = "sa"
PASSWORD = "S0rry!43"  # Update with your password
SERVER = "localhost"
DATABASE = "BikeStores"

def get_sqlserver_connection():
    """Hook up to SQL Server with smooth energy."""
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}"
        )
        return conn
    except Exception as e:
        raise ConnectionError(f"Connection vibes off: {e}")