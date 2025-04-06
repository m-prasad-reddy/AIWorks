import pyodbc

# Config vibes
SERVER = "localhost"
DATABASE = "BikeStores"  # Swapped from BikeStores per your code
USER = "sa"
PASSWORD = "S0rry!43"  # Update with your password

def get_sqlserver_connection():
    """Hook up to SQL Server with chill energy."""
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SERVER};DATABASE={DATABASE};UID={USER};PWD={PASSWORD}"
        )
        return conn
    except Exception as e:
        raise ConnectionError(f"Connection vibes off: {e}")