# database/connection.py
import pyodbc
from typing import Dict, Optional

class DatabaseConnection:
    def __init__(self):
        self.connection = None
        self.current_config = None

    def connect(self, config: Dict) -> bool:
        try:
            conn_str = (
                f"DRIVER={{{config['driver']}}};"
                f"SERVER={config['server']};"
                f"DATABASE={config['database']};"
                f"UID={config['username']};"
                f"PWD={config['password']}"
            )
            self.connection = pyodbc.connect(conn_str)
            self.current_config = config
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            self.current_config = None

    def is_connected(self) -> bool:
        return self.connection is not None

    def get_cursor(self) -> Optional[pyodbc.Cursor]:
        return self.connection.cursor() if self.connection else None