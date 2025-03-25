import pandas as pd
from sqlalchemy import create_engine
import pyodbc
import os

SERVER = 'localhost'
DATABASE = 'salesdb'
USERNAME = 'sa'
PASSWORD = ''
def get_connection_uri():
    return f'mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server'

def get_sql_connection():
    connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}'
    return pyodbc.connect(connection_string)

def dynamic_mssql_engine(user = os.getenv('user'), password = os.getenv('password')
                 ,host = os.getenv('SERVER_ADDRESS'),db = os.getenv('DATABASE')):
    engine = create_engine(f'mssql+pyodbc://{user}:{password}@{host}/{db}?driver=SQL+Server')
    return engine

def mssql_engine(user = USERNAME, password = PASSWORD
                 ,host = SERVER,db = DATABASE):
    #engine = create_engine(f'mssql+pyodbc://{user}:{password}@{host}/{db}?driver=SQL+Server')
    engine = create_engine(f'mssql+pyodbc://{user}:{password}@{host}/{db}?driver=ODBC+Driver+17+for+SQL+Server')
    return engine
