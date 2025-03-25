import pyodbc
import re
import nltk
import ssl 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sqlalchemy as sa
import sqlsrvapp as sqs
import pandas as pd

#from sqlalchemy import create_engine, inspect

# nltk.download('punkt')
# nltk.download('stopwords')

engine = sqs.mssql_engine()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

def clean_text(text):
    """Lowercase, remove special characters, and tokenize the input query."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)
    return " ".join(tokens)

# Fetch table schema
def get_msqldb_schema_metadata():
    """Extract schema details from the database."""
    query = """
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_NAME, ORDINAL_POSITION;
        """
    conn=sqs.get_sql_connection()
    schema_df = pd.read_sql(query, conn)
    conn.close()

    # Save schema for training
    schema_df.to_csv("adventureworks_schema.csv", index=False)
    print("Schema extracted and saved.")
    return schema_df

def get_schema_metadata():
    """Extract schema details from the database."""
    inspector = sa.inspect(engine)
    schema_info = {}
    
    for table_name in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        schema_info[table_name] = columns

    return schema_info

if __name__ == "__main__":
    sample_query = "Show me all sales orders created after 2007."
    # cleaned_query = clean_text(sample_query)
    cleaned_query = preprocess_text(sample_query)
    
    print("Cleaned Query:", cleaned_query)
    
    print("Schema Metadata:", get_msqldb_schema_metadata())
