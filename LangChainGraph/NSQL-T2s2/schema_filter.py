import pyodbc
import re

def get_all_table_info(conn_str: str):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    tables = cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
    """).fetchall()

    table_columns = {}
    for schema, table in tables:
        full_name = f"{schema}.{table}"
        cols = cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """, (schema, table)).fetchall()
        table_columns[full_name] = cols

    conn.close()
    return table_columns

def select_relevant_tables(user_query: str, table_columns: dict) -> dict:
    relevant = {}
    keywords = set(re.findall(r"\w+", user_query.lower()))
    for table, cols in table_columns.items():
        if any(col[0].lower() in keywords or col[1].lower() in keywords for col in cols):
            relevant[table] = cols
    return relevant

def build_schema_prompt_fragment(relevant_tables: dict) -> str:
    fragment = ""
    for table, cols in relevant_tables.items():
        _, table_name = table.split(".")
        fragment += f"CREATE TABLE {table_name} (\n"
        for col_name, col_type in cols:
            fragment += f"  {col_name} {col_type},\n"
        fragment = fragment.rstrip(",\n") + "\n)\n\n"
    return fragment
