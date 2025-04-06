# main.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import pyodbc
import re
from collections import defaultdict

# --- Load NSQL-6B Model ---
tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-6B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-6B")

# --- In-memory cache ---
query_cache = defaultdict(str)

# --- SQL Server Connection String ---
CONN_STR = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=BikeStores;UID=sa;PWD=S0rry!43"

# --- Get all table info ---
def get_all_table_info(conn_str: str):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # Step 1: Get tables
    tables = cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
    """).fetchall()

    # Step 2: Fetch descriptions for tables & columns
    table_descriptions = {}
    column_descriptions = {}

    # Table comments
    table_rows = cursor.execute("""
        SELECT s.name AS schema_name, t.name AS table_name, ep.value AS description
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        LEFT JOIN sys.extended_properties ep ON ep.major_id = t.object_id AND ep.minor_id = 0 AND ep.name = 'MS_Description'
    """).fetchall()

    for row in table_rows:
        table_descriptions[f"{row.schema_name}.{row.table_name}"] = row.description or ""

    # Column comments
    column_rows = cursor.execute("""
        SELECT s.name AS schema_name, t.name AS table_name, c.name AS column_name, ep.value AS description
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        INNER JOIN sys.columns c ON c.object_id = t.object_id
        LEFT JOIN sys.extended_properties ep 
            ON ep.major_id = t.object_id 
            AND ep.minor_id = c.column_id 
            AND ep.name = 'MS_Description'
    """).fetchall()

    for row in column_rows:
        key = f"{row.schema_name}.{row.table_name}.{row.column_name}"
        column_descriptions[key] = row.description or ""

    # Step 3: Get all columns
    table_columns = {}
    for schema, table in tables:
        full_name = f"{schema}.{table}"
        cols = cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """, (schema, table)).fetchall()
        table_columns[full_name] = {
            "columns": cols,
            "table_description": table_descriptions.get(full_name, ""),
            "column_descriptions": {
                col[0]: column_descriptions.get(f"{schema}.{table}.{col[0]}", "")
                for col in cols
            }
        }

    conn.close()
    return table_columns

# --- Select relevant tables based on query ---
def select_relevant_tables(user_query: str, table_columns: dict) -> dict:
    relevant = {}
    keywords = set(re.findall(r"\w+", user_query.lower()))
    for table, cols in table_columns.items():
        if any(col[0].lower() in keywords or col[1].lower() in keywords for col in cols):
            relevant[table] = cols
    return relevant

# --- Build prompt from relevant tables ---
def build_schema_prompt_fragment(relevant_tables: dict) -> str:
    fragment = ""
    for table, table_info in relevant_tables.items():
        schema, table_name = table.split(".")

        table_comment = table_info.get("table_description", "")
        fragment += f"-- {table_comment}\n" if table_comment else ""
        fragment += f"CREATE TABLE {schema}.{table_name} (\n"

        for col_name, col_type in table_info["columns"]:
            comment = table_info["column_descriptions"].get(col_name, "")
            line = f"  {col_name} {col_type}"
            if comment:
                line += f" -- {comment}"
            fragment += line + ",\n"

        fragment = fragment.rstrip(",\n") + "\n)\n\n"
    return fragment

# --- Check read-only SQL ---
def is_read_only(sql: str) -> bool:
    lowered = sql.lower()
    return lowered.strip().startswith("select") and all(x not in lowered for x in ["update", "delete", "insert", "alter", "drop"])

# --- Run NSQL Model ---
def run_nsql_model_old(prompt: str, max_tokens: int = 300) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=max_tokens)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def run_nsql_model(prompt: str, max_tokens: int = 300) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Explicitly set pad_token_id to eos_token_id
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# --- Execute the SQL query and return top 10 results ---
def execute_sql_query(conn_str: str, sql: str):
    if not is_read_only(sql):
        raise ValueError("Only SELECT queries are allowed!")

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute(sql)

    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchmany(10)

    conn.close()

    return [dict(zip(columns, row)) for row in rows]

# --- MAIN driver ---
if __name__ == "__main__":
    user_query = input("Enter your natural language query: ")

    # 1. Check cache
    if user_query.lower() in query_cache:
        final_sql = query_cache[user_query.lower()]
        print(f"\n💡 Using cached SQL:\n{final_sql}")
    else:
        print("\n🔎 Analyzing schema...")
        table_info = get_all_table_info(CONN_STR)
        relevant_tables = select_relevant_tables(user_query, table_info)
        schema_fragment = build_schema_prompt_fragment(relevant_tables)

        prompt = f"""{schema_fragment}
-- Using valid SQL Server syntax, answer the following question for the tables provided above.
-- {user_query}
SELECT"""

        print("\n🧠 Generating SQL using NSQL-6B...")
        raw_output = run_nsql_model(prompt)
        final_sql = "SELECT" + raw_output.split("SELECT", 1)[-1]

        if not is_read_only(final_sql):
            raise ValueError("⛔️ Model generated non-read-only SQL!")

        query_cache[user_query.lower()] = final_sql
        print(f"\n✅ SQL Generated:\n{final_sql}")

    # 2. Execute the SQL
    print("\n📊 Executing SQL and fetching top 10 results...")
    try:
        results = execute_sql_query(CONN_STR, final_sql)
        for row in results:
            print(row)
    except Exception as e:
        print("❌ Error executing query:", e)
