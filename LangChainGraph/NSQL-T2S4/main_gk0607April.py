import pyodbc
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import datetime
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

# Constants and file paths
CACHE_FILE = "schema_cache.json"
FEEDBACK_FILE = ".cache/feedback_log.json"
QUERY_CACHE_FILE = ".cache/query_cache.json"
TABLE_HINTS_FILE = ".cache/table_hints.json"

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    generated_sql: str
    corrected_sql: str

# Config vibes for BikeStores
SERVER = "localhost"
DATABASE = "BikeStores"
USER = "sa"
PASSWORD = ""  # Update with your password

# Hash connection details
def hash_conn_details(host: str, db: str, user: str) -> str:
    key = f"{host}|{db}|{user}"
    return hashlib.sha256(key.encode()).hexdigest()

# Establish ODBC connection
def get_connection() -> pyodbc.Connection:
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USER};PWD={PASSWORD}"
    return pyodbc.connect(conn_str)

# Extract schema metadata
def extract_schema_metadata(conn: pyodbc.Connection) -> Dict[str, Any]:
    cursor = conn.cursor()
    schema_data = {}
    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
    """)
    for row in cursor.fetchall():
        schema = row.TABLE_SCHEMA
        table = row.TABLE_NAME
        key = f"{schema}.{table}"
        if key not in schema_data:
            schema_data[key] = {"columns": [], "table_description": "", "column_descriptions": {}}
        schema_data[key]["columns"].append((row.COLUMN_NAME, row.DATA_TYPE))
    return schema_data

# Cache schema
def cache_schema(schema: Dict[str, Any], cache_key: str):
    Path(".cache").mkdir(exist_ok=True)
    with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}", "w") as f:
        json.dump(schema, f, indent=2)

# Load cached schema
def load_cached_schema(cache_key: str) -> Dict[str, Any]:
    try:
        with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Create table hints cache
def create_table_hints_cache(schema: Dict[str, Any], cache_path: str = TABLE_HINTS_FILE) -> Dict[str, List[str]]:
    Path(".cache").mkdir(exist_ok=True)
    table_hints = {
        "order": ["sales.orders", "sales.order_items"],
        "city": ["sales.stores", "sales.customers"],
        "ship": ["sales.orders"],
        "date": ["sales.orders"],
        "customer": ["sales.customers"],
        "store": ["sales.stores"],
        "product": ["production.products", "sales.order_items"]
    }

    for table_name, meta in schema.items():
        cols_lower = [col[0].lower() for col in meta["columns"]]
        for keyword in table_hints:
            if any(keyword in col for col in cols_lower) and table_name not in table_hints[keyword]:
                table_hints[keyword].append(table_name)

    with open(cache_path, "w") as f:
        json.dump(table_hints, f, indent=2)
    print(f"✅ Table hints cached to {cache_path}. Edit it if you vibe with tweaks!")
    return table_hints

# Load table hints
def load_table_hints(cache_path: str = TABLE_HINTS_FILE) -> Dict[str, List[str]]:
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Initialize Phase 1
def initialize_metadata_phase():
    cache_key = hash_conn_details(SERVER, DATABASE, USER)
    conn = get_connection()
    latest_schema = extract_schema_metadata(conn)
    cached_schema = load_cached_schema(cache_key)
    
    if cached_schema != latest_schema:
        print("⚠️ Schema has changed. Updating cache...")
        cache_schema(latest_schema, cache_key)
        create_table_hints_cache(latest_schema)
    else:
        print("✅ Loaded schema from cache (no changes detected).")
    
    return latest_schema

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-6B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-6B")

# Select relevant tables
def identify_relevant_tables(user_query: str, schema: Dict[str, Any]) -> List[str]:
    query_lower = user_query.lower()
    keywords = set(re.findall(r'\b\w+\b', query_lower))
    table_hints = load_table_hints()
    scores = {}
    
    for table_name, meta in schema.items():
        combined_text = (
            f"{table_name.lower()} " + 
            " ".join(col[0].lower() for col in meta["columns"])
        )
        match_count = len([word for word in keywords if word in combined_text])
        
        for keyword, tables in table_hints.items():
            if keyword in query_lower and table_name in tables:
                if keyword == "city" and "order" in query_lower and table_name == "sales.stores":
                    match_count += 5
                else:
                    match_count += 3
        
        if match_count:
            scores[table_name] = match_count
    
    return sorted(scores, key=scores.get, reverse=True)[:3]

# Format prompt with [SCHEMA].[TABLE]
def build_prompt(user_query: str, schema: Dict[str, Any], selected_tables: List[str]) -> str:
    ddl_section = ""
    for table in selected_tables:
        columns = schema[table]["columns"]
        schema_name, table_name = table.split(".")
        ddl = f"CREATE TABLE [{schema_name}].[{table_name}] (\n"
        for col, dtype in columns:
            ddl += f"    {col} {dtype}\n"
        ddl += ")\n\n"
        ddl_section += ddl
    return f"""{ddl_section}-- Using valid Transact-SQL (T-SQL), answer this (SELECT only):\n-- Use [SCHEMA].[TABLE] format (e.g., [sales].[orders]).\n-- Example: SELECT [sales].[stores].city, COUNT(*) FROM [sales].[stores] JOIN [sales].[orders] ON [sales].[stores].store_id = [sales].[orders].store_id\n-- {user_query}\nSELECT"""

# Generate SQL
def generate_sql(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)
    sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if not sql.strip().upper().startswith("SELECT"):
        sql = "SELECT " + sql.split("SELECT", 1)[-1] if "SELECT" in sql.upper() else "Error: Invalid SQL generated"
    # Remove trailing semicolon to avoid execution errors
    sql = sql.rstrip(';').strip()
    return sql

# JSON helpers
def load_json(path: str):
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path: str, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Cache query result
def cache_query_result(prompt: str, sql: str):
    query_cache = load_json(QUERY_CACHE_FILE)
    query_cache[hashlib.sha256(prompt.encode()).hexdigest()] = sql
    save_json(QUERY_CACHE_FILE, query_cache)

# Retrieve cached or feedback query
def get_cached_or_feedback_query(prompt: str) -> str:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    query_cache = load_json(QUERY_CACHE_FILE)
    if prompt_hash in query_cache:
        return query_cache[prompt_hash]
    feedback_log = load_json(FEEDBACK_FILE)
    if prompt_hash in feedback_log:
        return feedback_log[prompt_hash]["corrected"]
    return None

# Log feedback
def log_feedback(prompt: str, generated_sql: str, corrected_sql: str):
    feedback_log = load_json(FEEDBACK_FILE)
    feedback_log[hashlib.sha256(prompt.encode()).hexdigest()] = {
        "generated": generated_sql,
        "corrected": corrected_sql,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    save_json(FEEDBACK_FILE, feedback_log)
    print("📘 Feedback saved.")

# API Endpoints
@app.post("/query")
def query_handler(request: QueryRequest):
    schema = app.state.schema
    relevant = identify_relevant_tables(request.question, schema)
    prompt = build_prompt(request.question, schema, relevant)
    cached = get_cached_or_feedback_query(prompt)
    if cached:
        return {"sql": cached, "cached": True}
    sql = generate_sql(prompt)
    cache_query_result(prompt, sql)
    return {"sql": sql, "cached": False}

@app.post("/feedback")
def feedback_handler(feedback: FeedbackRequest):
    schema = app.state.schema
    relevant = identify_relevant_tables(feedback.question, schema)
    prompt = build_prompt(feedback.question, schema, relevant)
    log_feedback(prompt, feedback.generated_sql, feedback.corrected_sql)
    return {"message": "Feedback logged successfully."}

@app.on_event("startup")
async def load_schema_on_startup():
    schema = initialize_metadata_phase()
    app.state.schema = schema

# Main CLI
if __name__ == "__main__":
    schema = initialize_metadata_phase()
    mode = os.getenv("MODE", "cli")
    if mode == "api":
        app.state.schema = schema
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        conn = get_connection()
        while True:
            user_input = input("Your query (or 'exit' / 'reload'): ")
            if user_input.lower() == "exit":
                break
            if user_input.lower() == "reload":
                print("🔄 Reloading schema...")
                schema = initialize_metadata_phase()
                continue
            relevant_tables = identify_relevant_tables(user_input, schema)
            prompt = build_prompt(user_input, schema, relevant_tables)
            print("\n--- Prompt Sent to NSQL Model ---\n")
            print(prompt[:200] + "...")
            cached_or_feedback_sql = get_cached_or_feedback_query(prompt)
            if cached_or_feedback_sql:
                print("\n✅ Found cached or corrected SQL:\n")
                print(cached_or_feedback_sql)
            else:
                sql_response = generate_sql(prompt)
                print("\n--- SQL Generated ---\n")
                print(sql_response)
                feedback = input("\nWas the result correct? (yes/no): ").strip().lower()
                if feedback == "yes":
                    cache_query_result(prompt, sql_response)
                    print("✅ Query cached.")
                elif feedback == "no":
                    expected = input("Please provide the correct SQL: ").strip()
                    if expected:
                        log_feedback(prompt, sql_response, expected)
                        print("✅ Feedback logged.")
                    else:
                        print("⚠️ No correction provided.")
            # Execute and show results
            try:
                cursor = conn.cursor()
                cursor.execute(sql_response)
                results = cursor.fetchall()
                print(f"Results:\n{results}")
            except Exception as e:
                print(f"Execution vibes off: {str(e)}")
        conn.close()