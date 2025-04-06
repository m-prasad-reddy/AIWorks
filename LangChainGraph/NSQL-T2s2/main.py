# Phase 1 & 2: Metadata Extraction + Agent with NSQL Inference + Phase 3: Feedback Loop & Caching
import pyodbc
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import RunnableSequence
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
NSQL_MODEL_NAME = "NumbersStation/nsql-6B"

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    generated_sql: str
    corrected_sql: str

# Hash connection details to uniquely identify a database instance
def hash_conn_details(host: str, db: str, user: str) -> str:
    key = f"{host}|{db}|{user}"
    return hashlib.sha256(key.encode()).hexdigest()

# Establish ODBC connection
def get_connection(dsn: str = None, server: str = None, database: str = None, user: str = None, password: str = None) -> pyodbc.Connection:
    if dsn:
        return pyodbc.connect(dsn)
    elif all([server, database, user, password]):
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={user};PWD={password}"
        return pyodbc.connect(conn_str)
    else:
        raise ValueError("Provide either DSN or full connection parameters (server, database, user, password)")

# Extract schema metadata from SQL Server
def extract_schema_metadata(conn: pyodbc.Connection) -> Dict[str, Any]:
    cursor = conn.cursor()
    schema_data = {}

    # Get columns and datatypes
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

    # Get table comments
    cursor.execute("""
        SELECT s.name AS schema_name, t.name AS table_name, ep.value AS description
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        LEFT JOIN sys.extended_properties ep
          ON ep.major_id = t.object_id AND ep.minor_id = 0 AND ep.name = 'MS_Description'
    """)
    for row in cursor.fetchall():
        key = f"{row.schema_name}.{row.table_name}"
        if key in schema_data:
            schema_data[key]["table_description"] = row.description or ""

    # Get column comments
    cursor.execute("""
        SELECT s.name AS schema_name, t.name AS table_name, c.name AS column_name, ep.value AS description
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        JOIN sys.columns c ON c.object_id = t.object_id
        LEFT JOIN sys.extended_properties ep 
          ON ep.major_id = t.object_id AND ep.minor_id = c.column_id AND ep.name = 'MS_Description'
    """)
    for row in cursor.fetchall():
        key = f"{row.schema_name}.{row.table_name}"
        if key in schema_data:
            schema_data[key]["column_descriptions"][row.column_name] = row.description or ""

    return schema_data

# Save schema to local cache
def cache_schema(schema: Dict[str, Any], cache_key: str):
    Path(".cache").mkdir(exist_ok=True)
    with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}", "w") as f:
        json.dump(schema, f, indent=2)

# Load schema from cache
def load_cached_schema(cache_key: str) -> Dict[str, Any]:
    try:
        with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Initialize Phase 1: Extract and cache schema metadata
def initialize_metadata_phase(dsn: str = None, host: str = None, db: str = None, user: str = None, password: str = None):
    cache_key = hash_conn_details(host or dsn, db or dsn, user or dsn)
    existing_cache = load_cached_schema(cache_key)
    conn = get_connection(dsn=dsn, server=host, database=db, user=user, password=password)
    latest_schema = extract_schema_metadata(conn)

    if existing_cache != latest_schema:
        print("⚠️ Schema has changed. Updating cache...")
        cache_schema(latest_schema, cache_key)
    else:
        print("✅ Loaded schema from cache (no changes detected).")

    return latest_schema

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(NSQL_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(NSQL_MODEL_NAME)

# Heuristic to rank and select top-k relevant tables for prompt
def identify_relevant_tables(user_query: str, schema: Dict[str, Any]) -> List[str]:
    keywords = set(re.findall(r'\b\w+\b', user_query.lower()))
    scores = {}
    for table_name, meta in schema.items():
        combined_text = f"{table_name.lower()} " + ' '.join([col[0].lower() for col in meta['columns']]) + ' ' + meta.get('table_description', '')
        match_count = len([word for word in keywords if word in combined_text])
        if match_count:
            scores[table_name] = match_count
    return sorted(scores, key=scores.get, reverse=True)[:5]

# Format prompt in NSQL-style DDL + instruction
def build_prompt(user_query: str, schema: Dict[str, Any], selected_tables: List[str]) -> str:
    ddl_section = ""
    for table in selected_tables:
        columns = schema[table]['columns']
        ddl = f"CREATE TABLE {table} (\n"
        for col, dtype in columns:
            comment = schema[table]['column_descriptions'].get(col, '')
            ddl += f"    {col} {dtype} -- {comment}\n"
        ddl += ")\n\n"
        ddl_section += ddl
    return f"""{ddl_section}-- Using valid Transact-SQL (T-SQL), answer the following question. Use [SCHEMA].[TABLE] format for table names.:
-- {user_query}
SELECT"""

# Generate SQL using NSQL model
def generate_sql(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Load or save JSON helpers
def load_json(path: str):
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path: str, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Cache query result using hashed prompt
def cache_query_result(prompt: str, sql: str):
    query_cache = load_json(QUERY_CACHE_FILE)
    query_cache[hashlib.sha256(prompt.encode()).hexdigest()] = sql
    save_json(QUERY_CACHE_FILE, query_cache)

# Retrieve query from cache or feedback if available
def get_cached_or_feedback_query(prompt: str) -> str:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    query_cache = load_json(QUERY_CACHE_FILE)
    if prompt_hash in query_cache:
        return query_cache[prompt_hash]
    feedback_log = load_json(FEEDBACK_FILE)
    if prompt_hash in feedback_log:
        return feedback_log[prompt_hash]["corrected"]
    return None

# Log incorrect feedback with corrected SQL
def log_feedback(prompt: str, generated_sql: str, corrected_sql: str):
    feedback_log = load_json(FEEDBACK_FILE)
    feedback_log[hashlib.sha256(prompt.encode()).hexdigest()] = {
        "generated": generated_sql,
        "corrected": corrected_sql,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    save_json(FEEDBACK_FILE, feedback_log)
    print("📘 Feedback saved.")

# --- API Endpoint ---
@app.post("/query")
def query_handler(request: QueryRequest):
    schema = app.state.schema  # Access schema from app state
    relevant = identify_relevant_tables(request.question, schema)
    prompt = build_prompt(request.question, schema, relevant)
    cached = get_cached_or_feedback_query(prompt)
    if cached:
        return {"sql": cached, "cached": True}
    sql = generate_sql(prompt)
    cache_query_result(prompt, sql)
    return {"sql": sql, "cached": False}

# --- Feedback Endpoint ---
@app.post("/feedback")
def feedback_handler(feedback: FeedbackRequest):
    schema = app.state.schema
    relevant = identify_relevant_tables(feedback.question, schema)
    prompt = build_prompt(feedback.question, schema, relevant)
    log_feedback(prompt, feedback.generated_sql, feedback.corrected_sql)
    return {"message": "Feedback logged successfully."}

# --- Load schema on startup ---
@app.on_event("startup")
async def load_schema_on_startup():
    SERVER = "localhost"
    DATABASE = "AdventureWorks"
    USER = "sa"
    PASSWORD = "yourpassword"
    schema = initialize_metadata_phase(
        host=SERVER,
        db=DATABASE,
        user=USER,
        password=PASSWORD
    )
    app.state.schema = schema

# --- Main CLI or API switch ---
if __name__ == "__main__":
    SERVER = "localhost"
    DATABASE = "AdventureWorks"
    USER = "sa"
    PASSWORD = "S0rry!43"
    schema = initialize_metadata_phase(
        host=SERVER,
        db=DATABASE,
        user=USER,
        password=PASSWORD
    )

    mode = os.getenv("MODE", "cli")
    if mode == "api":
        app.state.schema = schema
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        while True:
            user_input = input("Enter your query (or 'exit' / 'reload'): ")
            if user_input.lower() == "exit":
                break
            if user_input.lower() == "reload":
                print("🔄 Reloading schema metadata...")
                schema = initialize_metadata_phase(
                    host=SERVER,
                    db=DATABASE,
                    user=USER,
                    password=PASSWORD
                )
                continue
            relevant_tables = identify_relevant_tables(user_input, schema)
            prompt = build_prompt(user_input, schema, relevant_tables)
            print("\n--- Prompt Sent to NSQL Model ---\n")
            print(prompt)
            cached_or_feedback_sql = get_cached_or_feedback_query(prompt)
            if cached_or_feedback_sql:
                print("\n✅ Found cached or corrected SQL for prompt:\n")
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
