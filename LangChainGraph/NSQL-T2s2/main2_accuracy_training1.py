# Phase 1 & 2: Metadata Extraction + Agent with NSQL Inference + Phase 3: Feedback Loop & Caching
import pyodbc
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import RunnableSequence
import re

# Constants and file paths
CACHE_FILE = "schema_cache.json"
FEEDBACK_FILE = ".cache/feedback_log.json"
QUERY_CACHE_FILE = ".cache/query_cache.json"
NSQL_MODEL_NAME = "NumbersStation/nsql-6B"

# Hash connection details to uniquely identify a database instance

def hash_conn_details(host: str, db: str, user: str) -> str:
    key = f"{host}|{db}|{user}"
    return hashlib.sha256(key.encode()).hexdigest()

# Establish ODBC connection

def get_connection(dsn: str) -> pyodbc.Connection:
    return pyodbc.connect(dsn)

# Extract schema metadata from SQL Server: tables, columns, and their descriptions

def extract_schema_metadata(conn: pyodbc.Connection) -> Dict[str, Any]:
    cursor = conn.cursor()
    schema_data = {}

    # Fetch columns and data types
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

    # Fetch table-level comments if available
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

    # Fetch column-level comments
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

# Load schema from cache if it exists

def load_cached_schema(cache_key: str) -> Dict[str, Any]:
    try:
        with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Initialize Phase 1: Extract and cache schema metadata

def initialize_metadata_phase(dsn: str, host: str, db: str, user: str):
    cache_key = hash_conn_details(host, db, user)
    existing_cache = load_cached_schema(cache_key)
    conn = get_connection(dsn)
    latest_schema = extract_schema_metadata(conn)

    # Check if schema has changed, and update cache only if needed
    if existing_cache != latest_schema:
        print("⚠️ Schema has changed. Updating cache...")
        cache_schema(latest_schema, cache_key)
    else:
        print("✅ Loaded schema from cache (no changes detected).")

    return latest_schema

# ------------------ Phase 2: NSQL Query Generation ------------------
# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(NSQL_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(NSQL_MODEL_NAME)

# Identify relevant tables based on overlap between user query and table/column names

def identify_relevant_tables(user_query: str, schema: Dict[str, Any]) -> List[str]:
    keywords = set(re.findall(r'\b\w+\b', user_query.lower()))
    scores = {}
    for table_name, meta in schema.items():
        combined_text = f"{table_name.lower()} " + ' '.join([col[0].lower() for col in meta['columns']]) + ' ' + meta.get('table_description', '')
        match_count = len([word for word in keywords if word in combined_text])
        if match_count:
            scores[table_name] = match_count
    return sorted(scores, key=scores.get, reverse=True)[:5]  # Return top 5 relevant tables

# Construct model prompt from relevant DDLs

def build_prompt(user_query: str, schema: Dict[str, Any], selected_tables: List[str]) -> str:
    ddl_section = ""
    for table in selected_tables:
        schema_name = table
        columns = schema[table]['columns']
        ddl = f"CREATE TABLE {schema_name} (\n"
        for col, dtype in columns:
            comment = schema[table]['column_descriptions'].get(col, '')
            ddl += f"    {col} {dtype} -- {comment}\n"
        ddl += ")\n\n"
        ddl_section += ddl

    prompt = f"""{ddl_section}-- Using valid Transact-SQL (T-SQL), answer the following question:
-- {user_query}
SELECT"""
    return prompt

# Generate SQL from model based on prompt

def generate_sql(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ------------------ Phase 3: Caching + Feedback ------------------
# Utility functions for caching and feedback logging

def load_json(path: str):
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path: str, data: Dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Cache a successful prompt-result pair

def cache_query_result(prompt: str, sql: str):
    query_cache = load_json(QUERY_CACHE_FILE)
    query_cache[hashlib.sha256(prompt.encode()).hexdigest()] = sql
    save_json(QUERY_CACHE_FILE, query_cache)

# Retrieve cached SQL for previously seen prompt

def get_cached_query(prompt: str) -> str:
    query_cache = load_json(QUERY_CACHE_FILE)
    return query_cache.get(hashlib.sha256(prompt.encode()).hexdigest())

# Store incorrect results with user correction for future fine-tuning

def log_feedback(prompt: str, generated_sql: str, corrected_sql: str):
    feedback_log = load_json(FEEDBACK_FILE)
    feedback_log[hashlib.sha256(prompt.encode()).hexdigest()] = {
        "generated": generated_sql,
        "corrected": corrected_sql
    }
    save_json(FEEDBACK_FILE, feedback_log)

# ------------------ Entry Point ------------------
if __name__ == "__main__":
    # Set your connection details here
    DSN = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=AdventureWorks;UID=sa;PWD=S0rry!43"

    # Start Phase 1: Load schema metadata
    schema = initialize_metadata_phase(
        dsn=DSN,
        host="localhost",
        db="AdventureWorks",
        user="sa"
    )

    # Phase 2+3 loop for user interaction
    while True:
        user_input = input("Enter your query (or 'exit'): ")
        if user_input.lower() == "exit":
            break

        # Identify tables and create prompt
        relevant_tables = identify_relevant_tables(user_input, schema)
        prompt = build_prompt(user_input, schema, relevant_tables)
        print("\n--- Prompt Sent to NSQL Model ---\n")
        print(prompt)

        # Check for cached SQL
        cached_sql = get_cached_query(prompt)
        if cached_sql:
            print("\n✅ Found cached SQL for prompt:\n")
            print(cached_sql)
        else:
            sql_response = generate_sql(prompt)
            print("\n--- SQL Generated ---\n")
            print(sql_response)

            # Phase 3: Ask user for feedback
            feedback = input("\n✅ Was the result correct? (yes/no): ")
            if feedback.lower() == "yes":
                cache_query_result(prompt, sql_response)
            else:
                expected = input("Please provide the correct SQL: ")
                log_feedback(prompt, sql_response, expected)
                print("Thank you, correction logged for future fine-tuning.")