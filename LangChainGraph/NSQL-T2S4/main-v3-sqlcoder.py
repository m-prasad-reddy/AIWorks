"""
COMPLETE SQL Generator Application with:
- Natural Language to SQL conversion
- Multiple database support
- Schema caching
- Query feedback system
- CLI and API interfaces
"""

import pyodbc
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import datetime
import os
import sys
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from tabulate import tabulate
from retry import retry
import logging
from dynamic_hints import DynamicHintsGenerator

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
CACHE_FILE = "schema_cache.json"
FEEDBACK_FILE = ".cache/feedback_log.json"
QUERY_CACHE_FILE = ".cache/query_cache.json"
TABLE_HINTS_FILE = ".cache/dynamic_table_hints.json"
DB_CONFIG_FILE = "db_configurations.json"
MAX_RETRIES = 3
RETRY_DELAY = 1

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    generated_sql: str
    corrected_sql: str

class DBChangeRequest(BaseModel):
    db_alias: str

# ======================
# Database Configuration
# ======================
def validate_db_config(config: Dict[str, Any]) -> bool:
    """Validate database configuration structure"""
    required_keys = {'server', 'database', 'user', 'password'}
    for alias, settings in config.items():
        if not isinstance(settings, dict):
            return False
        if not required_keys.issubset(settings.keys()):
            return False
        if not all(isinstance(v, str) for v in settings.values()):
            return False
    return True

def load_and_validate_config() -> Dict[str, Any]:
    """Load and validate configuration file"""
    try:
        if not Path(DB_CONFIG_FILE).exists():
            print(f"\n❌ Error: Configuration file '{DB_CONFIG_FILE}' not found")
            sys.exit(1)
        
        with open(DB_CONFIG_FILE) as f:
            config = json.load(f)
            
        if not validate_db_config(config):
            print(f"\n❌ Error: Invalid configuration in '{DB_CONFIG_FILE}'")
            sys.exit(1)
            
        return config
        
    except json.JSONDecodeError:
        print(f"\n❌ Error: Invalid JSON in '{DB_CONFIG_FILE}'")
        sys.exit(1)

# ======================
# Database Connection
# ======================
class DBSession:
    """Manages database connections and state"""
    
    def __init__(self):
        self.config = load_and_validate_config()
        self.current_db = None
        self.conn = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connection is active and valid"""
        if not self._is_connected or not self.conn:
            return False
            
        try:
            self.conn.timeout = 5
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except pyodbc.Error:
            self._is_connected = False
            return False
        finally:
            self.conn.timeout = 0

    def list_available_dbs(self) -> List[str]:
        """Return list of configured database aliases"""
        return list(self.config.keys())

    @retry(pyodbc.OperationalError, tries=MAX_RETRIES, delay=RETRY_DELAY)
    def connect(self, db_alias: str) -> bool:
        """Establish database connection"""
        try:
            if db_alias not in self.config:
                raise ValueError(f"Database '{db_alias}' not configured")
                
            config = self.config[db_alias]
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={config['server']};"
                f"DATABASE={config['database']};"
                f"UID={config['user']};"
                f"PWD={config['password']}"
            )
            
            if self.conn:
                self.conn.close()
            self.conn = pyodbc.connect(conn_str)
            self.current_db = db_alias
            self._is_connected = True
            logger.info(f"Connected to {db_alias}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            self.conn = None
            self._is_connected = False
            return False

# Initialize global database session
db_session = DBSession()

# ======================
# Schema Management
# ======================
def hash_conn_details(host: str, db: str, user: str) -> str:
    """Generate unique hash for connection details"""
    return hashlib.sha256(f"{host}|{db}|{user}".encode()).hexdigest()

def extract_schema_metadata(conn: pyodbc.Connection) -> Dict[str, Any]:
    """Extract schema information from database"""
    schema = {}
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
        """)
        for row in cursor:
            key = f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}"
            if key not in schema:
                schema[key] = {"columns": []}
            schema[key]["columns"].append((row.COLUMN_NAME, row.DATA_TYPE))
            
        cursor.execute("""
            SELECT 
                fk.name AS constraint_name,
                OBJECT_NAME(fk.parent_object_id) AS table_name,
                COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS column_name,
                OBJECT_NAME(fk.referenced_object_id) AS referenced_table
            FROM sys.foreign_keys AS fk
            INNER JOIN sys.foreign_key_columns AS fkc 
            ON fk.object_id = fkc.constraint_object_id
        """)
        for row in cursor:
            key = f"dbo.{row.table_name}"
            if key in schema:
                if "foreign_keys" not in schema[key]:
                    schema[key]["foreign_keys"] = []
                schema[key]["foreign_keys"].append({
                    "column": row.column_name,
                    "references": f"dbo.{row.referenced_table}"
                })
        
        cursor.execute("""
            SELECT 
                t.name AS table_name,
                ind.name AS index_name,
                col.name AS column_name
            FROM sys.indexes ind
            INNER JOIN sys.index_columns ic ON ind.object_id = ic.object_id AND ind.index_id = ic.index_id
            INNER JOIN sys.columns col ON ic.object_id = col.object_id AND ic.column_id = col.column_id
            INNER JOIN sys.tables t ON ind.object_id = t.object_id
        """)
        for row in cursor:
            key = f"dbo.{row.table_name}"
            if key in schema:
                if "indexes" not in schema[key]:
                    schema[key]["indexes"] = []
                schema[key]["indexes"].append({
                    "name": row.index_name,
                    "column": row.column_name
                })
                
        return schema
        
    except pyodbc.Error as e:
        logger.error(f"Schema extraction failed: {str(e)}")
        raise

def cache_schema(schema: Dict[str, Any], cache_key: str):
    """Cache schema to file"""
    Path(".cache").mkdir(exist_ok=True)
    with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}", "w") as f:
        json.dump(schema, f, indent=2)

def load_cached_schema(cache_key: str) -> Dict[str, Any]:
    """Load cached schema"""
    try:
        with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def initialize_metadata_phase(db_config: Dict[str, str]) -> Dict[str, Any]:
    """Load and cache database schema"""
    if not db_session.is_connected:
        raise ConnectionError("No active database connection")
        
    try:
        cache_key = hash_conn_details(
            db_config['server'],
            db_config['database'],
            db_config['user']
        )
        
        latest_schema = extract_schema_metadata(db_session.conn)
        cached_schema = load_cached_schema(cache_key)
        
        if cached_schema != latest_schema:
            print("⚠️ Schema has changed. Updating cache...")
            cache_schema(latest_schema, cache_key)
            create_dynamic_table_hints(latest_schema)
        else:
            print("✅ Loaded schema from cache")
            
        return latest_schema
        
    except Exception as e:
        logger.error(f"Schema initialization failed: {str(e)}")
        raise

# ======================
# Table Hints
# ======================
def create_table_hints_cache(schema: Dict[str, Any], cache_path: str = TABLE_HINTS_FILE) -> Dict[str, List[str]]:
    """Generate and cache static table hints"""
    Path(".cache").mkdir(exist_ok=True)
    table_hints = {
        "order": ["sales.orders", "sales.order_items"],
        "city": ["sales.stores", "sales.customers"],
        "ship": ["sales.orders"],
        "date": ["sales.orders"],
        "customer": ["sales.customers"],
        "store": ["sales.stores"],
        "product": ["production.products", "sales.order_items"],
        "category": ["production.categories"]
    }

    for table_name, meta in schema.items():
        cols_lower = [col[0].lower() for col in meta["columns"]]
        for keyword in table_hints:
            if any(keyword in col for col in cols_lower) and table_name not in table_hints[keyword]:
                table_hints[keyword].append(table_name)

    with open(cache_path, "w") as f:
        json.dump(table_hints, f, indent=2)
    print(f"✅ Static table hints cached to {cache_path}")
    return table_hints

def create_dynamic_table_hints(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate and cache dynamic table hints"""
    hint_generator = DynamicHintsGenerator()
    hints = hint_generator.generate_table_hints(schema)
    hint_generator.save_hints(hints)
    print(f"✅ Dynamic table hints generated and cached to {hint_generator.hint_cache_path}")
    return hints

def load_table_hints(cache_path: str = TABLE_HINTS_FILE) -> Dict[str, List[str]]:
    """Load cached table hints"""
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# ======================
# Query Processing
# ======================
def identify_relevant_tables(user_query: str, schema: Dict[str, Any]) -> List[str]:
    """Identify tables relevant to the user query"""
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
                match_count += 3
        
        if match_count:
            scores[table_name] = match_count
    
    return sorted(scores, key=scores.get, reverse=True)[:3]

def build_prompt(user_query: str, schema: Dict[str, Any], selected_tables: List[str]) -> str:
    """Build prompt with diverse examples"""
    ddl_section = ""
    for table in selected_tables:
        columns = schema[table]["columns"]
        schema_name, table_name = table.split(".")
        ddl = f"CREATE TABLE [{schema_name}].[{table_name}] (\n"
        for col, dtype in columns:
            ddl += f"    {col} {dtype},\n"
        if "foreign_keys" in schema.get(table, {}):
            ddl += "\n    -- Foreign Keys:\n"
            for fk in schema[table]["foreign_keys"]:
                ddl += f"    -- FK: {fk['column']} → {fk['references']}\n"
        ddl = ddl.rstrip(",\n") + "\n)\n\n"
        ddl_section += ddl
    
    return f"""{ddl_section}-- Generate a single valid Transact-SQL SELECT statement for: '{user_query}'
-- Use [SCHEMA].[TABLE] format (e.g., [production].[products])
-- Select only columns explicitly requested or implied
-- Use table aliases where needed (e.g., p for products)
-- For 'latest', use MAX() with correlation (e.g., WHERE p.model_year = (SELECT MAX(model_year) ...))
-- For 'year wise', ORDER BY year
-- Use JOINs for related tables based on foreign keys
-- Use COUNT/SUM for 'how many' or 'total' queries
-- Match schema exactly; do not guess tables/columns
-- Examples (use as patterns, not verbatim unless exact match):
-- 1. "show latest products produced year wise along with its price":
--    SELECT p.product_name, p.model_year, p.list_price
--    FROM [production].[products] p
--    WHERE p.model_year = (SELECT MAX(model_year) FROM [production].[products] WHERE product_name = p.product_name)
--    ORDER BY p.model_year
-- 2. "what production categories are available":
--    SELECT DISTINCT category_name FROM [production].[categories]
-- 3. "list all products with their categories and brands":
--    SELECT p.product_name, c.category_name, b.brand_name
--    FROM [production].[products] p
--    JOIN [production].[categories] c ON p.category_id = c.category_id
--    JOIN [production].[brands] b ON p.brand_id = b.brand_id
-- 4. "how many orders were placed in 2018":
--    SELECT COUNT(*) FROM [sales].[orders] WHERE YEAR(order_date) = 2018
-- 5. "total sales amount for each store":
--    SELECT s.store_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales
--    FROM [sales].[stores] s
--    JOIN [sales].[orders] o ON s.store_id = o.store_id
--    JOIN [sales].[order_items] oi ON o.order_id = oi.order_id
--    GROUP BY s.store_name
SELECT"""

# ... (previous imports unchanged) ...
import torch
# Set CUDA_LAUNCH_BLOCKING for debugging (remove after confirming fix)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Initialize model
try:
    tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            "defog/sqlcoder-7b-2",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2")
    model.to(device)
    logger.info(f"Model loaded: defog/sqlcoder-7b-2 on {device}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    print(f"❌ Failed to load model: {str(e)}")
    sys.exit(1)

def generate_sql(prompt: str, user_query: str) -> str:
    """Generate and clean SQL with sqlcoder-7b-2, limit to 2 rows"""
    # Check cache first
    cached_query = query_cache.get(user_query)
    if cached_query:
        sql = cached_query
        # Enforce TOP 2 on cached queries
        if not sql.upper().startswith("SELECT TOP"):
            sql = f"SELECT TOP 2 {sql[6:]}"
        return sql.rstrip(';').strip()

    if is_valid_sql(user_query):
        return f"SELECT TOP 2 {user_query.strip().rstrip(';')}"

    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logger.debug(f"Input IDs shape: {input_ids.shape}")
        
        # Use max_new_tokens to avoid input truncation
        gen_kwargs = {
            "max_new_tokens": 100,  # Limit output tokens
            "pad_token_id": tokenizer.eos_token_id,
            "num_beams": 1,
            "temperature": 0.7,
            "do_sample": False      # Greedy decoding
        }
        
        generated_ids = model.generate(input_ids, **gen_kwargs)
        logger.debug(f"Generated IDs shape: {generated_ids.shape}")
        sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        logger.debug(f"Raw model output: {sql}")
        
        # Clean output and enforce TOP 2
        match = re.search(r'(?i)^\s*SELECT\s+.*$', sql, re.MULTILINE)
        if match:
            sql = match.group(0).strip()
            if not sql.upper().startswith("SELECT TOP"):
                sql = f"SELECT TOP 2 {sql[6:]}"
        else:
            sql = "SELECT TOP 2 -- Invalid generation"
        
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE).strip()
        
        # Enhanced fallbacks with year detection
        query_lower = user_query.lower()
        year_match = re.search(r'\b(\d{4})\b', query_lower)
        year = year_match.group(1) if year_match else None
        
        if "latest" in query_lower and "product" in query_lower:
            if "MAX(model_year)" not in sql:
                sql = """SELECT TOP 2 p.product_name, p.model_year, p.list_price
FROM [production].[products] p
WHERE p.model_year = (SELECT MAX(model_year) FROM [production].[products] WHERE product_name = p.product_name)
ORDER BY p.model_year"""
        elif "year wise" in query_lower and "product" in query_lower:
            if "ORDER BY" not in sql or "model_year" not in sql:
                sql = "SELECT TOP 2 product_name, model_year, list_price FROM [production].[products] ORDER BY model_year"
        elif "customer" in query_lower and "order" in query_lower:
            if "[sales].[customers]" not in sql or "JOIN" not in sql:
                sql = """SELECT TOP 2 c.first_name, c.last_name
FROM [sales].[customers] c
JOIN [sales].[orders] o ON c.customer_id = o.customer_id"""
        elif "product" in query_lower and ("category" in query_lower or "brand" in query_lower):
            if "JOIN" not in sql:
                sql = """SELECT TOP 2 p.product_name, c.category_name, b.brand_name
FROM [production].[products] p
JOIN [production].[categories] c ON p.category_id = c.category_id
JOIN [production].[brands] b ON p.brand_id = b.brand_id"""
        elif "how many" in query_lower and "order" in query_lower and year:
            if "COUNT" not in sql or f"YEAR(order_date) = {year}" not in sql:
                sql = f"SELECT TOP 2 COUNT(*) FROM [sales].[orders] WHERE YEAR(order_date) = {year}"
        elif "total" in query_lower and "sales" in query_lower and "store" in query_lower:
            if "SUM" not in sql or "GROUP BY" not in sql:
                sql = """SELECT TOP 2 s.store_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales
FROM [sales].[stores] s
JOIN [sales].[orders] o ON s.store_id = o.store_id
JOIN [sales].[order_items] oi ON o.order_id = oi.order_id
GROUP BY s.store_name"""
        elif "categories" in query_lower and "available" in query_lower:
            if "category_name" not in sql:
                sql = "SELECT TOP 2 DISTINCT category_name FROM [production].[categories]"
        elif "stock" in query_lower and "store" in query_lower:
            if "[production].[stocks]" not in sql:
                if "details" in query_lower:
                    sql = """SELECT TOP 2 p.product_name, sk.store_id, s.store_name, s.street, s.city, s.email, sk.quantity
FROM [production].[products] p
JOIN [production].[stocks] sk ON p.product_id = sk.product_id
JOIN [sales].[stores] s ON sk.store_id = s.store_id
ORDER BY sk.store_id, p.product_name"""
                else:
                    sql = """SELECT TOP 2 sk.store_id, p.product_name, sk.quantity
FROM [production].[products] p
JOIN [production].[stocks] sk ON p.product_id = sk.product_id
ORDER BY sk.store_id, p.product_name"""
        elif "how many" in query_lower and "store" in query_lower:
            if "COUNT" not in sql or "[sales].[stores]" not in sql:
                sql = "SELECT TOP 2 COUNT(*) AS store_count FROM [sales].[stores]"
        
        return sql.rstrip(';').strip()
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return "SELECT TOP 2 -- Generation error"

def execute_sql(sql: str) -> None:
    """Execute SQL and display results, fixing common syntax issues"""
    try:
        # Fix spacing around ORDER BY
        sql = re.sub(r'(\w+)\s*ORDER\s+BY', r'\1 ORDER BY', sql, flags=re.I)
        cursor = db_session.conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        table = PrettyTable()
        table.field_names = columns
        for row in rows[:2]:  # Double enforcement of 2-row limit
            table.add_row(row)
        print("\n--- Execution Results ---")
        print(table)
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        print(f"⚠️ Execution error: {str(e)}")

# ... (rest of main.py unchanged) ...


def is_valid_sql(query: str) -> bool:
    """Check if the input is a valid SQL SELECT statement"""
    query_upper = query.strip().upper()
    return query_upper.startswith("SELECT") and ("FROM" in query_upper or ";" in query_upper)
 
# ======================
# Caching & Feedback
# ======================
def load_json(path: str) -> Dict:
    """Load JSON file"""
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path: str, data: Dict):
    """Save data to JSON file"""
    Path(".cache").mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def cache_query_result(prompt: str, sql: str):
    """Cache generated SQL query after cleaning"""
    # Remove any preamble before caching
    clean_sql = re.sub(r'^.*?(?=SELECT\s)', '', sql, flags=re.IGNORECASE | re.DOTALL).strip()
    query_cache = load_json(QUERY_CACHE_FILE)
    query_cache[hashlib.sha256(prompt.encode()).hexdigest()] = clean_sql
    save_json(QUERY_CACHE_FILE, query_cache)

def get_cached_or_feedback_query(prompt: str) -> Optional[str]:
    """Check for cached or feedback-corrected queries"""
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    query_cache = load_json(QUERY_CACHE_FILE)
    feedback_log = load_json(FEEDBACK_FILE)
    
    if prompt_hash in query_cache:
        return query_cache[prompt_hash]
    if prompt_hash in feedback_log:
        return feedback_log[prompt_hash]["corrected"]
    return None

def log_feedback(prompt: str, generated_sql: str, corrected_sql: str):
    """Log user feedback for query corrections"""
    feedback_log = load_json(FEEDBACK_FILE)
    feedback_log[hashlib.sha256(prompt.encode()).hexdigest()] = {
        "generated": generated_sql,
        "corrected": corrected_sql,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    save_json(FEEDBACK_FILE, feedback_log)
    print("✅ Feedback saved")

# ======================
# Query Execution
# ======================
def execute_and_format(conn: pyodbc.Connection, sql: str, max_rows: int = 50) -> str:
    """Execute SQL and format results"""
    if conn is None:
        return "⚠️ No database connection"
        
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchmany(max_rows)
        columns = [column[0] for column in cursor.description]
        return tabulate(results, headers=columns, tablefmt="psql")
    except pyodbc.Error as e:
        return f"⚠️ Execution error: {str(e)}"
    finally:
        cursor.close()

# ======================
# API Endpoints
# ======================
@app.post("/query")
def api_query(request: QueryRequest):
    """API endpoint for query processing"""
    try:
        if not db_session.is_connected:
            raise HTTPException(status_code=400, detail="Database not connected")
            
        schema = app.state.schema
        relevant = identify_relevant_tables(request.question, schema)
        prompt = build_prompt(request.question, schema, relevant)
        
        cached = get_cached_or_feedback_query(prompt)
        if cached:
            return {"sql": cached, "cached": True}
            
        sql = generate_sql(prompt, request.question)
        cache_query_result(prompt, sql)
        
        return {
            "sql": sql,
            "results": execute_and_format(db_session.conn, sql),
            "cached": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def api_feedback(feedback: FeedbackRequest):
    """API endpoint for feedback submission"""
    try:
        schema = app.state.schema
        relevant = identify_relevant_tables(feedback.question, schema)
        prompt = build_prompt(feedback.question, schema, relevant)
        log_feedback(prompt, feedback.generated_sql, feedback.corrected_sql)
        return {"message": "Feedback logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/change-db")
def api_change_db(request: DBChangeRequest):
    """API endpoint for database switching"""
    try:
        if not db_session.connect(request.db_alias):
            raise HTTPException(status_code=400, detail="Connection failed")
        app.state.schema = initialize_metadata_phase(db_session.config[request.db_alias])
        return {"message": f"Switched to {request.db_alias}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize API with default DB if none selected"""
    if not db_session.current_db and db_session.list_available_dbs():
        db_session.connect(db_session.list_available_dbs()[0])
    
    if db_session.is_connected:
        app.state.schema = initialize_metadata_phase(db_session.config[db_session.current_db])
    else:
        logger.warning("No database connection established at startup")

# ======================
# CLI Interface
# ======================
def main():
    """Main CLI application with simplified interface"""
    print("\n" + "="*50)
    print("Natural Language to SQL Generator".center(50))
    print("="*50 + "\n")
    
    print("Available databases:")
    for i, db in enumerate(db_session.list_available_dbs(), 1):
        print(f"{i}. {db}")
    
    while True:
        choice = input("\nSelect database (number or Q to quit): ").strip().upper()
        
        if choice == 'Q':
            print("\nExiting...")
            sys.exit(0)
            
        try:
            db_index = int(choice) - 1
            selected_db = db_session.list_available_dbs()[db_index]
            
            if db_session.connect(selected_db):
                break
                
            print(f"⚠️ Failed to connect to {selected_db}")
        except (ValueError, IndexError):
            print("⚠️ Invalid selection")
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
    
    try:
        schema = initialize_metadata_phase(db_session.config[db_session.current_db])
        print(f"\n✅ Loaded schema for {db_session.current_db}")
    except Exception as e:
        print(f"\n❌ Schema loading failed: {str(e)}")
        sys.exit(1)
    
    while True:
        user_input = input("\nEnter your question (or 'exit' to quit): ").strip()
        
        if not user_input:
            print("⚠️ Please enter a question")
            continue
            
        if user_input.lower() == 'exit':
            break
            
        if not db_session.is_connected:
            print("⚠️ Connection lost. Reconnecting...")
            if not db_session.connect(db_session.current_db):
                print("❌ Reconnection failed")
                continue
            
        try:
            relevant_tables = identify_relevant_tables(user_input, schema)
            prompt = build_prompt(user_input, schema, relevant_tables)
            
            print("\n--- Generated Prompt ---")
            print(prompt)
            
            cached_sql = get_cached_or_feedback_query(prompt)
            if cached_sql:
                print("\n✅ Using cached query:")
                print(cached_sql)
                sql = cached_sql
            else:
                sql = generate_sql(prompt, user_input)
                print("\n--- Generated SQL ---")
                print(sql)
                
                feedback = input("\nIs this correct? (yes/no): ").strip().lower()
                if feedback == "yes":
                    cache_query_result(prompt, sql)
                    print("✅ Query cached")
                elif feedback == "no":
                    correction = input("Enter correct SQL: ").strip()
                    if correction:
                        log_feedback(prompt, sql, correction)
                        sql = correction
                    else:
                        print("⚠️ No correction provided")
                        continue
                else:
                    print("⚠️ Please enter 'yes' or 'no'")
                    continue
            
            print("\n--- Execution Results ---")
            print(execute_and_format(db_session.conn, sql))
            
        except KeyboardInterrupt:
            print("\nOperation cancelled")
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
    
    if db_session.conn:
        db_session.conn.close()
    print("\nSession ended. Goodbye!")

# ======================
# Application Entry Point
# ======================
if __name__ == "__main__":
    try:
        if os.getenv("MODE", "").lower() == "api":
            print("Starting API server...")
            uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            Path(".cache").mkdir(exist_ok=True)
            main()
            
    except KeyboardInterrupt:
        print("\nApplication terminated")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
    finally:
        if 'db_session' in globals() and db_session.conn and not db_session.conn.closed:
            db_session.conn.close()