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
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for raw output visibility
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
                if keyword == "city" and "order" in query_lower and table_name == "sales.stores":
                    match_count += 5
                else:
                    match_count += 3
        
        if match_count:
            scores[table_name] = match_count
    
    return sorted(scores, key=scores.get, reverse=True)[:3]

def build_prompt(user_query: str, schema: Dict[str, Any], selected_tables: List[str]) -> str:
    """Build prompt with diverse examples and strict instructions"""
    ddl_section = ""
    for table in selected_tables:
        columns = schema[table]["columns"]
        schema_name, table_name = table.split(".")
        ddl = f"CREATE TABLE [{schema_name}].[{table_name}] (\n"
        for col, dtype in columns:
            ddl += f"    {col} {dtype},\n"
        if "foreign_keys" in schema[table]:
            ddl += "\n    -- Foreign Keys:\n"
            for fk in schema[table]["foreign_keys"]:
                ddl += f"    -- FK: {fk['column']} → {fk['references']}\n"
        if "indexes" in schema[table]:
            ddl += "\n    -- Indexes:\n"
            for idx in schema[table]["indexes"]:
                ddl += f"    -- IDX: {idx['name']} ({idx['column']})\n"
        ddl = ddl.rstrip(",\n") + "\n)\n\n"
        ddl_section += ddl
    
    return f"""{ddl_section}-- Generate a valid Transact-SQL SELECT statement based on the query below:
-- Use [SCHEMA].[TABLE] format (e.g., [production].[products])
-- Select only the columns explicitly requested or implied by the query
-- Use table aliases (e.g., p for [production].[products]) where appropriate
-- For 'latest', use a subquery with MAX() and correlate by a unique column (e.g., WHERE p.model_year = (SELECT MAX(model_year) ...))
-- For 'year wise', ORDER BY the year column
-- Match the query intent to the schema provided above, do NOT copy examples blindly
-- Examples:
-- 1. "show latest products produced year wise along with its price":
--    SELECT p.product_name, p.model_year, p.list_price
--    FROM [production].[products] p
--    WHERE p.model_year = (SELECT MAX(model_year) FROM [production].[products] WHERE product_name = p.product_name)
--    ORDER BY p.model_year
-- 2. "what production categories are available":
--    SELECT DISTINCT category_name FROM [production].[categories]
-- 3. "show product names produced year wise":
--    SELECT product_name, model_year FROM [production].[products] ORDER BY model_year
-- Query: {user_query}
SELECT"""

# Initialize model
tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-6B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-6B")

def generate_sql(prompt: str) -> str:
    """Generate and clean SQL"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id)
    sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Log raw output for debugging
    logger.debug(f"Raw model output: {sql}")
    
    # Extract the first valid SELECT statement
    lines = sql.splitlines()
    select_lines = []
    for line in lines:
        if line.strip().upper().startswith("SELECT"):
            select_lines.append(line)
            break
    for line in lines[len(select_lines):]:
        if line.strip() and not line.strip().startswith("--"):
            select_lines.append(line)
        elif line.strip().startswith("SELECT"):
            break
    
    sql = "\n".join(select_lines).strip()
    
    # Basic cleanup
    if "[" not in sql or "]" not in sql:
        sql = sql.replace("products", "[production].[products]").replace("categories", "[production].[categories]")
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE).strip()
    
    return sql.rstrip(';').strip()

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
    """Cache generated SQL query"""
    query_cache = load_json(QUERY_CACHE_FILE)
    query_cache[hashlib.sha256(prompt.encode()).hexdigest()] = sql
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
            
        sql = generate_sql(prompt)
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
                sql = generate_sql(prompt)
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
                        print("✅ Correction saved")
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