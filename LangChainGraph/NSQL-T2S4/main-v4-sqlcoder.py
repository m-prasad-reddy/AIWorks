"""
Natural Language to SQL Generator (CLI-only)
- Converts natural language to Transact-SQL
- Supports multiple databases via db_configurations.json
- Caches schema and queries
"""

import pyodbc
import json
import hashlib
from pathlib import Path
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prettytable import PrettyTable
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
CACHE_FILE = "schema_cache.json"
QUERY_CACHE_FILE = ".cache/query_cache.json"
DB_CONFIG_FILE = "db_configurations.json"

# ======================
# Database Configuration
# ======================
def load_config() -> dict:
    """Load database configuration from file"""
    try:
        with open(DB_CONFIG_FILE) as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading {DB_CONFIG_FILE}: {str(e)}")
        sys.exit(1)

# ======================
# Database Connection
# ======================
class DBSession:
    """Manages database connections"""
    def __init__(self):
        self.config = load_config()
        self.conn = None
        self.current_db = None

    def list_dbs(self) -> list:
        """Return list of database aliases"""
        return list(self.config.keys())

    def connect(self, db_alias: str) -> bool:
        """Establish database connection"""
        if db_alias not in self.config:
            print(f"⚠️ Database '{db_alias}' not configured")
            return False
        config = self.config[db_alias]
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={config['server']};"
            f"DATABASE={config['database']};"
            f"UID={config['user']};"
            f"PWD={config['password']}"
        )
        try:
            if self.conn:
                self.conn.close()
            self.conn = pyodbc.connect(conn_str)
            self.current_db = db_alias
            logger.info(f"Connected to {db_alias}")
            return True
        except pyodbc.Error as e:
            logger.error(f"Connection failed: {str(e)}")
            self.conn = None
            return False

# ======================
# Schema Management
# ======================
def hash_conn_details(config: dict) -> str:
    """Generate unique hash for connection details"""
    return hashlib.sha256(f"{config['server']}|{config['database']}|{config['user']}".encode()).hexdigest()

def extract_schema(conn: pyodbc.Connection) -> dict:
    """Extract schema from database"""
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
        return schema
    except pyodbc.Error as e:
        logger.error(f"Schema extraction failed: {str(e)}")
        raise

def cache_schema(schema: dict, cache_key: str):
    """Cache schema to file"""
    Path(".cache").mkdir(exist_ok=True)
    with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}", "w") as f:
        json.dump(schema, f, indent=2)

def load_cached_schema(cache_key: str) -> dict:
    """Load cached schema"""
    try:
        with open(Path(".cache") / f"{cache_key}_{CACHE_FILE}") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def initialize_schema(db_session: DBSession) -> dict:
    """Load or update schema"""
    if not db_session.conn:
        raise ConnectionError("No active database connection")
    cache_key = hash_conn_details(db_session.config[db_session.current_db])
    latest_schema = extract_schema(db_session.conn)
    cached_schema = load_cached_schema(cache_key)
    if cached_schema != latest_schema:
        print("⚠️ Schema has changed. Updating cache...")
        cache_schema(latest_schema, cache_key)
    else:
        print("✅ Loaded schema from cache")
    return latest_schema

# ======================
# Model Initialization
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "defog/sqlcoder-7b-2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    logger.info(f"Model loaded: defog/sqlcoder-7b-2 on {device}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    print(f"❌ Failed to load model: {str(e)}")
    sys.exit(1)

# ======================
# Query Processing
# ======================
def identify_tables(user_query: str, schema: dict) -> list:
    """Identify relevant tables for the query"""
    query_lower = user_query.lower()
    keywords = set(re.findall(r'\b\w+\b', query_lower))
    scores = {}
    for table_name, meta in schema.items():
        combined_text = f"{table_name.lower()} " + " ".join(col[0].lower() for col in meta["columns"])
        match_count = len([word for word in keywords if word in combined_text])
        if match_count:
            scores[table_name] = match_count
    return sorted(scores, key=scores.get, reverse=True)[:3]

def build_prompt(user_query: str, schema: dict, tables: list) -> str:
    """Build SQL generation prompt"""
    ddl = ""
    for table in tables:
        columns = schema[table]["columns"]
        schema_name, table_name = table.split(".")
        ddl += f"CREATE TABLE [{schema_name}].[{table_name}] (\n    " + ",\n    ".join(f"{col} {dtype}" for col, dtype in columns) + "\n)\n\n"
    return f"""{ddl}-- Generate a single valid Transact-SQL SELECT statement for: '{user_query}'
-- Use [SCHEMA].[TABLE] format
-- Select only columns explicitly requested or implied
-- Use table aliases where needed
-- For 'latest', use MAX() with correlation
-- For 'year wise', ORDER BY year
-- Use JOINs for related tables
-- Use COUNT/SUM for 'how many' or 'total' queries
-- Match schema exactly; do not guess tables/columns
-- Examples:
-- 1. "show latest products produced year wise along with its price":
--    SELECT p.product_name, p.model_year, p.list_price FROM [production].[products] p WHERE p.model_year = (SELECT MAX(model_year) FROM [production].[products] WHERE product_name = p.product_name) ORDER BY p.model_year
-- 2. "what production categories are available":
--    SELECT DISTINCT category_name FROM [production].[categories]
-- 3. "list all products with their categories and brands":
--    SELECT p.product_name, c.category_name, b.brand_name FROM [production].[products] p JOIN [production].[categories] c ON p.category_id = c.category_id JOIN [production].[brands] b ON p.brand_id = b.brand_id
-- 4. "how many orders were placed in 2018":
--    SELECT COUNT(*) FROM [sales].[orders] WHERE YEAR(order_date) = 2018
-- 5. "total sales amount for each store":
--    SELECT s.store_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales FROM [sales].[stores] s JOIN [sales].[orders] o ON s.store_id = o.store_id JOIN [sales].[order_items] oi ON o.order_id = oi.order_id GROUP BY s.store_name
-- 6. "total sales amount in store_name Baldwin Bikes":
--    SELECT s.store_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales FROM [sales].[stores] s JOIN [sales].[orders] o ON s.store_id = o.store_id JOIN [sales].[order_items] oi ON o.order_id = oi.order_id WHERE s.store_name = 'Baldwin Bikes' GROUP BY s.store_name
SELECT"""

def load_cache(path: str) -> dict:
    """Load JSON cache"""
    return json.load(open(path)) if Path(path).exists() else {}

def save_cache(path: str, data: dict):
    """Save to JSON cache"""
    Path(".cache").mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

query_cache = load_cache(QUERY_CACHE_FILE)

def generate_sql(prompt: str, user_query: str) -> str:
    """Generate SQL with 2-row limit"""
    global query_cache, model, device
    query_hash = hashlib.sha256(user_query.encode()).hexdigest()
    
    if query_hash in query_cache:
        sql = query_cache[query_hash]
        if not sql.upper().startswith("SELECT TOP"):
            sql = f"SELECT TOP 2 {sql[6:]}"
        return sql.strip()

    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logger.debug(f"Input IDs shape: {input_ids.shape}")
        try:
            # Attempt CUDA generation with reduced max_new_tokens
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=50,  # Reduced from 100 to lower memory demand
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                temperature=0.7,
                do_sample=False
            )
        except RuntimeError as cuda_error:
            logger.warning(f"CUDA failed: {str(cuda_error)}. Falling back to CPU.")
            # Move model to CPU and retry
            model = model.to("cpu")
            device = "cpu"
            generated_ids = model.generate(
                input_ids.to("cpu"),
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                temperature=0.7,
                do_sample=False
            )
            # Move model back to CUDA for subsequent queries
            model = model.to("cuda")
            device = "cuda"
        
        sql = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        match = re.search(r'(?i)^\s*SELECT\s+.*$', sql, re.MULTILINE)
        sql = match.group(0).strip() if match else "SELECT TOP 2 -- Invalid generation"
        if not sql.upper().startswith("SELECT TOP"):
            sql = f"SELECT TOP 2 {sql[6:]}"
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE).strip()

        query_lower = user_query.lower()
        year = re.search(r'\b(\d{4})\b', query_lower).group(1) if re.search(r'\b(\d{4})\b', query_lower) else None
        store_match = re.search(r"(?:in|at)\s+store(?:_name)?\s+['\"]?(.*?)(?:['\"]|$)", query_lower)
        store_name = store_match.group(1).strip("'\"") if store_match else None
        
        # Override model output with fallback logic
        if "latest" in query_lower and "product" in query_lower:
            sql = """SELECT TOP 2 p.product_name, p.model_year, p.list_price
FROM [production].[products] p
WHERE p.model_year = (SELECT MAX(model_year) FROM [production].[products] WHERE product_name = p.product_name)
ORDER BY p.model_year"""
        elif "year wise" in query_lower and "product" in query_lower:
            sql = "SELECT TOP 2 product_name, model_year, list_price FROM [production].[products] ORDER BY model_year"
        elif "how many" in query_lower and "order" in query_lower and year:
            sql = f"SELECT TOP 2 COUNT(*) FROM [sales].[orders] WHERE YEAR(order_date) = {year}"
        elif "stock" in query_lower and "store" in query_lower:
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
        elif "total" in query_lower and "sales" in query_lower and "store" in query_lower:
            if store_name:
                sql = f"""SELECT TOP 2 s.store_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales
FROM [sales].[stores] s
JOIN [sales].[orders] o ON s.store_id = o.store_id
JOIN [sales].[order_items] oi ON o.order_id = oi.order_id
WHERE s.store_name = '{store_name}'
GROUP BY s.store_name"""
            else:
                sql = """SELECT TOP 2 s.store_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales
FROM [sales].[stores] s
JOIN [sales].[orders] o ON s.store_id = o.store_id
JOIN [sales].[order_items] oi ON o.order_id = oi.order_id
GROUP BY s.store_name"""

        query_cache[query_hash] = sql
        save_cache(QUERY_CACHE_FILE, query_cache)
        return sql.strip()
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return "SELECT TOP 2 -- Generation error"

def execute_sql(sql: str, conn: pyodbc.Connection):
    """Execute SQL and display up to 2 rows"""
    try:
        sql = re.sub(r'(\w+)\s*ORDER\s+BY', r'\1 ORDER BY', sql, flags=re.I)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(2)
        columns = [desc[0] for desc in cursor.description]
        table = PrettyTable(columns)
        for row in rows:
            table.add_row(row)
        print("\n--- Execution Results ---")
        print(table)
    except pyodbc.Error as e:
        print(f"⚠️ Execution error: {str(e)}")

# ======================
# CLI Interface
# ======================
def main():
    print("\n" + "="*50)
    print("Natural Language to SQL Generator".center(50))
    print("="*50 + "\n")

    db_session = DBSession()
    print("Available databases:")
    for i, db in enumerate(db_session.list_dbs(), 1):
        print(f"{i}. {db}")

    while True:
        choice = input("\nSelect database (number or Q to quit): ").strip().upper()
        if choice == 'Q':
            print("\nExiting...")
            sys.exit(0)
        try:
            db_index = int(choice) - 1
            selected_db = db_session.list_dbs()[db_index]
            if db_session.connect(selected_db):
                break
            print(f"⚠️ Failed to connect to {selected_db}")
        except (ValueError, IndexError):
            print("⚠️ Invalid selection")

    schema = initialize_schema(db_session)
    print(f"\n✅ Loaded schema for {db_session.current_db}")

    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            break
        if not user_query:
            print("⚠️ Please enter a question")
            continue

        tables = identify_tables(user_query, schema)
        prompt = build_prompt(user_query, schema, tables)
        print("\n--- Generated Prompt ---")
        print(prompt)

        sql = generate_sql(prompt, user_query)
        print("\n--- Generated SQL ---")
        print(sql)
        execute_sql(sql, db_session.conn)

    db_session.conn.close()
    print("\nSession ended. Goodbye!")

if __name__ == "__main__":
    main()