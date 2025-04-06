import pyodbc
import re
import json
from pathlib import Path
from typing import Dict, Any, List

def extract_schema_metadata(conn: pyodbc.Connection) -> Dict[str, Any]:
    """Grab the schema vibes with all the juicy details."""
    cursor = conn.cursor()
    schema_data = {}

    # Columns and datatypes
    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
    """)
    for row in cursor.fetchall():
        schema, table, col, dtype = row
        key = f"{schema}.{table}"
        if key not in schema_data:
            schema_data[key] = {"columns": [], "table_description": "", "column_descriptions": {}}
        schema_data[key]["columns"].append((col, dtype))

    # Table comments
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

    # Column comments
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

def create_table_hints_cache(schema: Dict[str, Any], cache_path: str = ".cache/table_hints.json") -> Dict[str, List[str]]:
    """Generate and cache table hints based on BikeStores schema vibes."""
    Path(".cache").mkdir(exist_ok=True)
    table_hints = {}

    # BikeStores-specific mappings
    keyword_mappings = {
        "order": ["sales.orders", "sales.order_items"],
        "city": ["sales.customers", "sales.stores"],  # Prioritize customers for city
        "ship": ["sales.orders"],  # shipped_date is in orders
        "date": ["sales.orders"],  # order_date, shipped_date
        "customer": ["sales.customers"],
        "store": ["sales.stores"],
        "product": ["production.products", "sales.order_items"]
    }

    for table_name, meta in schema.items():
        table_lower = table_name.lower()
        cols_lower = [col[0].lower() for col in meta["columns"]]
        desc_lower = meta.get("table_description", "").lower()

        for keyword, triggers in keyword_mappings.items():
            # Check if table matches predefined triggers or has keyword in metadata
            if table_name in triggers or \
               any(keyword in col for col in cols_lower) or \
               keyword in desc_lower:
                if keyword not in table_hints:
                    table_hints[keyword] = []
                if table_name not in table_hints[keyword]:
                    table_hints[keyword].append(table_name)

    # Save to JSON cache
    with open(cache_path, "w") as f:
        json.dump(table_hints, f, indent=2)
    print(f"✅ Table hints cached to {cache_path}. Edit it if you vibe with tweaks!")

    return table_hints

def load_table_hints_cache(cache_path: str = ".cache/table_hints.json") -> Dict[str, List[str]]:
    """Load the table hints from cache, or return empty if missing."""
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def identify_relevant_tables(user_query: str, schema: Dict[str, Any], hints_cache_path: str = ".cache/table_hints.json") -> List[str]:
    """Pick the tables that vibe with the query, using cached hints."""
    query_lower = user_query.lower()
    keywords = set(re.findall(r'\b\w+\b', query_lower))
    table_hints = load_table_hints_cache(hints_cache_path)
    
    scores = {}
    for table_name, meta in schema.items():
        combined_text = (
            f"{table_name.lower()} "
            + " ".join([col[0].lower() for col in meta["columns"]])
            + " " + meta.get("table_description", "").lower()
        )
        
        # Base score from general keyword matches
        match_count = sum(1 for word in keywords if word in combined_text)
        
        # Boost score with hints from cache
        for keyword, tables in table_hints.items():
            if keyword in query_lower and table_name in tables:
                match_count += 3  # Higher weight for hint matches
        
        if match_count:
            scores[table_name] = match_count

    # Return top 3 tables, or fallback
    relevant = sorted(scores, key=scores.get, reverse=True)[:3]
    return relevant if relevant else [next(iter(schema.keys()))]

def format_schema(schema: Dict[str, Any], tables: List[str]) -> str:
    """Craft a tight schema string with [SCHEMA].[TABLE] vibes."""
    schema_str = ""
    for table in tables:
        meta = schema[table]
        schema_str += f"CREATE TABLE [{table.split('.')[0]}].[{table.split('.')[1]}] (\n"
        cols = [
            f"    {col} {dtype} -- {meta['column_descriptions'].get(col, 'No description')}"
            for col, dtype in meta["columns"]
        ]
        schema_str += ",\n".join(cols) + "\n)\n"
        schema_str += f"-- Table Description: {meta['table_description'] or 'No description'}\n\n"
    return schema_str.strip()