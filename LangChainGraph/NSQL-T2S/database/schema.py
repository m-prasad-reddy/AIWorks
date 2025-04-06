from collections import defaultdict

def extract_full_schema(conn):
    """Grab the full schema vibes from SQL Server, with comments included."""
    cursor = conn.cursor()
    schema_info = defaultdict(dict)

    # Fetch all tables with schema
    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE'
    """)
    tables = [(row[0], row[1]) for row in cursor.fetchall()]

    # Get column details and comments
    for schema, table in tables:
        full_table = f"{schema}.{table}"
        
        # Fetch columns and data types
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
        """)
        columns = {row[0]: {"type": row[1]} for row in cursor.fetchall()}
        
        # Fetch table-level comments
        cursor.execute(f"""
            SELECT CAST(ep.value AS NVARCHAR(512)) AS Comment
            FROM sys.extended_properties ep
            WHERE ep.class = 1 
                AND ep.minor_id = 0 
                AND ep.major_id = OBJECT_ID('{full_table}')
        """)
        table_comment = cursor.fetchone()
        table_comment = table_comment[0] if table_comment else "No description"

        # Fetch column-level comments
        cursor.execute(f"""
            SELECT c.name AS ColumnName, CAST(ep.value AS NVARCHAR(512)) AS Comment
            FROM sys.columns c
            LEFT JOIN sys.extended_properties ep 
                ON ep.major_id = c.object_id AND ep.minor_id = c.column_id
            WHERE c.object_id = OBJECT_ID('{full_table}')
        """)
        column_comments = {row[0]: row[1] for row in cursor.fetchall() if row[1]}

        # Combine into schema info
        for col in columns:
            columns[col]["comment"] = column_comments.get(col, "No description")
        schema_info[full_table] = {"columns": columns, "table_comment": table_comment}

    return schema_info

def identify_relevant_tables(schema_info, query):
    """Pick the tables that vibe with the query."""
    query_lower = query.lower()
    relevant_tables = []

    # Heuristic: match table names or related keywords
    table_keywords = {
        "orders": "sales.orders",
        "stores": "sales.stores",
        "customers": "sales.customers",
        "products": "production.products",
        "city": "sales.stores",  # Assuming city is in stores
        "shipped": "sales.orders"  # Assuming shipped relates to orders
    }

    for keyword, table in table_keywords.items():
        if keyword in query_lower and table in schema_info:
            relevant_tables.append(table)

    # Fallback: if no match, use a default table
    if not relevant_tables:
        relevant_tables.append(next(iter(schema_info.keys())))

    return list(set(relevant_tables))  # Unique tables only

def format_schema(schema_info, tables):
    """Craft a tight schema string with comments for the prompt."""
    schema_str = ""
    for table in tables:
        info = schema_info[table]
        schema_str += f"CREATE TABLE {table} (\n"
        cols = [
            f"    {col} {data['type']} -- {data['comment']}"
            for col, data in info["columns"].items()
        ]
        schema_str += ",\n".join(cols) + "\n)\n"
        schema_str += f"-- Table Description: {info['table_comment']}\n\n"
    return schema_str.strip()