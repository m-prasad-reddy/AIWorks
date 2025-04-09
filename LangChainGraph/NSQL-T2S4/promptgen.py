import pyodbc
import re
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load embedding and NSQL model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "NumbersStation/nsql-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
nsql_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Few-shot examples block
NSQL_FEWSHOT_EXAMPLES = '''### Database Schema
CREATE TABLE customers (customer_id int, first_name varchar, last_name varchar, email varchar);
CREATE TABLE orders (order_id int, customer_id int, order_date date);

### Question:
List all customers with their email addresses.
### SQL Query:
SELECT first_name, last_name, email FROM customers;

### Question:
Get order dates for customer John Doe.
### SQL Query:
SELECT o.order_date FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE c.first_name = 'John' AND c.last_name = 'Doe';
'''

def humanize_name(name):
    return re.sub(r'[_\-]', ' ', name).lower()

def get_table_text_for_embedding(table):
    parts = []
    if table.get("description"):
        parts.append(table["description"])
    else:
        parts.append(f"Table: {humanize_name(table['name'])}")
        for col in table["columns"]:
            col_text = f"{humanize_name(col['name'])} ({col['datatype']})"
            if col.get("description"):
                col_text += f" - {col['description']}"
            parts.append(col_text)
    return " ".join(parts)

def extract_metadata(connection_string):
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    tables_query = '''
        SELECT
            t.name AS table_name,
            c.name AS column_name,
            ty.name AS data_type,
            ep.value AS column_description,
            te.value AS table_description
        FROM sys.tables t
        JOIN sys.columns c ON t.object_id = c.object_id
        JOIN sys.types ty ON c.user_type_id = ty.user_type_id
        LEFT JOIN sys.extended_properties ep 
            ON t.object_id = ep.major_id AND c.column_id = ep.minor_id AND ep.name = 'MS_Description'
        LEFT JOIN sys.extended_properties te 
            ON t.object_id = te.major_id AND te.minor_id = 0 AND te.name = 'MS_Description'
        ORDER BY t.name, c.column_id
    '''

    cursor.execute(tables_query)
    rows = cursor.fetchall()

    metadata = {}
    for row in rows:
        table_name = row.table_name
        if table_name not in metadata:
            metadata[table_name] = {
                "name": table_name,
                "description": row.table_description,
                "columns": []
            }
        metadata[table_name]["columns"].append({
            "name": row.column_name,
            "datatype": row.data_type,
            "description": row.column_description
        })

    fk_query = '''
        SELECT 
            tp.name AS parent_table,
            cp.name AS parent_column,
            tr.name AS referenced_table,
            cr.name AS referenced_column
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
        JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id
        JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
        JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id
        JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
    '''

    cursor.execute(fk_query)
    fk_rows = cursor.fetchall()
    foreign_keys = [(r.parent_table, r.parent_column, r.referenced_table, r.referenced_column) for r in fk_rows]

    return list(metadata.values()), foreign_keys

def match_relevant_tables(user_query, metadata, threshold=0.4):
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
    matches = []
    for table in metadata:
        table_text = get_table_text_for_embedding(table)
        table_embedding = embedding_model.encode(table_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(query_embedding, table_embedding).item()
        if similarity > threshold:
            matches.append((table, similarity))
    matches.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in matches]

def generate_nsql_prompt_with_examples(user_query, matched_tables, foreign_keys):
    schema_lines = ["### Database Schema"]
    for table in matched_tables:
        column_defs = ", ".join([f"{col['name']} {col['datatype']}" for col in table['columns']])
        schema_lines.append(f"CREATE TABLE {table['name']} ({column_defs});")

    relationships = []
    for pt, pc, rt, rc in foreign_keys:
        if any(t['name'] == pt for t in matched_tables) and any(t['name'] == rt for t in matched_tables):
            relationships.append(f"-- {pt}({pc}) references {rt}({rc})")
    if relationships:
        schema_lines.append("-- Relationships")
        schema_lines.extend(relationships)

    full_prompt = NSQL_FEWSHOT_EXAMPLES + "\n" + "\n".join(schema_lines)
    full_prompt += f"\n\n### Question:\n{user_query}\n### SQL Query:"
    return full_prompt

def query_nsql_model(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = nsql_model.generate(**inputs, max_new_tokens=256, temperature=0.2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"### SQL Query:\s*(SELECT .*?);?\s*$", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else response

def main():
    connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=BikeStores;UID=sa;PWD=S0rry!43"
    print("\nWelcome to the NSQL CLI Interface!")

    while True:
        print("\nOptions:")
        print("Q - Query Database")
        print("E - Exit Application")
        choice = input("Enter your choice: ").strip().upper()

        if choice == "Q":
            user_query = input("\nEnter your natural language question: ").strip()
            if not user_query:
                print("Please enter a valid question.")
                continue
            metadata, foreign_keys = extract_metadata(connection_string)
            matched_tables = match_relevant_tables(user_query, metadata)
            prompt = generate_nsql_prompt_with_examples(user_query, matched_tables, foreign_keys)

            print("\nPrompt Sent to NSQL Model:\n")
            print(prompt)

            print("\nModel Output:\n")
            print(query_nsql_model(prompt))

        elif choice == "E":
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
