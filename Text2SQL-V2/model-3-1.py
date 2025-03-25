import spacy
import pyodbc
import sqlserverconnection as sqlconn
from dateutil import parser as date_parser
import re
# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Step 2: Metadata Extraction
def get_metadata(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
    tables = [row[0] for row in cursor.fetchall()]
    metadata = {}
    date_columns = {}
    numeric_columns = {}
    for table in tables:
        cursor.execute(f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'")
        columns = cursor.fetchall()
        metadata[table] = {col[0]: col[1] for col in columns}
        date_cols = [col[0] for col in columns if col[1] in ['date', 'datetime', 'smalldatetime']]
        date_columns[table] = date_cols
        num_cols = [col[0] for col in columns if col[1] in ['int', 'decimal', 'float', 'money', 'numeric']]
        numeric_columns[table] = num_cols
    return metadata, date_columns, numeric_columns

# Step 3: Natural Language Processing
class TextToSQL:
    def __init__(self, metadata, date_columns, numeric_columns):
        self.nlp = nlp
        self.metadata = metadata
        self.date_columns = date_columns
        self.numeric_columns = numeric_columns

    def preprocess(self, text):
        doc = self.nlp(text.lower())
        tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
        return tokens

    def get_intent(self, question):
        if 'how many' in question.lower():
            return 'COUNT'
        elif 'total' in question.lower():
            return 'SUM'
        else:
            return 'SELECT'

    def parse_date_entities(self, question):
        doc = self.nlp(question)
        dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
        return dates

    def identify_table(self, question):
        tokens = self.preprocess(question)
        for table in self.metadata.keys():
            if table.lower() in tokens or (table.lower() + 's') in tokens:
                return table
        return None

    def get_fields(self, question, table):
        if self.get_intent(question) in ['SUM', 'AVG']:
            if 'amount' in question.lower() and 'Orders' in table:
                return ['TotalAmount']
            else:
                return ['*']
        else:
            tokens = self.preprocess(question)
            selected_fields = []
            field_map = {
                'name': ['FirstName', 'LastName'],
                'email': 'Email',
                # Add more as needed
            }
            for keyword, fields in field_map.items():
                if keyword in tokens:
                    if isinstance(fields, list):
                        selected_fields.extend([field for field in fields if field in self.metadata[table]])
                    else:
                        if fields in self.metadata[table]:
                            selected_fields.append(fields)
            return selected_fields if selected_fields else ['*']
    keyword_mapping = {
        "name": ["FirstName", "LastName"],
        "contact": ["Email", "PhoneNumber"],
        "date": ["JoinDate", "OrderDate"],
        "amount": ["TotalAmount"],
        "order details": ["OrderID", "OrderDate", "TotalAmount"],
        "email": ["Email"],
        "phone": ["PhoneNumber"]
    }
    
    def generate_date_condition(self, date_str, date_column):
        try:
            date_obj = date_parser.parse(date_str)
            if date_obj.year and date_obj.month and not date_obj.day:
                # Year and month specified
                start_date = date_obj.replace(day=1)
                if date_obj.month == 12:
                    end_date = date_obj.replace(year=date_obj.year + 1, month=1, day=1)
                else:
                    end_date = date_obj.replace(month=date_obj.month + 1, day=1)
                return f"{date_column} >= '{start_date.strftime('%Y-%m-%d')}' AND {date_column} < '{end_date.strftime('%Y-%m-%d')}'"
            elif date_obj.year and not date_obj.month and not date_obj.day:
                # Only year
                start_date = date_obj.replace(month=1, day=1)
                end_date = date_obj.replace(year=date_obj.year + 1, month=1, day=1)
                return f"{date_column} >= '{start_date.strftime('%Y-%m-%d')}' AND {date_column} < '{end_date.strftime('%Y-%m-%d')}'"
            elif date_obj.year and date_obj.month and date_obj.day:
                # Specific date
                return f"{date_column} = '{date_obj.strftime('%Y-%m-%d')}'"
            else:
                return ''
        except ValueError:
            return ''

    def generate_sql(self, question):
        table = self.identify_table(question)
        if not table:
            return "Error: No table identified"
        intent = self.get_intent(question)
        fields = self.get_fields(question, table)
        
        if intent == 'COUNT':
            select_clause = "SELECT COUNT(*)"
        elif intent in ['SUM', 'AVG']:
            if self.numeric_columns[table]:
                field = self.numeric_columns[table][0]
                select_clause = f"SELECT {intent}({field})"
            else:
                return "Error: No numeric columns for aggregation"
        else:
            select_clause = f"SELECT {', '.join(fields)}"
            
        from_clause = f"FROM {table}"
        # Date conditions
        date_entities = self.parse_date_entities(question)
        where_clause = ""
        if date_entities and self.date_columns[table]:
            date_column = self.date_columns[table][0]  # Use first date column
            conditions = [self.generate_date_condition(date_str, date_column) for date_str in date_entities]
            conditions = [cond for cond in conditions if cond]
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
        return f"{select_clause} {from_clause} {where_clause}".strip()

    def execute_query(self, sql, conn):
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except Exception as e:
            return f"Error: {str(e)}"

# Step 4: Training Data Creation
def create_training_data():
    training_data = [
        ("how many customers joined in 2023", "SELECT COUNT(*) FROM Customers WHERE JoinDate >= '2023-01-01' AND JoinDate < '2024-01-01'"),
        ("what is the total amount of orders", "SELECT SUM(TotalAmount) FROM Orders"),
        ("show me customer names", "SELECT FirstName, LastName FROM Customers"),
        ("list orders from 2023", "SELECT * FROM Orders WHERE OrderDate >= '2023-01-01' AND OrderDate < '2024-01-01'"),
        ("Show me all customer names and their join dates", "SELECT FirstName, LastName, JoinDate FROM Customers"),
        ("What is the average order amount in 2023?", "SELECT AVG(TotalAmount) FROM Orders WHERE OrderDate >= '2023-01-01' AND OrderDate < '2024-01-01'"),
        #("List all orders placed by Jane Smith", "SELECT * FROM Orders WHERE CustomerID = (SELECT CustomerID FROM Customers WHERE FirstName = 'Jane' AND LastName = 'Smith')"),
        ("List all orders placed by Jane Smith", "SELECT * FROM Orders o JOIN Customers c ON o.CustomerID = c.CustomerID WHERE c.FirstName = 'Jane' AND c.LastName = 'Smith'"),
        ("Get the email addresses of all customers", "SELECT Email FROM Customers"),
        ("How many orders were placed in June 2023?", "SELECT COUNT(*) FROM Orders WHERE OrderDate >= '2023-06-01' AND OrderDate < '2023-07-01'"),
    ]
    return training_data

# Step 5: Main Function to Tie Everything Together
def main():
    # Connect to the database
    # conn = pyodbc.connect(
    #     'DRIVER={SQL Server};'
    #     'SERVER=your_server_name;'
    #     'DATABASE=SalesDB;'
    #     'Trusted_Connection=yes;'
    # )
    
    # Step 1: Database Setup and Connection
    conn = sqlconn.get_sql_connection()

    # Get metadata
    metadata, date_columns, numeric_columns = get_metadata(conn)

    # Initialize TextToSQL model
    converter = TextToSQL(metadata, date_columns, numeric_columns)

    # Create training data (for reference or future use)
    training_data = create_training_data()
    print("Training Data Created:", training_data)

    # Test questions
    test_questions = [
        "how many customers joined in 2023",
        "what is the total amount of orders",
        "show me customer names",
        "list orders from 2023"
    ]

    for question in test_questions:
        sql = converter.generate_sql(question)
        print(f"Question: {question}")
        print(f"SQL: {sql}")
        results = converter.execute_query(sql, conn)
        print(f"Results: {results}\n")

    conn.close()

if __name__ == "__main__":
    main()