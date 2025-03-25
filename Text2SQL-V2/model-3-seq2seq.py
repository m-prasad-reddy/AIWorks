import sqlserverconnection as sqlconn
import spacy
from dateutil import parser as date_parser
from collections import defaultdict
import re
from date_handler import DateHandler

# Load spaCy model for natural language processing
nlp = spacy.load("en_core_web_sm")

# Step 1: Create Sample Database and Insert Data
print("Sample database 'SalesDB' created and populated with data.")

# Step 2: Extract Database Metadata
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
        date_columns[table] = [(col[0], col[1]) for col in columns if col[1] in ['date', 'datetime', 'smalldatetime']]
        numeric_columns[table] = [col[0] for col in columns if col[1] in ['int', 'decimal', 'float', 'money', 'numeric']]
    # print(metadata, date_columns, numeric_columns)
    return metadata, date_columns, numeric_columns

# Step 3: Create Training Data
def create_training_data():
    """Generate a list of question-SQL pairs for training."""
    return  [
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

# Step 4: Text-to-SQL Conversion Class with Training Data Integration
class TextToSQL:
    def __init__(self, metadata, date_columns, numeric_columns, training_data):
        self.nlp = nlp
        self.metadata = metadata
        self.date_columns = date_columns
        self.numeric_columns = numeric_columns
        self.training_data = training_data
        self.intent_map = defaultdict(lambda: 'SELECT')
        self.field_map = defaultdict(list)
        self.table_map = defaultdict(lambda: None)
        self.date_handler = DateHandler()
        self._learn_from_training_data()

    def _learn_from_training_data(self):
        for question, sql in self.training_data:
            question_lower = question.lower()
            if 'how many' in question_lower:
                self.intent_map['how many'] = 'COUNT'
            elif 'total' in question_lower:
                self.intent_map['total'] = 'SUM'
            elif 'average' in question_lower:
                self.intent_map['average'] = 'AVG'
            elif 'show' in question_lower or 'list' in question_lower:
                self.intent_map['show'] = 'SELECT'
                self.intent_map['list'] = 'SELECT'

            tokens = self.preprocess(question)
            for table in self.metadata.keys():
                table_lower = table.lower()
                self.table_map[table_lower] = table
                self.table_map[table_lower + 's'] = table
                if table_lower in tokens or (table_lower + 's') in tokens:
                    self.table_map[table_lower] = table

            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
            if select_match:
                fields_str = select_match.group(1)
                if fields_str != '*':
                    fields = [f.strip() for f in fields_str.split(',')]
                    for token in tokens:
                        if token not in ['how', 'many', 'what', 'is', 'in', 'from', 'did', 'place']:
                            for field in fields:
                                if field in self.metadata.get(self.identify_table(question), {}):
                                    self.field_map[token].append(field)

    def preprocess(self, text):
        doc = self.nlp(text.lower())
        tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
        return tokens

    def get_intent(self, question):
        question_lower = question.lower()
        for phrase in self.intent_map:
            if phrase in question_lower:
                return self.intent_map[phrase]
        return 'SELECT'

    # def parse_date_entities(self, question):
    #     doc = self.nlp(question)
    #     dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
    #     return dates

    def parse_date_entities(self, question):
        return self.date_handler.parse_date_entities(question, self.nlp)
    
    # def identify_table(self, question):
    #     tokens = self.preprocess(question)
    #     question_lower = question.lower()
    #     for table in self.metadata.keys():
    #         if table.lower() in question_lower or (table.lower() + 's') in question_lower:
    #             return table
    #     return list(self.metadata.keys())[0]  # Default to first table

    def identify_table(self, question):
        """Identify the table referenced in the question."""
        tokens = self.preprocess(question)
        for token in tokens:
            if token in self.table_map:
                return self.table_map[token]
        return list(self.metadata.keys())[0]  # Default to first table if none found

    def get_fields(self, question, table):
        intent = self.get_intent(question)
        tokens = self.preprocess(question)
        selected_fields = []

        if intent in ['SUM', 'AVG']:
            for token in tokens:
                if token in ['amount', 'total']:
                    for field in self.field_map.get(token, []):
                        if field in self.numeric_columns[table]:
                            return [field]
            return [self.numeric_columns[table][0]] if self.numeric_columns[table] else ['*']
        
        for token in tokens:
            if token in self.field_map:
                selected_fields.extend([f for f in self.field_map[token] if f in self.metadata[table]])
        
        return selected_fields if selected_fields else ['*']

    # def generate_date_condition(self, date_str, date_column):
    #     try:
    #         date_obj = date_parser.parse(date_str, default=date_parser.parse('2000-01-01'))
    #         if date_str.lower() in ['last year', 'this year', 'next year']:
    #             return ''
    #         elif date_obj.year and date_obj.month and not date_obj.day:
    #             start_date = date_obj.replace(day=1)
    #             end_date = start_date.replace(month=start_date.month % 12 + 1, year=start_date.year + (start_date.month // 12))
    #             return f"{date_column} >= '{start_date.strftime('%Y-%m-%d')}' AND {date_column} < '{end_date.strftime('%Y-%m-%d')}'"
    #         elif date_obj.year and not date_obj.month:
    #             start_date = date_obj.replace(month=1, day=1)
    #             end_date = date_obj.replace(year=date_obj.year + 1, month=1, day=1)
    #             return f"{date_column} >= '{start_date.strftime('%Y-%m-%d')}' AND {date_column} < '{end_date.strftime('%Y-%m-%d')}'"
    #         else:
    #             return f"{date_column} = '{date_obj.strftime('%Y-%m-%d')}'"
    #     except ValueError:
    #         return ''

    def generate_date_condition(self, date_str, date_column):
        # Kept for compatibility, but not used directly
        return self.date_handler.generate_date_condition(date_str, date_column, 'date')
    
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
                field = fields[0] if fields != ['*'] else self.numeric_columns[table][0]
                select_clause = f"SELECT {intent}({field})"
            else:
                return "Error: No numeric columns for aggregation"
        else:
            select_clause = f"SELECT {', '.join(fields)}"

        from_clause = f"FROM {table}"
        where_clause = ""
        
        # date_entities = self.parse_date_entities(question)
        # if date_entities and self.date_columns[table]:
        #     date_column = self.date_columns[table][0]
        #     conditions = [self.generate_date_condition(date_str, date_column) for date_str in date_entities]
        #     conditions = [cond for cond in conditions if cond]
        #     if conditions:
        #         where_clause = "WHERE " + " AND ".join(conditions)
        
        if self.date_columns[table]:
            date_column, column_type = self.date_columns[table][0]  # Get column name and type
            where_clause = self.date_handler.generate_conditions(question, date_column, column_type, self.nlp)
            
        sql = f"{select_clause} {from_clause} {where_clause}".strip()
        return sql if not sql.startswith("Error") else sql

    def execute_query(self, sql, conn):
        if sql.startswith("Error"):
            return sql
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except Exception as e:
            return f"Error: {str(e)}"

# Step 5: Main Function to Run the Program
def main():
    """Main function to set up database, integrate training data, and test queries."""
    #create_sample_database()

    conn = sqlconn.get_sql_connection()

    metadata, date_columns, numeric_columns = get_metadata(conn)
    training_data = create_training_data()

    converter = TextToSQL(metadata, date_columns, numeric_columns, training_data)

    print("\n--- Training Data ---")
    for question, sql in training_data:
        print(f"Question: {question}")
        print(f"SQL: {sql}\n")

    test_questions = [
        "how many customers joined in 2023",
        "what is the total amount of orders in 2023",
        "show me customer names and emails",
        "list orders from February 2023",
        "what is the average order amount in 2023"
    ]

    print("\n--- Testing Text-to-SQL ---")
    for question in test_questions:
        sql = converter.generate_sql(question)
        print(f"Question: {question}")
        print(f"Generated SQL: {sql}")
        results = converter.execute_query(sql, conn)
        print(f"Results: {results}\n")

    conn.close()

if __name__ == "__main__":
    main()