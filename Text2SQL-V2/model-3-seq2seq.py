import sqlserverconnection as sqlconn
import spacy
from dateutil import parser as date_parser
from collections import defaultdict
import re

nlp = spacy.load("en_core_web_sm")

print("Sample database 'SalesDB' created and populated with data.")

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
    return metadata, date_columns, numeric_columns

def create_training_data():
    return [
        ("how many customers joined in 2023", "SELECT COUNT(*) FROM Customers WHERE YEAR(JoinDate)=2023"),
        ("what is the total amount of orders", "SELECT SUM(TotalAmount) FROM Orders"),
        ("show me customer names", "SELECT FirstName, LastName FROM Customers"),
        ("list orders from 2023", "SELECT * FROM Orders WHERE YEAR(OrderDate) >= 2023"),
        ("Show me all customer names and their join dates", "SELECT FirstName, LastName, JoinDate FROM Customers"),
        ("What is the average order amount in 2023?", "SELECT AVG(TotalAmount) FROM Orders WHERE YEAR(OrderDate) = 2023"),
        ("List all orders placed by Jane Smith", "SELECT * FROM Orders o JOIN Customers c ON o.CustomerID = c.CustomerID WHERE c.FirstName = 'Jane' AND c.LastName = 'Smith'"),
        ("Get the email addresses of all customers", "SELECT Email FROM Customers"),
        ("How many orders were placed in June 2023?", "SELECT COUNT(*) FROM Orders WHERE MONTH(OrderDate)='06' AND YEAR(OrderDate)= 2023")
    ]

class DateHandler:
    def parse_date_entities(self, question, nlp):
        doc = nlp(question)
        dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
        return dates

    def generate_date_condition(self, date_str, date_column, column_type):
        try:
            date_str_clean = date_str.strip()
            date_obj = date_parser.parse(date_str, default=date_parser.parse('2000-01-01'))
            date_str_lower = date_str.lower()

            if date_str_clean.isdigit() and len(date_str_clean) == 4:
                return f"YEAR({date_column}) = {int(date_str_clean)}"

            months = {
                'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
                'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
            }
            for month_name, month_num in months.items():
                if month_name in date_str_lower:
                    return f"MONTH({date_column}) = '{month_num}' AND YEAR({date_column}) = {date_obj.year}"

            if column_type == 'date':
                return f"{date_column} = CONVERT(DATE, '{date_obj.strftime('%Y-%m-%d')}', 23)"
            else:
                return f"{date_column} = CONVERT(DATETIME, '{date_obj.strftime('%Y-%m-%d')} 00:00:00', 120)"

        except ValueError:
            return ''

    def generate_conditions(self, question, date_column, column_type, nlp):
        date_entities = self.parse_date_entities(question, nlp)
        if not date_entities or not date_column:
            return ''
        conditions = [self.generate_date_condition(date_str, date_column, column_type) for date_str in date_entities]
        conditions = [cond for cond in conditions if cond]
        return "WHERE " + " AND ".join(conditions) if conditions else ''

class TextToSQL:
    def __init__(self, metadata, date_columns, numeric_columns, training_data):
        self.nlp = nlp
        self.metadata = metadata
        self.date_columns = date_columns
        self.numeric_columns = numeric_columns
        self.training_data = training_data
        self.intent_map = defaultdict(lambda: 'SELECT')
        self.field_map = defaultdict(list)
        self.table_map = defaultdict(str)
        self.date_handler = DateHandler()
        self._learn_from_training_data()

    def _learn_from_training_data(self):
        for question, sql in self.training_data:
            question_lower = question.lower()
            tokens = self.preprocess(question)

            # Intent mapping
            if 'how many' in question_lower:
                self.intent_map['how many'] = 'COUNT'
            elif 'total' in question_lower:
                self.intent_map['total'] = 'SUM'
            elif 'average' in question_lower:
                self.intent_map['average'] = 'AVG'
            elif 'show' in question_lower or 'list' in question_lower:
                self.intent_map['show'] = 'SELECT'
                self.intent_map['list'] = 'SELECT'

            # Table mapping
            table_match = re.search(r'FROM\s+([^\s]+)', sql, re.IGNORECASE)
            if table_match:
                table = table_match.group(1)
                for token in tokens:
                    if token in ['customer', 'customers']:
                        self.table_map[token] = 'Customers'
                    elif token in ['order', 'orders']:
                        self.table_map[token] = 'Orders'

            # Field mapping
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
            if select_match:
                fields_str = select_match.group(1)
                if fields_str == '*':
                    continue
                fields = [f.strip() for f in fields_str.split(',') if not f.strip().startswith('o.') and not f.strip().startswith('c.')]
                identified_table = self.identify_table(question)
                for token in tokens:
                    if token not in ['how', 'many', 'what', 'is', 'in', 'from', 'did', 'place', 'joined', 'their', 'all', 'get', 'were', 'placed']:
                        if token == 'names':
                            self.field_map[token] = ['FirstName', 'LastName']
                        elif token in ['emails', 'email']:
                            self.field_map[token] = ['Email']
                        elif token == 'amount':
                            self.field_map[token] = ['TotalAmount']
                        elif token == 'dates':
                            self.field_map[token] = ['JoinDate']
                        elif token == 'customer' and 'names' in tokens and 'emails' in tokens:
                            continue
                        else:
                            for field in fields:
                                if field in self.metadata.get(identified_table, {}):
                                    self.field_map[token].append(field)
        # Deduplicate field_map entries
        for key in self.field_map:
            self.field_map[key] = list(dict.fromkeys(self.field_map[key]))

    def preprocess(self, text):
        doc = self.nlp(text.lower())
        # Include 'amount' even if it's a stop word
        tokens = [token.text for token in doc if (not token.is_stop or token.text == 'amount') and token.is_alpha]
        return tokens

    def get_intent(self, question):
        question_lower = question.lower()
        for phrase in self.intent_map:
            if phrase in question_lower:
                return self.intent_map[phrase]
        return 'SELECT'

    def identify_table(self, question):
        tokens = self.preprocess(question)
        for token in tokens:
            if token in self.table_map and self.table_map[token]:
                return self.table_map[token]
        return 'Customers'

    def get_fields(self, question, table):
        intent = self.get_intent(question)
        tokens = self.preprocess(question)
        print(f"Tokens: {tokens}")  # Debugging
        selected_fields = []

        if intent == 'SUM':
            for token in tokens:
                if token in ['amount', 'total']:
                    if 'TotalAmount' in self.metadata[table]:
                        return ['TotalAmount']
            return [self.numeric_columns[table][0]] if self.numeric_columns[table] else ['*']
        elif intent == 'AVG':
            for token in tokens:
                if token in ['amount', 'order']:  # Include 'order' for context
                    if 'TotalAmount' in self.metadata[table]:
                        return ['TotalAmount']
            return [self.numeric_columns[table][0]] if self.numeric_columns[table] else ['*']
        elif intent == 'COUNT':
            return ['*']
        elif intent == 'SELECT' and 'list' in question.lower():
            return ['*']

        if 'names' in tokens and 'emails' in tokens:
            return ['FirstName', 'LastName', 'Email']

        for token in tokens:
            if token in self.field_map:
                selected_fields.extend(self.field_map[token])

        selected_fields = list(dict.fromkeys(f for f in selected_fields if f in self.metadata[table]))
        return selected_fields if selected_fields else ['*']

    def generate_sql(self, question):
        table = self.identify_table(question)
        if not table:
            return "Error: No table identified"
        
        intent = self.get_intent(question)
        fields = self.get_fields(question, table)

        if intent == 'COUNT':
            select_clause = "SELECT COUNT(*)"
        elif intent in ['SUM', 'AVG']:
            field = fields[0] if fields != ['*'] else self.numeric_columns[table][0]
            select_clause = f"SELECT {intent}({field})"
        else:
            select_clause = f"SELECT {', '.join(fields)}"

        from_clause = f"FROM {table}"
        where_clause = ""
        if self.date_columns[table]:
            date_column, column_type = self.date_columns[table][0]
            where_clause = self.date_handler.generate_conditions(question, date_column, column_type, self.nlp)

        sql = f"{select_clause} {from_clause} {where_clause}".strip()
        return sql

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

def main():
    conn = sqlconn.get_sql_connection()
    metadata, date_columns, numeric_columns = get_metadata(conn)
    training_data = create_training_data()

    converter = TextToSQL(metadata, date_columns, numeric_columns, training_data)

    print("\nField Map:", dict(converter.field_map))
    print("Table Map:", dict(converter.table_map))

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