import pyodbc
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sqlserverconnection as sqlconn

# Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# Database connection
def connect_db():
    # conn = pyodbc.connect(
    #     'DRIVER={SQL Server};'
    #     'SERVER=your_server_name;'
    #     'DATABASE=SalesDB;'
    #     'Trusted_Connection=yes;'
    # )
    conn = sqlconn.get_sql_connection()
    return conn

# Schema definition
SCHEMA = {
    'sales.Customers': ['CustomerID', 'FirstName', 'LastName', 'Email', 'JoinDate'],
    'sales.Orders': ['OrderID', 'CustomerID', 'OrderDate', 'TotalAmount']
}

# Mapping of natural language terms to SQL components
INTENT_MAP = {
    'how many': 'COUNT',
    'what': 'SELECT',
    'who': 'SELECT',
    'show': 'SELECT',
    'list': 'SELECT',
    'total': 'SUM',
    'average': 'AVG'
}

TABLE_MAP = {
    'customer': 'sales.Customers',
    'order': 'sales.Orders',
    'people': 'sales.Customers',
    'sales': 'salaes.Orders'
}

FIELD_MAP = {
    'name': ['First_Name', 'Last_Name'],
    'email': 'Email',
    'date': ['Join_Date', 'Order_Date'],
    'amount': 'Total_Amount',
    'id': ['Customer_ID', 'Order_ID']
}

# Training data for intent classification
TRAINING_DATA = [
    ("how many customers joined in 2023", "SELECT COUNT(*) FROM sales.Customers WHERE Join_Date LIKE '2023%'"),
    ("what is the total amount of orders", "SELECT SUM(Total_Amount) FROM sales.Orders"),
    ("show me customer names", "SELECT First_Name, Last_Name FROM sales.Customers"),
    ("list orders from 2023", "SELECT * FROM sales.Orders WHERE Order_Date LIKE '2023%'"),
]

class TextToSQL:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.train_questions = [q for q, _ in TRAINING_DATA]
        self.train_sql = [sql for _, sql in TRAINING_DATA]
        self.X = self.vectorizer.fit_transform(self.train_questions)

    def preprocess(self, text):
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words and token.isalnum()]

    def get_intent(self, question):
        tokens = self.preprocess(question)
        for word, sql_func in INTENT_MAP.items():
            if word in question.lower():
                return sql_func
        # Use TF-IDF similarity as fallback
        question_vec = self.vectorizer.transform([question])
        similarities = question_vec.dot(self.X.T).toarray()[0]
        best_match_idx = np.argmax(similarities)
        return INTENT_MAP.get(self.train_questions[best_match_idx].split()[0], 'SELECT')

    def get_table(self, question):
        tokens = self.preprocess(question)
        for word, table in TABLE_MAP.items():
            if word in tokens:
                return table
        return 'Customers'  # Default table

    def get_fields(self, question):
        tokens = self.preprocess(question)
        fields = []
        for word, field in FIELD_MAP.items():
            if word in tokens:
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        return fields if fields else ['*']

    def get_conditions(self, question):
        conditions = []
        # Simple date condition detection
        date_pattern = r'\b(20\d{2})\b'
        dates = re.findall(date_pattern, question)
        if dates:
            conditions.append(f"WHERE Order_Date LIKE '{dates[0]}%'")
        return ' '.join(conditions)

    def generate_sql(self, question):
        intent = self.get_intent(question)
        table = self.get_table(question)
        fields = self.get_fields(question)
        
        # Construct base query
        if intent in ['COUNT', 'SUM', 'AVG']:
            select_clause = f"SELECT {intent}({fields[0]})"
        else:
            select_clause = f"SELECT {', '.join(fields)}"
            
        from_clause = f"FROM {table}"
        where_clause = self.get_conditions(question)
        
        return f"{select_clause} {from_clause} {where_clause}".strip()

    def execute_query(self, sql, conn):
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except Exception as e:
            return f"Error: {str(e)}"

# Usage example
def main():
    converter = TextToSQL()
    conn = connect_db()
    
    # Test questions
    test_questions = [
        #"how many customers joined in 2023",
        #"what is the total amount of orders",
        "show me 10 customer names"
        # "list orders from 2023"
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
    
    