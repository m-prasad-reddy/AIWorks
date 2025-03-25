import spacy
import pyodbc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sqlserverconnection as sqlconn
nlp = spacy.load("en_core_web_sm")

def connect_db():
    # conn = pyodbc.connect(
    #     'DRIVER={SQL Server};'
    #     'SERVER=your_server_name;'
    #     'DATABASE=SalesDB;'
    #     'Trusted_Connection=yes;'
    # )
    conn = sqlconn.get_sql_connection()
    return conn

SCHEMA = {
    'Customers': ['CustomerID', 'FirstName', 'LastName', 'Email', 'JoinDate'],
    'Orders': ['OrderID', 'CustomerID', 'OrderDate', 'TotalAmount']
}

INTENT_MAP = {
    'how many': 'COUNT',
    'what': 'SELECT',
    'show': 'SELECT',
    'total': 'SUM',
    'average': 'AVG'
}

TABLE_MAP = {
    'customer': 'Customers',
    'order': 'Orders'
}

FIELD_MAP = {
    'name': ['FirstName', 'LastName'],
    'email': 'Email',
    'date': ['JoinDate', 'OrderDate'],
    'amount': 'TotalAmount'
}

class AdvancedTextToSQL:
    def __init__(self):
        self.nlp = nlp
        #self.stop_words = set(stopwords.words('english'))
        # Use spaCy's built-in stop words
        self.stop_words = self.nlp.Defaults.stop_words

    def preprocess(self, text):
        # tokens = word_tokenize(text.lower())
        # return [token for token in tokens if token not in self.stop_words and token.isalnum()]
        # Use spaCy for tokenization and preprocessing
        doc = self.nlp(text.lower())
        tokens = [token.text for token in doc if token.text not in self.stop_words and token.is_alpha]
        return tokens
    

    # def get_intent(self, question):
    #     tokens = self.preprocess(question)
    #     for word, sql_func in INTENT_MAP.items():
    #         if word in question.lower():
    #             return sql_func
    #     return 'SELECT'

    def get_intent(self, question):
        tokens = self.preprocess(question)
        question_lower = question.lower()
        for phrase, sql_func in INTENT_MAP.items():
            if phrase in question_lower:
                return sql_func
        return 'SELECT'  # Default intent
    
    def get_table(self, question):
        tokens = self.preprocess(question)
        for word, table in TABLE_MAP.items():
            if word in tokens:
                return table
        return 'Customers'

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
        if "2023" in question:
            conditions.append("WHERE JoinDate LIKE '2023%'")
        return ' '.join(conditions)

    def generate_sql(self, question):
        intent = self.get_intent(question)
        table = self.get_table(question)
        fields = self.get_fields(question)
        
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

def main():
    converter = AdvancedTextToSQL()
    conn = connect_db()
    
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
    