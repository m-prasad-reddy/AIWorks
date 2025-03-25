import sqlserverconnection as sqlconn
import spacy
from dateutil import parser as date_parser
from collections import defaultdict
import re
import random

nlp = spacy.load("en_core_web_sm")

# Replace these with your actual credentials and database details
USERNAME = "sa"
PASSWORD = 'S0rry!43'
SERVER = "localhost"
DATABASE = "BikeStores"

print("Initializing T2SQLDataGenEngine for multi-schema support...")

def get_schema_metadata(conn):
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT TABLE_SCHEMA, TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE'
    """)
    tables = [(row[0], row[1]) for row in cursor.fetchall()]
    
    metadata = {}
    date_columns = {}
    numeric_columns = {}
    comments = {}
    
    for schema, table in tables:
        full_table = f"{schema}.{table}"
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
        """)
        columns = cursor.fetchall()
        metadata[full_table] = {col[0]: col[1] for col in columns}
        date_columns[full_table] = [(col[0], col[1]) for col in columns if col[1] in ['date', 'datetime', 'smalldatetime']]
        numeric_columns[full_table] = [col[0] for col in columns if col[1] in ['int', 'decimal', 'float', 'money', 'numeric']]
        
        cursor.execute(f"""
            SELECT 
                OBJECT_NAME(ep.major_id) AS TableName, 
                NULL AS ColumnName, 
                CAST(ep.value AS NVARCHAR(512)) AS Comment
            FROM sys.extended_properties ep
            WHERE ep.class = 1 
                AND ep.minor_id = 0 
                AND ep.major_id = OBJECT_ID('{schema}.{table}')
            UNION ALL
            SELECT 
                OBJECT_NAME(ep.major_id) AS TableName, 
                c.name AS ColumnName, 
                CAST(ep.value AS NVARCHAR(512)) AS Comment
            FROM sys.extended_properties ep
            JOIN sys.columns c 
                ON ep.major_id = c.object_id AND ep.minor_id = c.column_id
            WHERE ep.class = 1 
                AND ep.minor_id > 0 
                AND ep.major_id = OBJECT_ID('{schema}.{table}')
        """)
        table_comments = cursor.fetchall()
        comments[full_table] = {row[1] if row[1] else row[0]: row[2] for row in table_comments if row[2] is not None}
    
    return metadata, date_columns, numeric_columns, comments

def generate_training_data(metadata, date_columns, numeric_columns, comments):
    training_data = []
    
    for full_table in metadata:
        schema, table = full_table.split('.')
        table_lower = table.lower()
        schema_lower = schema.lower()
        table_comments = comments.get(full_table, {})
        
        if table_comments.get(table):
            question = f"How many {schema_lower} {table_lower} are there according to {table_comments[table]}?"
        else:
            question = f"How many {schema_lower} {table_lower} are there?"
        sql = f"SELECT COUNT(*) FROM {full_table}"
        training_data.append((question, sql))
        
        if date_columns[full_table]:
            date_col = date_columns[full_table][0][0]
            date_comment = table_comments.get(date_col, f"{date_col}")
            question = f"How many {schema_lower} {table_lower} {date_comment.lower()} in 2023?"
            sql = f"SELECT COUNT(*) FROM {full_table} WHERE YEAR({date_col}) = 2023"
            training_data.append((question, sql))
            
            month = random.choice(['January', 'June', 'December'])
            month_num = {'January': '01', 'June': '06', 'December': '12'}[month]
            question = f"How many {schema_lower} {table_lower} {date_comment.lower()} in {month} 2023?"
            sql = f"SELECT COUNT(*) FROM {full_table} WHERE MONTH({date_col}) = '{month_num}' AND YEAR({date_col}) = 2023"
            training_data.append((question, sql))
        
        if numeric_columns[full_table]:
            num_col = numeric_columns[full_table][0]
            num_comment = table_comments.get(num_col, f"{num_col}")
            question = f"What is the total {num_comment.lower()} of {schema_lower} {table_lower}?"
            sql = f"SELECT SUM({num_col}) FROM {full_table}"
            training_data.append((question, sql))
            question = f"What is the average {num_comment.lower()} of {schema_lower} {table_lower} in 2023?"
            sql = f"SELECT AVG({num_col}) FROM {full_table} WHERE YEAR({date_columns[full_table][0][0]}) = 2023" if date_columns[full_table] else f"SELECT AVG({num_col}) FROM {full_table}"
            training_data.append((question, sql))
        
        cols = list(metadata[full_table].keys())
        if len(cols) >= 2:
            sel_cols = random.sample(cols, min(2, len(cols)))
            col_comments = [table_comments.get(col, col).lower() for col in sel_cols]
            question = f"Show me {schema_lower} {table_lower} {col_comments[0]} and {col_comments[1]}"
            sql = f"SELECT {', '.join(sel_cols)} FROM {full_table}"
            training_data.append((question, sql))
        
        question = f"List all {schema_lower} {table_lower} from 2023"
        sql = f"SELECT * FROM {full_table} WHERE YEAR({date_columns[full_table][0][0]}) = 2023" if date_columns[full_table] else f"SELECT * FROM {full_table}"
        training_data.append((question, sql))
    
    return training_data

def generate_testing_data(training_data):
    testing_data = []
    for question, sql in random.sample(training_data, min(5, len(training_data))):
        if '2023' in question:
            new_question = question.replace('2023', '2024')
            new_sql = sql.replace('2023', '2024')
        else:
            new_question = question.replace('are there', 'exist').replace('Show me', 'Display')
            new_sql = sql
        testing_data.append((new_question, new_sql))
    return testing_data

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
        # First pass: Populate table_map with fully qualified names
        for question, sql in self.training_data:
            question_lower = question.lower()
            tokens = self.preprocess(question)

            if 'how many' in question_lower:
                self.intent_map['how many'] = 'COUNT'
            elif 'total' in question_lower:
                self.intent_map['total'] = 'SUM'
            elif 'average' in question_lower:
                self.intent_map['average'] = 'AVG'
            elif 'show' in question_lower or 'list' in question_lower or 'display' in question_lower:
                self.intent_map['show'] = 'SELECT'
                self.intent_map['list'] = 'SELECT'
                self.intent_map['display'] = 'SELECT'

            table_match = re.search(r'FROM\s+([^\s]+)', sql, re.IGNORECASE)
            if table_match:
                full_table = table_match.group(1)
                schema, table = full_table.split('.')
                for token in tokens:
                    if token in [schema.lower(), table.lower(), table.lower() + 's']:
                        self.table_map[token] = full_table

        # Second pass: Populate field_map after table_map is complete
        for question, sql in self.training_data:
            tokens = self.preprocess(question)
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
            if select_match:
                fields_str = select_match.group(1)
                if fields_str == '*':
                    continue
                fields = [f.strip() for f in fields_str.split(',') if not f.strip().startswith('o.') and not f.strip().startswith('c.')]
                identified_table = self.identify_table(question)
                for token in tokens:
                    if token not in ['how', 'many', 'what', 'is', 'in', 'from', 'did', 'place', 'joined', 'their', 'all', 'get', 'were', 'placed', 'exist', 'according', 'to']:
                        if token.endswith('s') and token[:-1] in self.table_map:
                            continue
                        if token == 'names':
                            self.field_map[token] = ['FirstName', 'LastName']
                        elif token in ['emails', 'email']:
                            self.field_map[token] = ['Email']
                        elif token == 'amount':
                            self.field_map[token] = ['TotalAmount']
                        elif token == 'dates':
                            self.field_map[token] = ['JoinDate']
                        else:
                            for field in fields:
                                if field in self.metadata.get(identified_table, {}):
                                    self.field_map[token].append(field)
        for key in self.field_map:
            self.field_map[key] = list(dict.fromkeys(self.field_map[key]))

    def preprocess(self, text):
        doc = self.nlp(text.lower())
        tokens = [token.text for token in doc if (not token.is_stop or token.text in ['amount', 'total']) and token.is_alpha]
        return tokens

    def get_intent(self, question):
        question_lower = question.lower()
        for phrase in self.intent_map:
            if phrase in question_lower:
                return self.intent_map[phrase]
        return 'SELECT'

    def identify_table(self, question):
        tokens = self.preprocess(question)
        schema_scores = defaultdict(int)
        
        for token in tokens:
            for full_table in self.table_map:
                if '.' in full_table:  # Ensure full_table is schema-qualified
                    schema, table = full_table.split('.')
                    if token == schema.lower():
                        schema_scores[full_table] += 2
                    elif token in [table.lower(), table.lower() + 's']:
                        schema_scores[full_table] += 1
        
        if schema_scores:
            return max(schema_scores, key=schema_scores.get)
        return list(self.metadata.keys())[0]

    def get_fields(self, question, table):
        intent = self.get_intent(question)
        tokens = self.preprocess(question)
        selected_fields = []

        if intent == 'SUM':
            for token in tokens:
                if token in ['amount', 'total']:
                    if 'TotalAmount' in self.metadata[table]:
                        return ['TotalAmount']
            return [self.numeric_columns[table][0]] if self.numeric_columns[table] else ['*']
        elif intent == 'AVG':
            for token in tokens:
                if token in ['amount', 'order']:
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
    conn = sqlconn.get_sqls_connection(USERNAME, PASSWORD, SERVER,DATABASE)
    metadata, date_columns, numeric_columns, comments = get_schema_metadata(conn)
    print("Meta data:",metadata)
    print("Date Columns:",date_columns)
    print("Numeric columns:",numeric_columns)
    training_data = generate_training_data(metadata, date_columns, numeric_columns, comments)
    testing_data = generate_testing_data(training_data)
    
    converter = TextToSQL(metadata, date_columns, numeric_columns, training_data)

    print("\nField Map:", dict(converter.field_map))
    print("Table Map:", dict(converter.table_map))

    print("\n--- Generated Training Data ---")
    for question, sql in training_data:
        print(f"Question: {question}")
        print(f"SQL: {sql}\n")

    print("\n--- Generated Testing Data ---")
    for question, sql in testing_data:
        print(f"Question: {question}")
        print(f"SQL: {sql}\n")

    print("\n--- Testing Text-to-SQL with Generated Testing Data ---")
    for question, _ in testing_data:
        sql = converter.generate_sql(question)
        print(f"Question: {question}")
        print(f"Generated SQL: {sql}")
        results = converter.execute_query(sql, conn)
        print(f"Results: {results}\n")

    conn.close()

if __name__ == "__main__":
    main()