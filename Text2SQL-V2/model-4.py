import sqlserverconnection as sqlconn
from vanna.base import VannaBase
from vanna.chromadb import ChromaDB_VectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyodbc

def connect_db():
    conn = sqlconn.get_sql_connection()
    return conn

connection = connect_db()
cursor = connection.cursor()

# Step 1: Custom LLM Class for Hugging Face Model Integration
class HFLLM:
    def __init__(self, model_name="defog/llama-3-sqlcoder-8b"):
        # Load the tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Use GPU if available, otherwise CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
    def get_completion(self, messages, **kwargs):
        # Format messages into a prompt for the model
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant:"
        # Tokenize and generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        #outputs = self.model.generate(**inputs, max_length=512, num_return_sequences=1)
        # Use max_new_tokens instead of max_length to limit output length
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,  # Limit output to 100 tokens (enough for SQL)
            do_sample=False,     # Deterministic output for consistency
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant:")[1].strip()
    

# Step 2: Custom Vanna Class with ChromaDB and SQL Server Integration
class MyVanna(ChromaDB_VectorStore):
    def __init__(self, connection_string):
        # Initialize ChromaDB for local vector storage
        ChromaDB_VectorStore.__init__(self)
        # Set up the custom LLM
        self.llm = HFLLM()
        # Connect to SQL Server
        self.connection = pyodbc.connect(connection_string)
        self.cursor = self.connection.cursor()
        self.schema_prompt = """
        ===Tables
        CREATE TABLE [dbo].[Customers] (
            [CustomerID] [int] NOT NULL,
            [FirstName] [varchar](50) NULL,
            [LastName] [varchar](50) NULL,
            [Email] [varchar](100) NULL,
            [JoinDate] [date] NULL
        )
        CREATE TABLE [dbo].[Orders] (
            [OrderID] [int] NOT NULL,
            [CustomerID] [int] NULL,
            [OrderDate] [date] NULL,
            [TotalAmount] [decimal](10, 2) NULL,
            FOREIGN KEY ([CustomerID]) REFERENCES [dbo].[Customers] ([CustomerID])
        )
        """
        print("Connected to SQL Server")
        
# Implement abstract methods from VannaBase
    def system_message(self, content: str) -> dict:
        return {"role": "system", "content": content}

    def user_message(self, content: str) -> dict:
        return {"role": "user", "content": content}

    def assistant_message(self, content: str) -> dict:
        return {"role": "assistant", "content": content}

    # def submit_prompt(self, prompt: list[dict], **kwargs) -> str:
    #     # Submit the prompt to the LLM and return the response
    #     return self.llm.get_completion(prompt)
    def submit_prompt(self, prompt: list[dict]) -> str:
        # Custom prompt: only include system instruction, schema, and current question
        system_content = "You are a SQL expert for Microsoft SQL Server. Generate a valid SQL query based on the schema below and the user’s question. Use [dbo].[TableName] format and SQL Server syntax (e.g., TOP instead of LIMIT)."
        filtered_prompt = [
            self.system_message(system_content + "\n" + self.schema_prompt),
            self.user_message(prompt[-1]['content'])  # Only the latest user question
        ]
        print("Filtered SQL Prompt:", filtered_prompt)  # Debug output
        return self.llm.get_completion(filtered_prompt)
    
    def run_sql(self, sql: str):
        # Temporary fixes for SQL Server compatibility
        # if "user" in sql.lower() and "[user]" not in sql.lower():
        #     #sql = sql.replace("user", "[user]")  # Escape reserved keyword
        #     sql = sql.replace("user", "[Customers]")  # Redirect 'user' to 'Customers'
        sql = sql.replace("customers", "[dbo].[Customers]").replace("orders", "[dbo].[Orders]")
        if "EXTRACT(YEAR FROM" in sql:
            sql = sql.replace("EXTRACT(YEAR FROM ", "YEAR(").replace(")", ")")
        if "joindate = '2023'" in sql.lower():
            sql = sql.replace("joindate = '2023'", "YEAR([JoinDate]) = 2023")
        if "LIMIT" in sql:
            limit_value = sql.split("LIMIT")[1].strip()
            sql = sql.replace(f"LIMIT {limit_value}", f"TOP {limit_value}")
        # Execute SQL query and return results or error
        try:
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            return f"SQL Error: {str(e)}"

    def close(self):
        # Close database connection when done
        self.connection.close()
        print("Database connection closed")
        
# # Step 3: Build Training Data from SQL Server Schema
# def build_training_data(vn, cursor):
#     # Get all table names from the database
#     cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
#     tables = [row[0] for row in cursor.fetchall()]

#     # Generate DDL for each table
#     for table in tables:
#         cursor.execute(f"""
#             SELECT COLUMN_NAME, DATA_TYPE
#             FROM INFORMATION_SCHEMA.COLUMNS
#             WHERE TABLE_NAME = '{table}'
#         """)
#         columns = cursor.fetchall()
#         # Create a simple DDL statement
#         ddl = f"CREATE TABLE [{table}] ("  # Escape table name
#         ddl += ", ".join([f"{col[0]} {col[1]}" for col in columns])
#         ddl += ")"
#         # Train Vanna with the DDL
#         vn.train(ddl=ddl)
#         print(f"Trained on table: {table}")
        
#     # Add example SQL Server db-specific question-SQL pair (customize as needed)
#     vn.train(
#         question="Find all customers joined in 2023",
#         sql="SELECT * FROM customers WHERE joindate = '2023'"
#     )
#     print("Added example question-SQL pair")
# # end of build_training_data

def build_training_data(vn, cursor):
    # Use your specific DDL for Customers and Orders
    customers_ddl = """
    CREATE TABLE [dbo].[Customers] (
        [CustomerID] [int] NOT NULL,
        [FirstName] [varchar](50) NULL,
        [LastName] [varchar](50) NULL,
        [Email] [varchar](100) NULL,
        [JoinDate] [date] NULL
    )
    """
    orders_ddl = """
    CREATE TABLE [dbo].[Orders] (
        [OrderID] [int] NOT NULL,
        [CustomerID] [int] NULL,
        [OrderDate] [date] NULL,
        [TotalAmount] [decimal](10, 2) NULL,
        FOREIGN KEY ([CustomerID]) REFERENCES [dbo].[Customers] ([CustomerID])
    )
    """
    
    # Train Vanna with the DDL
    vn.train(ddl=customers_ddl)
    vn.train(ddl=orders_ddl)
    print("Trained on Customers and Orders tables")

    # Add example questions specific to your schema
    vn.train(
        question="Show all customers",
        sql="SELECT * FROM [dbo].[Customers]"
    )
    vn.train(
        question="Find orders after January 1, 2023",
        sql="SELECT * FROM [dbo].[Orders] WHERE [OrderDate] > '2023-01-01'"
    )
    vn.train(
        question="Get the top 5 most expensive orders",
        sql="SELECT TOP 5 * FROM [dbo].[Orders] ORDER BY [TotalAmount] DESC"
    )
    vn.train(
        question="List customers who joined after 2020",
        sql="SELECT * FROM [dbo].[Customers] WHERE [JoinDate] > '2020-01-01'"
    )
    vn.train(question="Show all customers who joined in 2023", 
             sql="SELECT * FROM [dbo].[Customers] WHERE YEAR([JoinDate]) = 2023")
    vn.train(question="List customers with first and last names who joined in 2022", sql="SELECT [FirstName], [LastName] FROM [dbo].[Customers] WHERE YEAR([JoinDate]) = 2022")
    print("Added SQL Server-specific example question-SQL pairs")
    
# Step 4: Application Implementation
def main():
    SERVER = 'localhost'
    DATABASE = 'salesdb'
    USERNAME = 'sa'
    PASSWORD = 'S0rry!43'
    # Get SQL Server connection string (replace with your details or use env variable)
    #connection_string = os.getenv('SQL_CONNECTION_STRING')
    #if not connection_string:
        #connection_string = input("Enter your SQL Server connection string (e.g., 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=your_db;UID=your_user;PWD=your_pass'): ")
    connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}'
        

    # Initialize Vanna with ChromaDB and SQL Server
    vn = MyVanna(connection_string)

    # Build training data from the database schema
    build_training_data(vn, vn.cursor)

    # Command-line interface for user queries
    print("\nWelcome to the Text-to-SQL Application!")
    print("Type your question or 'exit' to quit.")
    while True:
        question = input("Your question: ")
        if question.lower() == 'exit':
            break
        try:
            # Generate SQL from the natural language question
            sql = vn.generate_sql(question)
            print("Generated SQL:", sql)

            # Execute the SQL and display results
            results = vn.run_sql(sql)
            if isinstance(results, str):
                print("Error:", results)
            else:
                print("Results:")
                for row in results:
                    print(row)
        except Exception as e:
            print("An error occurred:", str(e))

    # Clean up
    vn.close()

# Run the application
if __name__ == "__main__":
    main()    