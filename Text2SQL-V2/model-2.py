from langchain.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
import sqlserverconnection as sqlconn
import os
import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#   os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


connection_string =sqlconn.get_connection_uri()
db = SQLDatabase.from_uri(connection_string)
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
chain = create_sql_query_chain(llm, db)
query = "How many customers joined in 2023?"
result = chain.invoke({"question": query})
