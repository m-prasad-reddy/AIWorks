from dotenv import load_dotenv
load_dotenv() ## Load the .env file, load all the environment variables

import streamlit as st
import os
import sqlite3

import google.generativeai as genai

# configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Function to load Google Gemini model and provide sql query as response
def get_gemini_response(question,prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt[0],question])
    return response.text

#Function to retrieve query from the sql database
def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cursor=conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows


## Define Your Prompt 
prompt = [
"""
You are an expert in converting English questions to SQL query! 
The SQL database has the name STUDENT and has the following columns - NAME, CLASS, 
SECTION \n\nFor example,\nExamp1e 1 - How many entries of records are present?, 
the SQL command will be something like this SELECT COUNT(*) FROM STUDENT ; 
\nExamp1e 2 - Tell me all the students studying in Data Science class?,  
the SQL command will be something like this SELECT * FROM STUDENT 
where CLASS="Data Science" ; 
also the sql code should not have ``` in beginning or end and sql word in  the output
"""
]


## Streamlit App 

#st.title("SQL Query Generator")
st.set_page_config(page_title="SQL Query Generator")
st.header("Know Your Data")
question=st.text_input("Input: ",key="input")
submit = st.button("Ask the question")

# If submit is clicked

if submit:
    print(question)
    response = get_gemini_response(question,prompt)
    print(response)
    data=read_sql_query(response,"student.db")
    st.subheader("The Response is")
    for row in data:
        print(row)
        st.header(row)