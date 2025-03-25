import sqlite3 

#Connect to sqllite database
connection = sqlite3.connect('student.db')

# Create a cursor object to insert record, create table, retrieve   records
cursor = connection.cursor()
#create table
table_info = """
            CREATE TABLE student (
              NAME VARCHAR(20), 
              CLASS VARCHAR(25), 
              SECTION VARCHAR(25), 
              MARKS REAL
              )
              """
cursor.execute(table_info)
#Insert records
cursor.execute('''INSERT INTO student VALUES("Krish", "Data Science", "A", 90.5)''')
cursor.execute('''INSERT INTO student VALUES("Sudhanshu", "Data Science", "B", 100.0)''')
cursor.execute('''INSERT INTO student VALUES("Darius", "Data Science", "A", 86)''')
cursor.execute('''INSERT INTO student VALUES("Vikash", "DEVOPS", "A", 50)''')
cursor.execute('''INSERT INTO student VALUES("Dipesh", "DEVOPS", "A", 35)''')
print(" Inserted records successfully")
#Retrieve records
cursor.execute('SELECT * FROM student')
print(cursor.fetchall())
connection.commit()

