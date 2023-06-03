# Import
import sqlite3

connection = sqlite3.connect("db")

cursor = connection.cursor()
cursor.execute("""
        CREATE TABLE IF NOT EXISTS question_response (
            question text,
            response text,
            timestamp text DEFAULT CURRENT_TIMESTAMP
        )
        """)
connection.commit()

cursor_object = connection.execute("SELECT * FROM question_response")
print(cursor_object.fetchall())

connection.close()
