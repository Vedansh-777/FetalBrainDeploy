import sqlite3
import os

def create_table():
    # Get the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Connect to SQLite database (creates a new database file if not exists)
    with sqlite3.connect(os.path.join(script_dir, 'fetalBrain.db')) as conn:
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()

        # Create a table for user data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tbl_fetal_main (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                upload_image BLOB,
                annotated_image BLOB
            )
        ''')

        # Commit the changes
        conn.commit()

def execute_query(query, *params):
    try:
        # Get the absolute path to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Connect to SQLite database (creates a new database file if not exists)
        with sqlite3.connect(os.path.join(script_dir, 'fetalBrain.db')) as conn:
            # Create a cursor object to execute SQL queries
            cursor = conn.cursor()

            # Execute the query with optional parameters
            cursor.execute(query, params)

            # Commit the changes for insert, update, delete operations
            if query.strip().lower().startswith(('insert', 'update', 'delete')):
                conn.commit()

            # Fetch results for select operations
            if query.strip().lower().startswith('select'):
                results = cursor.fetchall()
                return results

    except sqlite3.Error as e:
        print("SQLite error:", e)
        return None

    finally:
        # Close the cursor (the 'with' statement already closed the connection)
        cursor.close()

# Create the table
create_table()

# Insert data
insert_query = "INSERT INTO tbl_fetal_main (name, username, email, password) VALUES (?, ?, ?, ?)"
execute_query(insert_query, 'Vedansh', 'ved_07', 'vedansh.svp@gmail.com', '123456')

# Select data
select_query = "SELECT * FROM tbl_fetal_main"
select_results = execute_query(select_query)
print(select_results)



