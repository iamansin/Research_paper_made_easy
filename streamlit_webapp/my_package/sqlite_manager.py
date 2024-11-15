import sqlite3
import os 


class SQLiteManager:
    def __init__(self, db_name="pdf_storage.db"):
        self.db_name = db_name
        self.conn = None
    def create_database_directory(self):
        # Create the database directory if it doesn't exist
        db_directory = os.path.dirname(self.db_name)
        if not os.path.exists(db_directory):
            os.makedirs(db_directory)
            
    def get_connection(self):
        try:
            self.create_database_directory
            print(f"Connecting to database at: {self.db_name}") 
            if self.conn is None or not self.conn:
                self.conn = sqlite3.connect(self.db_name)
                self.create_table()
                return self.conn
        except Exception as e:
            print(e)
        

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pdf_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mongo_id TEXT,
                filename TEXT
            )
        ''')
        self.conn.commit()

    def close_connection(self):
        if self.conn:
            self.conn.close()
            self.conn = None

def insert_pdf_to_sqlite(db, mongo_id, filename):
    conn = db.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO pdf_files (mongo_id, filename) VALUES (?, ?)
        ''', (mongo_id, filename))
        conn.commit()
    except Exception as e:
        print(f"problem in creating table {e}")
        
        
