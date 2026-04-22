import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS receipts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  merchant TEXT,
                  total REAL,
                  currency TEXT,
                  category TEXT)''')
    conn.commit()
    conn.close()

def save_receipt(data):
    conn = sqlite3.connect('expenses.db')
    c = conn.cursor()
    c.execute("INSERT INTO receipts (date, merchant, total, currency, category) VALUES (?, ?, ?, ?, ?)",
                (data.get('date'), data.get('merchant'), data.get('total'), data.get('currency'), data.get('category')))
    conn.commit()
    conn.close()

def get_all_receipts():
    conn = sqlite3.connect('expenses.db')
    df = pd.read_sql_query("SELECT * FROM receipts ORDER BY id DESC", conn)
    conn.close()
    return df