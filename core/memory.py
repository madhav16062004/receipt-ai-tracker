"""
memory.py — Dual Database Layer (SQLite + ChromaDB)
Handles all data persistence for structured receipts and vector embeddings.
"""

import sqlite3
import pandas as pd
import chromadb
import ollama
import os
import uuid
from datetime import datetime

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SQLITE_PATH = os.path.join(DATA_DIR, "expenses.db")
CHROMA_PATH = os.path.join(DATA_DIR, "chromadb")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)


# ──────────────────────────────────────────────
# SQLite Functions
# ──────────────────────────────────────────────

def init_dbs():
    """Initialize both SQLite and ChromaDB on app startup."""
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS receipts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        merchant TEXT,
        total REAL,
        currency TEXT,
        category TEXT
    )''')
    conn.commit()
    conn.close()

    # Touch ChromaDB so the collection exists
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    client.get_or_create_collection(name="receipts")


def reset_dbs():
    """Reset both SQLite and ChromaDB by deleting all data."""
    # Reset SQLite
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS receipts")
    conn.commit()
    conn.close()
    
    # Reset ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(name="receipts")
    except ValueError:
        pass # Collection might not exist yet
    
    # Reinitialize empty tables
    init_dbs()


def save_receipt_to_sql(data: dict):
    """Insert a single structured receipt row into SQLite."""
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO receipts (date, merchant, total, currency, category) VALUES (?, ?, ?, ?, ?)",
        (
            data.get("date"),
            data.get("merchant"),
            data.get("total"),
            data.get("currency"),
            data.get("category"),
        ),
    )
    conn.commit()
    conn.close()


def get_all_receipts() -> pd.DataFrame:
    """Return every receipt as a DataFrame (most recent first)."""
    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query("SELECT * FROM receipts ORDER BY id DESC", conn)
    conn.close()
    return df


# Hardcoded exchange rates to INR
CURRENCY_TO_INR = {
    "$": 85.50,
    "USD": 85.50,
    "€": 93.00,
    "EUR": 93.00,
    "£": 108.00,
    "GBP": 108.00,
    "₹": 1.0,
    "INR": 1.0,
    "Rp": 0.0053,
    "IDR": 0.0053,
    "IDR": 0.0053,
}


def convert_to_inr(amount: float, currency: str) -> float:
    """Convert an amount to INR using hardcoded exchange rates."""
    rate = CURRENCY_TO_INR.get(currency.strip(), 1.0)
    return round(amount * rate, 2)


def get_current_month_total() -> float:
    """Quick sidebar metric: total spend for the current calendar month, in INR."""
    now = datetime.now()
    month_prefix = now.strftime("%Y-%m")  # e.g. "2026-04"
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT COALESCE(total, 0), COALESCE(currency, 'INR') FROM receipts WHERE date LIKE ?",
        (f"{month_prefix}%",),
    )
    rows = c.fetchall()
    conn.close()
    return sum(convert_to_inr(row[0], row[1]) for row in rows)


def get_total_spend() -> float:
    """All-time total spend across all receipts, converted to INR."""
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute("SELECT COALESCE(total, 0), COALESCE(currency, 'INR') FROM receipts")
    rows = c.fetchall()
    conn.close()
    return sum(convert_to_inr(row[0], row[1]) for row in rows)


def execute_sql(query: str) -> str:
    """Execute an arbitrary SELECT query and return the result as a formatted string.
    Used by the agent's SQL tool. Only SELECT statements are allowed.
    """
    try:
        # Basic safety check — only allow SELECT
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."
        conn = sqlite3.connect(SQLITE_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            return "Query returned no results."
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {e}"


# ──────────────────────────────────────────────
# ChromaDB Functions
# ──────────────────────────────────────────────

def _get_chroma_collection():
    """Return the ChromaDB receipts collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(name="receipts")


def save_receipt_to_vector(raw_text: str, structured_data: dict):
    """Embed the raw OCR text with nomic-embed-text and store it in ChromaDB,
    with the structured fields attached as metadata for filtering.
    """
    collection = _get_chroma_collection()
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=raw_text)
        embedding = response["embedding"]

        metadata = {
            "merchant": str(structured_data.get("merchant", "Unknown")),
            "date": str(structured_data.get("date", "Unknown")),
            "total": float(structured_data.get("total", 0.0)),
            "currency": str(structured_data.get("currency", "")),
            "category": str(structured_data.get("category", "Other")),
        }

        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[raw_text],
            metadatas=[metadata],
        )
    except Exception as e:
        print(f"[memory] Error saving to ChromaDB: {e}")


def save_receipt_data(raw_text: str, structured_data: dict):
    """Convenience wrapper — saves to BOTH databases in one call."""
    save_receipt_to_sql(structured_data)
    save_receipt_to_vector(raw_text, structured_data)


def semantic_search(query: str, n_results: int = 3) -> str:
    """Search ChromaDB using a natural-language query.
    Returns a formatted string of matching receipts + metadata.
    """
    collection = _get_chroma_collection()
    try:
        if collection.count() == 0:
            return "No receipts stored yet."

        response = ollama.embeddings(model="nomic-embed-text", prompt=query)
        embedding = response["embedding"]

        results = collection.query(query_embeddings=[embedding], n_results=n_results)

        if not results["documents"] or len(results["documents"][0]) == 0:
            return "No matching receipts found."

        output = ""
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            output += (
                f"--- Receipt {i + 1} ---\n"
                f"Raw Text: {doc}\n"
                f"Merchant: {meta.get('merchant')} | "
                f"Total: {meta.get('total')} | "
                f"Date: {meta.get('date')} | "
                f"Category: {meta.get('category')}\n\n"
            )
        return output

    except Exception as e:
        return f"Vector search error: {e}"
