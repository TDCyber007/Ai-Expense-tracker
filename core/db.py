"""
core/db.py
SQLite persistence layer for the AI Expense Tracker.

Keeps all SQL in one place so the ML modules and the web layer
can work with plain Python data structures / DataFrames.
"""
import os
import sqlite3
from datetime import datetime

# Database lives under <project>/data/expenses.db by default.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "expenses.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def get_connection():
    """Return a SQLite connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they do not yet exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                description  TEXT    NOT NULL,
                amount       REAL    NOT NULL,
                category     TEXT    NOT NULL,
                date         TEXT    NOT NULL,          -- ISO YYYY-MM-DD
                created_at   TEXT    NOT NULL,          -- ISO datetime
                source       TEXT    NOT NULL DEFAULT 'manual',
                confidence   REAL    NOT NULL DEFAULT 0.0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_expenses_date "
            "ON expenses(date)"
        )
        # Per-category monthly budget limits (optional feature).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS budgets (
                category TEXT PRIMARY KEY,
                amount   REAL NOT NULL
            )
            """
        )


# ---------------------------------------------------------------------------
# Expenses
# ---------------------------------------------------------------------------
def add_expense(description, amount, category, date,
                source="manual", confidence=0.0):
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO expenses
                (description, amount, category, date,
                 created_at, source, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                description.strip(),
                float(amount),
                category,
                date,
                datetime.now().isoformat(timespec="seconds"),
                source,
                float(confidence),
            ),
        )
        return cur.lastrowid


def delete_expense(expense_id):
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM expenses WHERE id = ?", (expense_id,)
        )
        return cur.rowcount


def list_expenses(limit=None, start_date=None, end_date=None):
    sql = "SELECT * FROM expenses"
    clauses, params = [], []
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY date DESC, id DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def count_expenses():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) c FROM expenses").fetchone()["c"]


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------
def set_budget(category, amount):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO budgets(category, amount) VALUES(?, ?) "
            "ON CONFLICT(category) DO UPDATE SET amount = excluded.amount",
            (category, float(amount)),
        )


def get_budgets():
    with get_connection() as conn:
        return {r["category"]: r["amount"]
                for r in conn.execute("SELECT * FROM budgets")}


def clear_all():
    """Remove every transaction and budget (used by the reset endpoint)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM expenses")
        conn.execute("DELETE FROM budgets")


if __name__ == "__main__":
    init_db()
    print(f"Database ready at {DB_PATH}")