import sqlite3
import os
from typing import List, Dict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions.db")


def get_db_connection():
    """Returns a SQLite connection to sessions.db."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the messages table and indices in SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phone ON messages(phone_number)")
        conn.commit()


def save_message(phone_number: str, role: str, content: str):
    """Saves a message (user or assistant) into the database."""
    if not phone_number or not content:
        return
    # Guardrail: Never persist internal error traces or raw tool tags into chat history
    if role == "assistant" and ("Error code:" in content or "<tool_call>" in content or "tool_use_failed" in content):
        return
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (phone_number, role, content) VALUES (?, ?, ?)",
            (phone_number, role, content.strip())
        )
        conn.commit()


def get_history(phone_number: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    Retrieves the last `limit` messages for a user in chronological order.
    Filters out any past error traces or malformed tags.
    Returns: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    """
    if not phone_number:
        return []
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Query the most recent `limit` messages
        cursor.execute(
            """
            SELECT role, content FROM messages 
            WHERE phone_number = ? 
            ORDER BY id DESC 
            LIMIT ?
            """,
            (phone_number, limit * 2)  # fetch extra to account for filtering
        )
        rows = cursor.fetchall()
        
    filtered = []
    for row in rows:
        c = row["content"]
        if "Error code:" in c or "<tool_call>" in c or "tool_use_failed" in c:
            continue
        filtered.append({"role": row["role"], "content": c})
        if len(filtered) >= limit:
            break

    # Reverse to return them in chronological order (oldest to newest)
    return list(reversed(filtered))


def clear_history(phone_number: str) -> bool:
    """Deletes all conversation history for a given phone number."""
    if not phone_number:
        return False
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE phone_number = ?", (phone_number,))
        conn.commit()
    print(f"🧹 Cleared chat history for phone: {phone_number}", flush=True)
    return True

# Auto-initialize database on import
init_db()
