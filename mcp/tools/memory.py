import sqlite3
import os
import re
import uuid
import logging
from typing import Dict, Any, List, Optional
import httpx
import atexit
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
logger = logging.getLogger("MemoryRAG")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "memories.db")
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant_db")
COLLECTION_NAME = "user_memories"
VECTOR_DIM = 768

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Singleton Qdrant Embedded client
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Returns the local embedded Qdrant database instance."""
    global _qdrant_client
    if _qdrant_client is None:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        if not _qdrant_client.collection_exists(COLLECTION_NAME):
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
            )
            logger.info("Initialized Qdrant collection 'user_memories'.")
    return _qdrant_client


def close_qdrant():
    """Cleanly flushes and closes the Qdrant embedded database on shutdown."""
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass
        _qdrant_client = None


atexit.register(close_qdrant)


def get_embedding(text: str) -> Optional[List[float]]:
    """Generates a 768-dimensional semantic embedding via Google Gemini API."""
    if not GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": VECTOR_DIM
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json().get("embedding", {}).get("values", [])
            logger.warning(f"Gemini embedding API error: HTTP {resp.status_code} - {resp.text[:100]}")
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
    return None


def get_db_connection():
    """Returns a SQLite connection to memories.db."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db():
    """Creates the permanent memories SQLite table and syncs missing points to Qdrant."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(phone_number, key)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_phone ON memories(phone_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)")
        conn.commit()

        # Sync existing SQLite memories into Qdrant if collection is empty
        try:
            q_client = get_qdrant_client()
            col_info = q_client.get_collection(COLLECTION_NAME)
            if col_info.points_count == 0:
                cursor.execute("SELECT phone_number, key, value FROM memories")
                rows = cursor.fetchall()
                points = []
                for r in rows:
                    fact = f"{r['key'].replace('_', ' ').title()}: {r['value']}"
                    vec = get_embedding(fact)
                    if vec:
                        pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{r['phone_number']}:{r['key']}"))
                        points.append(
                            PointStruct(
                                id=pt_id,
                                vector=vec,
                                payload={
                                    "phone_number": r["phone_number"],
                                    "key": r["key"],
                                    "value": r["value"],
                                    "fact": fact
                                }
                            )
                        )
                if points:
                    q_client.upsert(collection_name=COLLECTION_NAME, points=points)
                    logger.info(f"Synced {len(points)} memories from SQLite into Qdrant vector store.")
        except Exception as e:
            logger.warning(f"Could not auto-sync SQLite to Qdrant on startup: {e}")


# Initialize database table and Qdrant on import
init_memory_db()


def save_memory(key: str, value: str, phone_number: str = "default_user") -> str:
    """
    Saves or updates a permanent personal fact in SQLite and indexes its vector in Qdrant.
    """
    key_clean = key.strip().lower()
    value_clean = value.strip()
    
    # 1. Save to SQLite for relational durability
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (phone_number, key, value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(phone_number, key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
        """, (phone_number, key_clean, value_clean))
        conn.commit()

    # 2. Vectorize and save into Qdrant
    try:
        q_client = get_qdrant_client()
        fact_text = f"{key_clean.replace('_', ' ').title()}: {value_clean}"
        vector = get_embedding(fact_text)
        if vector:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{phone_number}:{key_clean}"))
            q_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "phone_number": phone_number,
                            "key": key_clean,
                            "value": value_clean,
                            "fact": fact_text
                        }
                    )
                ]
            )
    except Exception as q_err:
        logger.warning(f"Could not index memory in Qdrant: {q_err}")

    return f"Successfully saved memory: '{key_clean}' = '{value_clean}'"


def search_memory(query: str, phone_number: str = "default_user") -> str:
    """
    Hybrid semantic RAG search:
    1. Runs HNSW vector search in Qdrant using cosine similarity.
    2. Runs SQLite keyword match for exact keys.
    3. Combines and deduplicates results for the AI agent.
    """
    results_map: Dict[str, str] = {}

    # 1. Semantic Vector Search via Qdrant
    try:
        query_vec = get_embedding(query.strip())
        if query_vec:
            q_client = get_qdrant_client()
            search_res = q_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vec,
                limit=5,
                score_threshold=0.50
            )
            for hit in search_res.points:
                k = hit.payload.get("key", "")
                v = hit.payload.get("value", "")
                if k and v:
                    results_map[k] = f"• {k.replace('_', ' ').title()}: {v}"
    except Exception as e:
        logger.warning(f"Qdrant semantic search failed: {e}")

    # 2. Keyword fallback in SQLite for exact matches
    query_clean = f"%{query.strip().lower()}%"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT key, value FROM memories
            WHERE (phone_number = ? OR phone_number = 'default_user') AND (key LIKE ? OR value LIKE ?)
            ORDER BY updated_at DESC
            LIMIT 5
        """, (phone_number, query_clean, query_clean))
        for row in cursor.fetchall():
            k = row["key"]
            if k not in results_map:
                results_map[k] = f"• {k.replace('_', ' ').title()}: {row['value']}"

    if not results_map:
        return f"No memories found matching query '{query}'."

    return "\n".join(results_map.values())


# -------------------------------------------------------------------
# Tool Schemas for the LLM
# -------------------------------------------------------------------

SAVE_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "key": {
            "type": "string",
            "description": "Short topic label, e.g. 'passport', 'car_registration', 'wifi_password', 'doctor_name'"
        },
        "value": {
            "type": "string",
            "description": "The exact fact or detail to remember, e.g. 'Z1234567', 'KL-07-CD-1234', 'Dr. Thomas'"
        }
    },
    "required": ["key", "value"]
}

SEARCH_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Keyword or topic to search for, e.g. 'passport', 'car', 'wifi', 'doctor'"
        }
    },
    "required": ["query"]
}
