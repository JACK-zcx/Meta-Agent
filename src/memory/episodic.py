"""
Episodic Memory — event sequence storage with vector retrieval.

Characteristics:
- Full timeline recording with session context
- SQLite for structured metadata (timestamp, session, event_type, importance)
- ChromaDB for semantic vector retrieval
- Supports memory chain construction (before/after links)
- Persistent across sessions

Storage: SQLite + ChromaDB hybrid
"""

import os
import re
import json
import uuid
import sqlite3
import threading
from datetime import datetime
from src.memory.base import BaseMemory

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False


class EpisodicMemory(BaseMemory):
    """Event-sequence memory with SQLite metadata + ChromaDB vectors."""

    def __init__(self, storage_path: str = "./memory_data",
                 collection_name: str = "episodic"):
        super().__init__()
        self._storage_path = storage_path
        self._lock = threading.Lock()
        os.makedirs(storage_path, exist_ok=True)

        # ── SQLite for metadata ──
        self._db_path = os.path.join(storage_path, "episodic.db")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._init_sqlite()

        # ── ChromaDB for vectors ──
        self._collection = None
        if _CHROMA_OK:
            try:
                client = chromadb.PersistentClient(
                    path=os.path.join(storage_path, "chroma_episodic"),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                self._collection = None

        # ── Embedding function reference (set externally) ──
        self._embed_fn = None

    def _init_sqlite(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                event_type TEXT DEFAULT '',
                session_id TEXT DEFAULT '',
                metadata_json TEXT DEFAULT '{}',
                prev_id TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_session
            ON episodes(session_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_created
            ON episodes(created_at)
        """)
        self._conn.commit()

    def set_embed_fn(self, fn):
        """Set external embedding function (e.g. from RAG tool)."""
        self._embed_fn = fn

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn:
            return self._embed_fn(texts)
        # Fallback: simple bag-of-bigrams
        return self._keyword_embed(texts)

    def _keyword_embed(self, texts: list[str]) -> list[list[float]]:
        dim = 128
        results = []
        for text in texts:
            vec = [0.0] * dim
            text_lower = re.sub(r'[^a-zA-Z一-鿿\s]', '', (text or "").lower())
            for word in text_lower.split():
                for i in range(len(word) - 1):
                    idx = hash(word[i:i + 2]) % dim
                    vec[idx] += 1.0
            total = (sum(v * v for v in vec)) ** 0.5
            if total > 0:
                vec = [v / total for v in vec]
            results.append(vec)
        return results

    # ---- Public API ----

    def add(self, content: str, importance: float = 0.5,
            metadata: dict = None, event_type: str = "",
            session_id: str = "", prev_id: str = "") -> str:
        meta = metadata or {}
        item_id = str(uuid.uuid4())[:8]

        with self._lock:
            # SQLite insert
            self._conn.execute(
                """INSERT INTO episodes (id, content, importance, event_type,
                   session_id, metadata_json, prev_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (item_id, content, importance, event_type, session_id,
                 json.dumps(meta, ensure_ascii=False), prev_id),
            )
            self._conn.commit()

            # ChromaDB insert
            if self._collection and self._embed_fn:
                try:
                    emb = self._embed([content])
                    self._collection.add(
                        ids=[item_id],
                        embeddings=emb,
                        documents=[content],
                        metadatas=[{"importance": importance,
                                    "event_type": event_type,
                                    "session_id": session_id}],
                    )
                except Exception:
                    pass
        return item_id

    def search(self, query: str, limit: int = 5) -> list[dict]:
        with self._lock:
            if self._collection and self._embed_fn:
                return self._vector_search(query, limit)
            return self._keyword_search(query, limit)

    def _vector_search(self, query: str, limit: int) -> list[dict]:
        try:
            q_emb = self._embed([query])
            results = self._collection.query(
                query_embeddings=q_emb, n_results=limit)
            formatted = []
            if results and results.get("ids") and results["ids"][0]:
                for i, cid in enumerate(results["ids"][0]):
                    row = self._conn.execute(
                        "SELECT * FROM episodes WHERE id=?",
                        (cid,)).fetchone()
                    if row:
                        formatted.append(self._row_to_dict(row))
            return formatted
        except Exception:
            return self._keyword_search(query, limit)

    def _keyword_search(self, query: str, limit: int) -> list[dict]:
        query_lower = (query or "").lower()
        words = query_lower.split()
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY created_at DESC LIMIT 500"
        ).fetchall()

        scored = []
        for row in rows:
            content = (row[1] or "").lower()
            score = 0.0
            for w in words:
                if w in content:
                    score += 1.0 / len(words)
            score *= (0.5 + (row[2] or 0.5) * 0.5)  # importance factor
            if score > 0:
                scored.append((row, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._row_to_dict(row) for row, _ in scored[:limit]]

    def forget(self, item_id: str = None, threshold: float = None) -> int:
        with self._lock:
            removed = 0
            if item_id:
                cur = self._conn.execute(
                    "DELETE FROM episodes WHERE id=?", (item_id,))
                removed = cur.rowcount
            elif threshold is not None:
                cur = self._conn.execute(
                    "DELETE FROM episodes WHERE importance < ?",
                    (threshold,))
                removed = cur.rowcount
            self._conn.commit()

            # ChromaDB delete
            if self._collection and item_id:
                try:
                    self._collection.delete(ids=[item_id])
                except Exception:
                    pass
            return removed

    def get_chain(self, item_id: str) -> list[dict]:
        """Follow prev_id links to build an event chain."""
        chain = []
        seen = set()
        current = item_id
        while current and current not in seen:
            seen.add(current)
            row = self._conn.execute(
                "SELECT * FROM episodes WHERE id=?", (current,)).fetchone()
            if not row:
                break
            chain.append(self._row_to_dict(row))
            current = row[6]  # prev_id
        return chain

    def list_by_session(self, session_id: str, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict:
        row = self._conn.execute("SELECT COUNT(*), AVG(importance) FROM episodes").fetchone()
        return {
            "type": "episodic",
            "count": row[0] or 0,
            "avg_importance": round(row[1], 3) if row[1] else 0,
            "storage": f"SQLite + {'ChromaDB' if self._collection else 'keyword'}",
            "db_path": self._db_path,
        }

    def _row_to_dict(self, row) -> dict:
        return {
            "id": row[0],
            "content": row[1],
            "importance": row[2],
            "event_type": row[3],
            "session_id": row[4],
            "metadata": json.loads(row[5]) if row[5] else {},
            "prev_id": row[6],
            "created_at": row[7],
        }
