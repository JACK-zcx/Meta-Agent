"""
Semantic Memory — conceptual knowledge & entity relations.

Characteristics:
- Stores concept definitions, facts, and entity relationships
- ChromaDB for semantic vector retrieval
- SQLite for entity-relation graph storage (simplified Neo4j replacement)
- Supports structured queries: entities, relations, concepts
- Long-term persistent storage

Storage: SQLite (entity graph) + ChromaDB (semantic vectors)
"""

import os
import re
import json
import uuid
import sqlite3
import threading
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False


class SemanticMemory:
    """Conceptual knowledge store with entity-relation graph."""

    def __init__(self, storage_path: str = "./memory_data",
                 collection_name: str = "semantic"):
        self._storage_path = storage_path
        self._lock = threading.Lock()
        os.makedirs(storage_path, exist_ok=True)

        # ── SQLite for entity-relation graph ──
        self._db_path = os.path.join(storage_path, "semantic.db")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._init_sqlite()

        # ── ChromaDB for concept vectors ──
        self._collection = None
        if _CHROMA_OK:
            try:
                client = chromadb.PersistentClient(
                    path=os.path.join(storage_path, "chroma_semantic"),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                self._collection = None

        self._embed_fn = None

    def _init_sqlite(self):
        # Concepts: definitions, facts, abstract knowledge
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                domain TEXT DEFAULT '',
                importance REAL DEFAULT 0.5,
                metadata_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Entities: named things with attributes
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT '',
                attributes_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Relations: (entity_a) -[relation]-> (entity_b)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                entity_a TEXT NOT NULL,
                relation TEXT NOT NULL,
                entity_b TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_a ON relations(entity_a)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_b ON relations(entity_b)")
        self._conn.commit()

    def set_embed_fn(self, fn):
        self._embed_fn = fn

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn:
            return self._embed_fn(texts)
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

    # ---- Concepts ----

    def add_concept(self, name: str, content: str, domain: str = "",
                    importance: float = 0.5, metadata: dict = None) -> str:
        """Add a concept (fact, definition, knowledge)."""
        cid = str(uuid.uuid4())[:8]
        meta = metadata or {}
        with self._lock:
            self._conn.execute(
                "INSERT INTO concepts (id,name,content,domain,importance,metadata_json) VALUES (?,?,?,?,?,?)",
                (cid, name, content, domain, importance, json.dumps(meta, ensure_ascii=False)),
            )
            self._conn.commit()

            if self._collection and self._embed_fn:
                try:
                    emb = self._embed([content])
                    self._collection.add(
                        ids=[cid], embeddings=emb, documents=[content],
                        metadatas=[{"name": name, "domain": domain, "type": "concept"}],
                    )
                except Exception:
                    pass
        return cid

    def search_concepts(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search over concepts."""
        with self._lock:
            if self._collection and self._embed_fn:
                return self._vector_search_concepts(query, limit)
            return self._keyword_search_concepts(query, limit)

    def _vector_search_concepts(self, query, limit):
        try:
            q_emb = self._embed([query])
            results = self._collection.query(query_embeddings=q_emb, n_results=limit)
            formatted = []
            if results and results.get("ids") and results["ids"][0]:
                for cid in results["ids"][0]:
                    row = self._conn.execute(
                        "SELECT * FROM concepts WHERE id=?", (cid,)).fetchone()
                    if row:
                        formatted.append(self._concept_row(row))
            return formatted
        except Exception:
            return self._keyword_search_concepts(query, limit)

    def _keyword_search_concepts(self, query, limit):
        words = (query or "").lower().split()
        rows = self._conn.execute(
            "SELECT * FROM concepts ORDER BY created_at DESC LIMIT 500").fetchall()
        scored = []
        for row in rows:
            text = (row[2] or "").lower()
            s = sum(1.0 for w in words if w in text) / max(len(words), 1)
            s *= (0.5 + (row[4] or 0.5) * 0.5)
            if s > 0:
                scored.append((row, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._concept_row(r) for r, _ in scored[:limit]]

    def _concept_row(self, row) -> dict:
        return {"id": row[0], "name": row[1], "content": row[2],
                "domain": row[3], "importance": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
                "created_at": row[6]}

    # ---- Entities & Relations ----

    def add_entity(self, name: str, entity_type: str = "",
                   attributes: dict = None) -> str:
        """Add a named entity."""
        eid = str(uuid.uuid4())[:8]
        with self._lock:
            self._conn.execute(
                "INSERT INTO entities (id,name,entity_type,attributes_json) VALUES (?,?,?,?)",
                (eid, name, entity_type, json.dumps(attributes or {}, ensure_ascii=False)),
            )
            self._conn.commit()
        return eid

    def add_relation(self, entity_a: str, relation: str,
                     entity_b: str, weight: float = 1.0) -> str:
        """Add a directed relation between two entities."""
        rid = str(uuid.uuid4())[:8]
        with self._lock:
            self._conn.execute(
                "INSERT INTO relations (id,entity_a,relation,entity_b,weight) VALUES (?,?,?,?,?)",
                (rid, entity_a, relation, entity_b, weight),
            )
            self._conn.commit()
        return rid

    def query_entity(self, name: str) -> list[dict]:
        """Find entity by name (fuzzy)."""
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE name LIKE ?",
            (f"%{name}%",),
        ).fetchall()
        return [{"id": r[0], "name": r[1], "type": r[2],
                 "attributes": json.loads(r[3]) if r[3] else {},
                 "created_at": r[4]} for r in rows]

    def query_relations(self, entity_name: str, direction: str = "both") -> list[dict]:
        """Get all relations for an entity."""
        results = []
        if direction in ("out", "both"):
            rows = self._conn.execute(
                "SELECT * FROM relations WHERE entity_a LIKE ?",
                (f"%{entity_name}%",),
            ).fetchall()
            for r in rows:
                results.append({
                    "id": r[0], "from": r[1], "relation": r[2],
                    "to": r[3], "weight": r[4], "direction": "out",
                })
        if direction in ("in", "both"):
            rows = self._conn.execute(
                "SELECT * FROM relations WHERE entity_b LIKE ?",
                (f"%{entity_name}%",),
            ).fetchall()
            for r in rows:
                results.append({
                    "id": r[0], "from": r[1], "relation": r[2],
                    "to": r[3], "weight": r[4], "direction": "in",
                })
        return results

    def get_graph_neighborhood(self, entity_name: str, depth: int = 1) -> dict:
        """Return entity + its immediate neighbors and relations (simplified graph query)."""
        entities = self.query_entity(entity_name)
        relations = self.query_relations(entity_name)
        neighbors = set()
        for rel in relations:
            neighbors.add(rel["from"])
            neighbors.add(rel["to"])
        neighbors.discard(entity_name)

        neighbor_entities = []
        for n in list(neighbors)[:20]:
            neighbor_entities.extend(self.query_entity(n))

        return {
            "entity": entity_name,
            "matches": entities,
            "relations": relations,
            "neighbors": neighbor_entities,
        }

    # ---- Stats & Maintenance ----

    def forget_concept(self, concept_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM concepts WHERE id=?", (concept_id,))
            self._conn.commit()
            if self._collection:
                try:
                    self._collection.delete(ids=[concept_id])
                except Exception:
                    pass
            return cur.rowcount > 0

    def stats(self) -> dict:
        concepts = self._conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        entities = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relations = self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        return {
            "type": "semantic",
            "concepts": concepts,
            "entities": entities,
            "relations": relations,
            "storage": f"SQLite (graph) + {'ChromaDB' if self._collection else 'keyword'}",
        }
