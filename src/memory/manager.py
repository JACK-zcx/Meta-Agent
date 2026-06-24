"""
MemoryManager — unified orchestrator for all memory types.

Routing logic:
- Short-lived temporary info → WorkingMemory
- Event sequences, conversations, session logs → EpisodicMemory
- Facts, concepts, entity relationships → SemanticMemory

Provides a single interface for the agent to interact with all memory types.
"""

import os
from datetime import datetime
from src.memory.working import WorkingMemory
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory


class MemoryManager:
    """Unified memory orchestrator."""

    def __init__(self, storage_path: str = "./memory_data",
                 working_capacity: int = 50, working_ttl_minutes: int = 60,
                 session_id: str = ""):
        self.storage_path = storage_path
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(storage_path, exist_ok=True)

        self.working = WorkingMemory(
            capacity=working_capacity, ttl_minutes=working_ttl_minutes)
        self.episodic = EpisodicMemory(storage_path=storage_path)
        self.semantic = SemanticMemory(storage_path=storage_path)

        self._embed_fn = None

    def set_embed_fn(self, fn):
        """Share embedding function across all vector-enabled memories."""
        self._embed_fn = fn
        self.episodic.set_embed_fn(fn)
        self.semantic.set_embed_fn(fn)

    # ---- Smart Add (auto-route by memory type hint) ----

    def add(self, content: str, memory_type: str = "working",
            importance: float = 0.5, metadata: dict = None,
            event_type: str = "", **kwargs) -> str:
        """
        Add to the specified memory store.

        memory_type: 'working' | 'episodic' | 'semantic' | 'auto'
        """
        if memory_type == "working":
            return self.working.add(content, importance, metadata)
        elif memory_type == "episodic":
            return self.episodic.add(content, importance, metadata,
                                     event_type=event_type,
                                     session_id=self.session_id, **kwargs)
        elif memory_type == "semantic":
            name = (metadata or {}).get("name", content[:50])
            domain = (metadata or {}).get("domain", "")
            return self.semantic.add_concept(
                name=name, content=content, domain=domain,
                importance=importance, metadata=metadata)
        elif memory_type == "auto":
            # Auto-routing heuristic
            if importance < 0.3 and len(content) < 200:
                return self.working.add(content, importance, metadata)
            elif event_type or (metadata or {}).get("event_type"):
                return self.episodic.add(content, importance, metadata,
                                         event_type=event_type,
                                         session_id=self.session_id)
            else:
                return self.semantic.add_concept(
                    name=content[:50], content=content,
                    importance=importance, metadata=metadata)
        else:
            return self.working.add(content, importance, metadata)

    # ---- Unified Search ----

    def search(self, query: str, memory_types: list[str] = None,
               limit: int = 5) -> dict:
        """
        Search across specified memory types.

        Returns: {"working": [...], "episodic": [...], "semantic": [...]}
        """
        types = memory_types or ["working", "episodic", "semantic"]
        results = {}
        if "working" in types:
            results["working"] = self.working.search(query, limit)
        if "episodic" in types:
            results["episodic"] = self.episodic.search(query, limit)
        if "semantic" in types:
            raw = self.semantic.search_concepts(query, limit)
            results["semantic"] = [
                {"id": r["id"], "content": r["content"],
                 "name": r.get("name", ""), "score": r.get("score", 0),
                 "domain": r.get("domain", "")}
                for r in raw
            ]
        return results

    # ---- Session Management ----

    def get_session_episodes(self, session_id: str = None,
                             limit: int = 50) -> list[dict]:
        """Get all episodic memories for a session."""
        sid = session_id or self.session_id
        return self.episodic.list_by_session(sid, limit)

    # ---- Consolidation (promote important working → episodic) ----

    def consolidate(self, importance_threshold: float = 0.7) -> int:
        """
        Move high-importance working memories into episodic storage.
        Returns number of items consolidated.
        """
        results = self.working.search("", limit=self.working._capacity)
        count = 0
        for item in results:
            if item["importance"] >= importance_threshold:
                self.episodic.add(
                    content=item["content"],
                    importance=item["importance"],
                    event_type="consolidated",
                    session_id=self.session_id,
                )
                self.working.forget(item_id=item["id"])
                count += 1
        return count

    # ---- Stats ----

    def stats(self) -> dict:
        return {
            "working": self.working.stats(),
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
            "session_id": self.session_id,
        }

    # ---- Knowledge graph (semantic memory) shortcuts ----

    def add_knowledge(self, concept_name: str, definition: str,
                      domain: str = "", importance: float = 0.7) -> str:
        """Add a concept/definition to semantic memory."""
        return self.semantic.add_concept(
            name=concept_name, content=definition, domain=domain,
            importance=importance)

    def add_entity_relation(self, entity_a: str, relation: str,
                            entity_b: str) -> str:
        """Add an entity and a relation (auto-creates entities)."""
        self.semantic.add_entity(entity_a)
        self.semantic.add_entity(entity_b)
        return self.semantic.add_relation(entity_a, relation, entity_b)

    def query_knowledge(self, entity_name: str) -> dict:
        """Get an entity's full neighborhood in the knowledge graph."""
        return self.semantic.get_graph_neighborhood(entity_name)

    def search_knowledge(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search over concepts."""
        raw = self.semantic.search_concepts(query, limit)
        return [{"id": r["id"], "name": r.get("name", ""),
                 "content": r["content"], "domain": r.get("domain", "")}
                for r in raw]

    # ---- Forget ----

    def forget(self, memory_type: str, item_id: str = None,
               threshold: float = None) -> int:
        if memory_type == "working":
            return self.working.forget(item_id, threshold)
        elif memory_type == "episodic":
            return self.episodic.forget(item_id, threshold)
        return 0
