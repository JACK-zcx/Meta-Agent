"""
Working Memory — pure in-memory storage with TTL management.

Characteristics (modeled after human working memory):
- Limited capacity (default 50 items)
- TTL expiry (default 60 minutes)
- Automatic cleanup of expired / low-importance items
- Fastest access (no disk I/O)
- Session-scoped (lost on restart)
- Time decay scoring for retrieval ranking
"""

import re
import time
import uuid
import threading
from datetime import datetime, timedelta
from src.memory.base import BaseMemory, MemoryItem


class WorkingMemory(BaseMemory):
    """Pure in-memory store with TTL-based expiration."""

    def __init__(self, capacity: int = 50, ttl_minutes: int = 60):
        super().__init__()
        self._capacity = capacity
        self._ttl_minutes = ttl_minutes
        self._lock = threading.Lock()
        self._items: dict[str, MemoryItem] = {}  # id → item

    # ---- Public API ----

    def add(self, content: str, importance: float = 0.5,
            metadata: dict = None) -> str:
        with self._lock:
            self._cleanup_expired()
            self._enforce_capacity()

            item_id = str(uuid.uuid4())[:8]
            item = MemoryItem(content, "working", importance, metadata)
            self._items[item_id] = item
            return item_id

    def search(self, query: str, limit: int = 5) -> list[dict]:
        with self._lock:
            self._cleanup_expired()
            scored = []
            for item_id, item in self._items.items():
                score = self._score(item, query)
                if score > 0:
                    scored.append((item_id, item, score))
            scored.sort(key=lambda x: x[2], reverse=True)
            results = []
            for item_id, item, score in scored[:limit]:
                item.touch()
                results.append({
                    "id": item_id,
                    "content": item.content,
                    "score": round(score, 4),
                    "importance": item.importance,
                    "created_at": item.created_at.isoformat(),
                })
            return results

    def forget(self, item_id: str = None, threshold: float = None) -> int:
        with self._lock:
            removed = 0
            if item_id and item_id in self._items:
                del self._items[item_id]
                removed = 1
            elif threshold is not None:
                to_remove = [
                    k for k, v in self._items.items()
                    if v.importance < threshold
                ]
                for k in to_remove:
                    del self._items[k]
                removed = len(to_remove)
            return removed

    def stats(self) -> dict:
        with self._lock:
            self._cleanup_expired()
            return {
                "type": "working",
                "count": len(self._items),
                "capacity": self._capacity,
                "ttl_minutes": self._ttl_minutes,
                "storage": "memory",
            }

    # ---- Internal ----

    def _score(self, item: MemoryItem, query: str) -> float:
        """Hybrid scoring: keyword match + importance + time decay."""
        content_lower = (item.content or "").lower()
        query_lower = (query or "").lower()

        # ── 1. Keyword / overlap score ──
        keyword_score = 0.0
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        if query_words:
            overlap = len(query_words & content_words)
            keyword_score = overlap / len(query_words)

        # ── 2. Substring bonus ──
        if query_lower in content_lower:
            keyword_score = max(keyword_score, 0.6)

        if keyword_score == 0:
            # Check individual words as substrings
            matches = sum(1 for w in query_words if w in content_lower)
            if matches > 0:
                keyword_score = 0.2 * (matches / len(query_words))

        # ── 3. Importance multiplier ──
        importance_factor = 0.5 + item.importance * 0.5  # 0.5–1.0

        # ── 4. Time decay (half-life = ttl/2) ──
        age_minutes = (datetime.now() - item.created_at).total_seconds() / 60.0
        half_life = self._ttl_minutes / 2.0
        time_factor = 2.0 ** (-age_minutes / half_life)

        return round(keyword_score * importance_factor * time_factor, 4)

    def _cleanup_expired(self):
        cutoff = datetime.now() - timedelta(minutes=self._ttl_minutes)
        expired = [k for k, v in self._items.items() if v.created_at < cutoff]
        for k in expired:
            del self._items[k]

    def _enforce_capacity(self):
        if len(self._items) < self._capacity:
            return
        # Remove lowest-importance items
        sorted_items = sorted(
            self._items.items(), key=lambda x: x[1].importance)
        to_remove = sorted_items[:max(1, len(sorted_items) - self._capacity + 1)]
        for k, _ in to_remove:
            del self._items[k]
