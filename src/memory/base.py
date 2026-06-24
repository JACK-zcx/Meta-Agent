"""Base classes for the memory system."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class MemoryItem:
    """A single memory entry."""

    def __init__(self, content: str, memory_type: str = "working",
                 importance: float = 0.5, metadata: dict = None):
        self.content = content
        self.memory_type = memory_type
        self.importance = max(0.0, min(1.0, importance))
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.metadata = metadata or {}

    def touch(self):
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


class BaseMemory(ABC):
    """Abstract base for all memory types."""

    def __init__(self):
        self._items: list[MemoryItem] = []

    @abstractmethod
    def add(self, content: str, importance: float = 0.5,
            metadata: dict = None) -> str:
        """Add a memory item. Returns item ID."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search memories. Returns list of result dicts."""
        pass

    @abstractmethod
    def forget(self, item_id: str = None, threshold: float = None) -> int:
        """Remove memories. Returns count removed."""
        pass

    @abstractmethod
    def stats(self) -> dict:
        """Return memory statistics."""
        pass
