from typing import Callable, Any
from datetime import datetime
import json
import threading

EventCallback = Callable[[dict], None]


class EventEmitter:
    """Thread-safe event emitter for agent execution events."""

    def __init__(self):
        self._callbacks: list[EventCallback] = []
        self._lock = threading.Lock()

    def on_event(self, callback: EventCallback):
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: EventCallback):
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def emit(self, event_type: str, **data):
        event = {"type": event_type, "timestamp": datetime.now().isoformat(), **data}
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def clear(self):
        with self._lock:
            self._callbacks.clear()


# Module-level emitter instance
emitter = EventEmitter()
