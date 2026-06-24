"""Lightweight tracing system for multi-agent observability.

Records LLM calls (latency, token estimate, input/output) and agent
node executions. Stores traces as JSON files for later analysis.
"""

from datetime import datetime, timezone
from typing import Any
import threading
import json
import uuid
import time
import os


class TraceCollector:
    """Thread-safe collector of agent execution traces."""

    def __init__(self, storage_dir: str = "traces"):
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        self._storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._enabled = True

    # ---- Public API ----

    def new_session(self, query: str) -> str:
        """Start a new trace session. Returns session_id."""
        sid = str(uuid.uuid4())[:8]
        session = {
            "session_id": sid,
            "query": query,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "llm_calls": [],
            "agent_steps": [],
        }
        with self._lock:
            self._sessions[sid] = session
        return sid

    def finish_session(self, session_id: str, output: str):
        """Mark a session as complete and persist to disk."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return
            session["status"] = "completed"
            session["output"] = output
            session["finished_at"] = datetime.now(timezone.utc).isoformat()
            total_ms = 0
            for c in session["llm_calls"]:
                total_ms += c.get("latency_ms", 0)
            session["total_llm_latency_ms"] = total_ms
            session["total_llm_calls"] = len(session["llm_calls"])
            session["total_agent_steps"] = len(session["agent_steps"])
        self._persist(session_id)

    def fail_session(self, session_id: str, error: str):
        """Mark a session as failed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return
            session["status"] = "failed"
            session["error"] = error
        self._persist(session_id)

    def record_llm_call(
        self, session_id: str, model: str, provider: str,
        messages: list, output: str, latency_ms: float,
        agent: str = "", node: str = ""
    ):
        """Record an LLM call with latency and token estimate."""
        if not self._enabled:
            return
        inp_chars = sum(len(str(m)) for m in messages)
        out_chars = len(str(output))
        token_est = int((inp_chars / 3.5) + (out_chars / 3.5))

        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "provider": provider,
            "agent": agent,
            "node": node,
            "input_chars": inp_chars,
            "output_chars": out_chars,
            "token_estimate": token_est,
            "latency_ms": round(latency_ms, 1),
        }
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["llm_calls"].append(entry)

    def record_agent_step(
        self, session_id: str, event: dict
    ):
        """Record an agent execution step from an event."""
        if not self._enabled:
            return
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": event.get("ts", datetime.now(timezone.utc).isoformat()),
            "agent": event.get("agent", ""),
            "type": event.get("type", ""),
            "data": self._extract_data(event),
        }
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session["agent_steps"].append(entry)

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions (from in-memory + disk)."""
        with self._lock:
            result = list(self._sessions.values())
        # Also load from disk
        if os.path.exists(self._storage_dir):
            files = sorted(
                [f for f in os.listdir(self._storage_dir) if f.endswith(".json")],
                reverse=True,
            )[:limit]
            for f in files:
                try:
                    path = os.path.join(self._storage_dir, f)
                    with open(path, "r", encoding="utf-8") as fp:
                        result.append(json.load(fp))
                except Exception:
                    pass
        return result[:limit]

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    # ---- Internal ----

    def _extract_data(self, event: dict) -> dict:
        """Extract meaningful fields from an event for storage."""
        keep = {"agent", "type", "iteration", "route", "sub_agent", "sub_type"}
        data = {k: v for k, v in event.items() if k in keep and v}
        # Truncate long fields
        for key in list(data.keys()):
            val = data[key]
            if isinstance(val, str) and len(val) > 500:
                data[key] = val[:500] + "..."
            elif isinstance(val, (list, dict)):
                s = json.dumps(val, default=str)
                if len(s) > 500:
                    data[key] = s[:500] + "..."
                else:
                    data[key] = val
        return data

    def _persist(self, session_id: str):
        """Save a session trace to disk."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return
            data = dict(session)
            # Don't keep full messages in memory forever after persist
            # (keep in file, trim in memory)
            path = os.path.join(self._storage_dir, f"{session_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        except Exception:
            pass


# Global instance
tracer = TraceCollector()
