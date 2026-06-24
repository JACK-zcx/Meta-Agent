import asyncio
import json
import os
import re
import sys
import threading
from datetime import datetime
from contextlib import asynccontextmanager

# Force UTF-8 output on Windows (prevents GBK emoji crashes)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from sqlmodel import SQLModel, Session, select
import uvicorn

from src.inference.openai_compatible import ChatOpenAI
from src.agent.meta import MetaAgent
from src.events import EventEmitter
from src.tracing import tracer
from api.conversation import conversation
from api.integration import integration
from api.init_database import engine
from api.message import message
from api.models import Agent
from api.agent import agent as agent_router
from api.user import user as user_router
from api.tool import tool as tool_router
from experimental import weather_tool
from src.tool.prebuilt import (rag_add_document, rag_search, rag_ask,
                                     memory_remember, memory_recall,
                                     memory_learn, memory_record_event,
                                     get_memory_manager, get_rag_tool)

load_dotenv()
api_key = os.environ.get('LLM_API_KEY')
base_url = os.environ.get('LLM_BASE_URL')
model_id = os.environ.get('LLM_MODEL_ID')


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Server starting...')
    SQLModel.metadata.create_all(engine)
    yield
    print('Server stopping...')


app = FastAPI(
    title='Meta Agent with More Agents', version=1.0,
    description="Hierarchical multi-agent system with live visualization",
    lifespan=lifespan)

llm = ChatOpenAI(model=model_id, api_key=api_key, base_url=base_url, temperature=0.7)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


# ============================================================
#  Simple, robust approach:
#  - Use a thread-safe list to collect events
#  - Poll + push via WebSocket from the main async loop
#  - Don't rely on queue.Queue across threads in complex ways
# ============================================================

class CollectorEmitter(EventEmitter):
    """Event emitter that appends events to a list (thread-safe)."""

    def __init__(self):
        super().__init__()
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def emit(self, event_type: str, **data):
        event = {"type": event_type, "ts": datetime.now().isoformat()}
        for key, val in data.items():
            if isinstance(val, (str, int, float, bool, list, dict, type(None))):
                event[key] = val
            else:
                try:
                    event[key] = str(val)
                except Exception:
                    event[key] = repr(val)[:500]
        with self._lock:
            self.events.append(event)
        # Also forward to parent's callbacks (if any)
        super().emit(event_type, **data)

    def drain(self) -> list[dict]:
        """Atomically get and clear all pending events."""
        with self._lock:
            pending = self.events[:]
            self.events.clear()
        return pending


collector = CollectorEmitter()


async def _handle_rag_add(websocket, doc_data: dict):
    """Directly ingest a document into RAG without going through MetaAgent."""
    title = doc_data.get("title", "")
    text = doc_data.get("text", "")
    if not text:
        await websocket.send_json({"type": "error", "message": "Empty document"})
        return

    rag = get_rag_tool()
    result = rag.add_text(text, doc_id=(title or "")[:30], metadata={"title": title})
    if result.get("status") == "ok":
        msg = f"文档 '{title or result['doc_id']}' 已添加到RAG知识库 ({result['chunk_count']} chunks)"
    else:
        msg = f"添加失败: {result.get('message', 'Unknown error')}"

    await websocket.send_json({"type": "rag_confirm", "message": msg,
                                "chunks": result.get("chunk_count", 0)})
    print(f"[RAG] {msg}")


def _build_memory_context(mm, query: str) -> str:
    """Search all memory stores and build a context string for the prompt."""
    try:
        results = mm.search(query, memory_types=["working", "episodic", "semantic"], limit=3)
        parts = []

        # Working memory — most recent, highest priority
        if results.get("working"):
            parts.append("Recent context (working memory):")
            for item in results["working"]:
                parts.append(f"- {item['content'][:200]}")

        # Episodic — past conversations
        if results.get("episodic"):
            parts.append("Past interactions (episodic memory):")
            for item in results["episodic"]:
                parts.append(f"- {item['content'][:200]}")

        # Semantic — learned knowledge
        if results.get("semantic"):
            parts.append("Learned knowledge (semantic memory):")
            for item in results["semantic"]:
                name = item.get("name", "")
                content = item.get("content", "")[:150]
                parts.append(f"- {name}: {content}" if name else f"- {content}")

        return "\n".join(parts) if parts else ""
    except Exception as e:
        print(f"[Memory] Recall error: {e}")
        return ""


def _auto_record(mm, query: str, answer: str):
    """Automatically record the Q&A into working + episodic memory."""
    try:
        # Working memory: brief, high-importance
        mm.working.add(
            f"User: {query[:150]} | AI: {answer[:150]}",
            importance=0.7,
            metadata={"type": "qa_pair"}
        )
        # Episodic memory: full conversation
        mm.episodic.add(
            f"User asked: {query[:300]}\nAI answered: {answer[:300]}",
            importance=0.6,
            event_type="conversation",
            session_id=mm.session_id,
            metadata={"query": query[:200], "answer_len": len(answer)}
        )
        stats = mm.working.stats()
        print(f"[Memory] Auto-recorded. Working: {stats['count']} | Episodic: {mm.episodic.stats()['count']}")
    except Exception as e:
        print(f"[Memory] Record error: {e}")


def _serialize(obj):
    """Make any object JSON-safe and GBK-safe (strip emoji)."""
    if isinstance(obj, str):
        # Strip emoji and other non-BMP chars that can't survive GBK/JSON
        return obj.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='replace')
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    return str(obj).encode('utf-8', errors='replace').decode('utf-8', errors='replace')


@app.websocket("/ws")
async def socket(websocket: WebSocket):
    # Load / create agent configs
    with Session(engine) as session:
        db_agents = session.exec(select(Agent)).all()
        if db_agents:
            agent_configs = [{
                'name': ag.name,
                'description': ag.description,
                'tools': [eval(t.function_name) for t in ag.tools]
            } for ag in db_agents]
        else:
            agent_configs = [{
                'name': 'Assistant Agent',
                'description': '通用助手，提供天气查询、知识检索、文档分析等功能。',
                'tools': [weather_tool, rag_add_document, rag_search, rag_ask,
                          memory_remember, memory_recall,
                          memory_learn, memory_record_event]
            }]

    await websocket.accept()
    print("[WS] Client connected")

    # Get memory manager once for this connection's lifetime
    mm = get_memory_manager()

    try:
        while True:
            raw_msg = await websocket.receive_text()

            # --- Detect RAG document ingestion (JSON message) ---
            try:
                doc_data = json.loads(raw_msg)
                if doc_data.get("type") == "rag_add":
                    await _handle_rag_add(websocket, doc_data)
                    continue
            except (json.JSONDecodeError, TypeError):
                pass  # Not JSON, treat as plain text query

            query = raw_msg
            print(f"[WS] Received query: {query}")

            # --- Memory Recall: search all 3 stores before answering ---
            memory_context = _build_memory_context(mm, query)
            if memory_context:
                print(f"[Memory] Injected {len(memory_context)} chars of context")
                await websocket.send_json({"type": "memory_context",
                                           "summary": f"Recalled {memory_context[:100]}..."})

            # --- Tracing: new session ---
            session_id = tracer.new_session(query)
            llm._trace_session_id = session_id

            # Send initial status
            await websocket.send_json({"type": "status", "message": "started", "query": query})
            collector.events.clear()

            # Run agent in a thread
            meta_agent = MetaAgent(agents=agent_configs, llm=llm, verbose=True,
                                   emitter=collector, max_iteration=3)

            def run():
                try:
                    result = meta_agent.invoke(query, memory_context=memory_context)
                    collector.emit("final_result", answer=result)
                    tracer.finish_session(session_id, result)
                    # --- Auto-record to memory ---
                    _auto_record(mm, query, result)
                except Exception as e:
                    collector.emit("error", message=str(e))
                    tracer.fail_session(session_id, str(e))

            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, run)

            # Poll events while agent runs, send to WebSocket
            sent_count = 0
            while not task.done():
                pending = collector.drain()
                for evt in pending:
                    # Record agent steps in trace
                    tracer.record_agent_step(session_id, evt)
                    await websocket.send_json(_serialize(evt))
                    sent_count += 1
                await asyncio.sleep(0.1)

            # Drain final batch
            pending = collector.drain()
            for evt in pending:
                tracer.record_agent_step(session_id, evt)
                await websocket.send_json(_serialize(evt))
                sent_count += 1

            await websocket.send_json({"type": "done"})
            print(f"[WS] Sent {sent_count} events, session={session_id}")
            llm._trace_session_id = None

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


# ============================================================
# Static files
# ============================================================
static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, 'index.html'))


# ============================================================
# Trace API endpoints
# ============================================================
from fastapi import Query as FastQuery


@app.get("/api/traces")
async def list_traces(limit: int = FastQuery(default=20)):
    """List recent trace sessions."""
    sessions = tracer.list_sessions(limit)
    result = []
    for s in sessions:
        result.append({
            "session_id": s.get("session_id", ""),
            "query": s.get("query", ""),
            "started_at": s.get("started_at", ""),
            "status": s.get("status", ""),
            "total_llm_calls": s.get("total_llm_calls", 0),
            "total_llm_latency_ms": s.get("total_llm_latency_ms", 0),
            "total_agent_steps": s.get("total_agent_steps", 0),
        })
    return {"traces": result}


@app.get("/api/traces/{session_id}")
async def get_trace(session_id: str):
    """Get full trace details for a session."""
    s = tracer.get_session(session_id)
    if not s:
        # Try loading from disk
        path = os.path.join("traces", f"{session_id}.json")
        if os.path.exists(path):
            import json
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
    if not s:
        return {"error": "Session not found"}, 404
    return s


# Include API routers
app.include_router(user_router)
app.include_router(conversation)
app.include_router(integration)
app.include_router(message)
app.include_router(agent_router)
app.include_router(tool_router)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
