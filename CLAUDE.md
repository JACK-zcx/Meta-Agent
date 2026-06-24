# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run commands

```bash
# CLI mode (simplest entry point)
python app.py

# Web server with live visualization UI (WebSocket-based)
python server.py           # starts on http://0.0.0.0:8000
                           # Open browser to http://localhost:8000 for live agent topology + event stream

# Install deps
pip install -r requirements.txt
```

### Live visualization UI (new)

When `server.py` is running, the root URL (`http://localhost:8000`) serves a real-time monitoring dashboard:
- **Left panel** — Chat interface: user queries and AI answers displayed as conversation bubbles
- **Right panel** — Three-tab view:
  - *Trace* — Colored cards showing each agent step (Thought/Action/Observation/Answer) in chronological order
  - *Code* — Real-time Python code preview when ToolAgent generates tools
  - *Stats* — Post-execution performance metrics: LLM call count, latency, estimated tokens, agent step breakdown
- WebSocket-driven: every agent node emits structured events pushed to the browser instantly

### Observability / Tracing (new)

`src/tracing.py` provides a lightweight self-contained tracing system:
- **LLM call recording**: latency (ms), estimated token count, model/provider info, input/output character counts
- **Agent step recording**: every agent event is logged with timestamp, agent name, event type
- **JSON file storage**: traces persisted to `traces/` directory for later analysis
- **REST API**: `GET /api/traces` lists recent sessions, `GET /api/traces/{id}` returns full trace detail
- Integrated into `server.py` WebSocket flow — each query automatically creates a new trace session
- Also fixed `ChatOpenAI.invoke()` to remove `exit()` call on error (now returns error message gracefully)

## Environment

Requires `.env` at project root. Two inference backends are supported:

- **OpenAI-compatible** (default for app.py): set `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_ID`
- **Gemini** (default for server.py): set `GOOGLE_API_KEY`

The codebase hardcodes per-file imports — `app.py` imports `ChatOpenAI` from `src/inference/openai_compatible.py`, while `server.py` imports `ChatGemini` from `src/inference/gemini.py`. To switch providers, change the import and LLM class in the entry point.

Optional tool API keys: `OPENWEATHERMAP_API_KEY`, `SERPAPI_API_KEY`.

## Architecture

This is a hierarchical multi-agent system built on **LangGraph StateGraph**. Agents are not LLM-call wrappers — each agent is a **state machine** whose nodes are LLM-invocation steps and whose edges define the agent's decision loop.

### Agent hierarchy (top-down orchestration)

```
MetaAgent (src/agent/meta/)          ← top-level orchestrator
  ├─ ReactAgent (src/agent/react/)   ← tool-using agent
  │     └─ ToolAgent (src/agent/tool/) ← dynamic tool code generation
  └─ COTAgent (src/agent/cot/)       ← reasoning-only agent (Chain-of-Thought + Reflexion)
```

### Memory system (new)

The project has a three-tier memory system (`src/memory/`):

| Memory Type | Storage | TTL | Purpose |
|---|---|---|---|
| WorkingMemory | Pure in-memory dict | 60min | Temporary session info, fast access |
| EpisodicMemory | SQLite + ChromaDB | Persistent | Event sequences, conversations, session logs |
| SemanticMemory | SQLite (entity graph) + ChromaDB | Persistent | Concepts, entity relations, knowledge graph |

**MemoryManager** (`src/memory/manager.py`) orchestrates all three types:
- Smart routing: `add(content, memory_type='auto')` heuristically routes to the right store
- Unified search across all stores simultaneously
- Consolidation: promotes high-importance working → episodic memory
- Shared embedding function with RAG tool (API-based)

**MetaAgent** receives the user query, calls the LLM with `json=True` for structured output, then routes to one of four branches each iteration:
- `React` — the task needs external tools → spawns a ReactAgent
- `COT` — the task is pure reasoning → spawns a COTAgent
- `Parallel` — multiple independent sub-tasks → spawns multiple agents concurrently via ThreadPoolExecutor
- `Answer` — all sub-tasks done → return final answer

After React/COT/Parallel completes, control returns to MetaAgent which decides the next step. This loop continues until `max_iteration` is reached or MetaAgent emits a final answer.

### Structured JSON decision protocol (recently upgraded from XML)

MetaAgent now uses `json=True` + a JSON Schema prompt. The LLM outputs:
```json
{"action": "answer", "answer": "Direct response text..."}
// or
{"action": "dispatch", "agents": [{"name": "...", "query": "...", "tasks": [...], "tool": "..."}]}
```

`controller()` reads the `action` field deterministically — no fragile XML parsing. A `_parse_llm_response()` fallback handles malformed output by treating raw text as an answer. This eliminated "Agent Name: None" errors and the COTAgent NoneType crash.

### Parallel agent execution (new)

When MetaAgent detects multiple independent sub-tasks, it outputs multiple `<Agent>` XML blocks. The `parallel_expert` node runs them concurrently using `concurrent.futures.ThreadPoolExecutor`. Results are collected and fed back to MetaAgent for final synthesis. Configurable via `max_workers` parameter (default 4). The LLM decides whether to use parallel vs sequential based on task dependencies.

**ReactAgent** follows the ReAct pattern (Reason → Act → Observe). Its state graph has three routes from the `reason` node: `action` (use an existing tool), `tool` (invoke ToolAgent to create/update/debug/delete a missing tool), or `final` (answer is ready).

**ToolAgent** can dynamically generate, update, debug, or delete Python tool functions at runtime. It uses `LLMRouter` to classify the request, then an LLM to generate Python code, validates it with `ast.parse`, writes it to `experimental.py`, and reloads the module with `importlib.reload`.

**COTAgent** implements Chain-of-Thought reasoning with Reflexion: the `reason` node produces a thought + observation, the `reflection` node reviews it, and the loop continues until a final answer is reached.

### Key patterns

**Every agent is a StateGraph.** Build pattern: define state as a `TypedDict` in `state.py`, create nodes as methods, wire them in `create_graph()`, then `invoke()` or `stream()` kicks off the graph. The `controller` method is always the conditional edge router — it reads the LLM's last response and returns the next node name.

**LLM abstraction.** `src/inference/BaseInference` defines the interface: `invoke(messages, json=False) -> AIMessage` and `stream(messages) -> Generator`. Provider classes (ChatGroq, ChatGemini, ChatOpenAI, etc.) implement this. The `json=True` parameter toggles structured JSON output mode on the provider.

**Tool decorator system.** `src/tool/__init__.py` defines `@tool(name, args_schema)` which attaches `.name`, `.schema`, and `.description` attributes to functions. Tools are extracted/saved/removed from module files by parsing the AST of the source code (not runtime inspection).

**Message types.** `src/message.py` defines `SystemMessage`, `HumanMessage`, `AIMessage`, `ImageMessage`, `ToolMessage` — all inherit from `BaseMessage` and provide `to_dict()` for serialization to the LLM API format.

**Prompt files.** Each agent reads its system prompt from a `.md` file in its `prompt/` subdirectory. These prompts use `{placeholder}` format strings filled at runtime with agent metadata (name, tools, instructions).

**RAG (Retrieval-Augmented Generation).** `src/tool/rag.py` provides a self-contained RAG pipeline using ChromaDB (local vector store) + LLM API embeddings. Three sub-tools are exposed to agents: `rag_add_document` (ingest), `rag_search` (retrieve), `rag_ask` (context for answering). Embedding priority: LLM API → local SentenceTransformer → keyword fallback.

### Inference backends

| File | Class | Notes |
|------|-------|-------|
| `src/inference/openai_compatible.py` | ChatOpenAI | Generic OpenAI-compatible (DeepSeek, etc.) |
| `src/inference/groq.py` | ChatGroq | Groq API |
| `src/inference/gemini.py` | ChatGemini | Google Gemini |
| `src/inference/mistral.py` | ChatMistral | Mistral API |
| `src/inference/ollama.py` | ChatOllama | Local Ollama |

### API layer (server.py only)

`server.py` is a FastAPI app with WebSocket endpoint `/ws` for streaming agent responses, plus REST routes for user/agent/tool/conversation CRUD backed by SQLModel/SQLite.

## Encoding and compatibility

- Python source files and prompts **must be UTF-8**. On Windows Chinese editions, `open()` defaults to GBK — always pass `encoding='utf-8'`.
- The codebase originally used Python 3.12 f-string syntax (same-type quotes nested inside f-strings). When running on Python 3.11, extract nested expressions to variables before interpolating.
- `.py` files with BOM or GBK-encoded non-ASCII characters (like `°`) will cause `SyntaxError` — rewrite the file as proper UTF-8.
