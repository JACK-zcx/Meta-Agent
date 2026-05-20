# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run commands

```bash
# CLI mode (simplest entry point)
python app.py

# Web server with UI (WebSocket-based)
python server.py           # starts on http://0.0.0.0:8000

# Install deps
pip install -r requirements.txt
```

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

**MetaAgent** receives the user query, calls the LLM to decide the next action, then routes to one of three branches each iteration:
- `React` — the task needs external tools → spawns a ReactAgent
- `COT` — the task is pure reasoning → spawns a COTAgent
- `Answer` — all sub-tasks done → return final answer

After React or COT completes, control returns to MetaAgent which decides the next step. This loop continues until `max_iteration` is reached or MetaAgent emits a final answer.

**ReactAgent** follows the ReAct pattern (Reason → Act → Observe). Its state graph has three routes from the `reason` node: `action` (use an existing tool), `tool` (invoke ToolAgent to create/update/debug/delete a missing tool), or `final` (answer is ready).

**ToolAgent** can dynamically generate, update, debug, or delete Python tool functions at runtime. It uses `LLMRouter` to classify the request, then an LLM to generate Python code, validates it with `ast.parse`, writes it to `experimental.py`, and reloads the module with `importlib.reload`.

**COTAgent** implements Chain-of-Thought reasoning with Reflexion: the `reason` node produces a thought + observation, the `reflection` node reviews it, and the loop continues until a final answer is reached.

### Key patterns

**Every agent is a StateGraph.** Build pattern: define state as a `TypedDict` in `state.py`, create nodes as methods, wire them in `create_graph()`, then `invoke()` or `stream()` kicks off the graph. The `controller` method is always the conditional edge router — it reads the LLM's last response and returns the next node name.

**LLM abstraction.** `src/inference/BaseInference` defines the interface: `invoke(messages, json=False) -> AIMessage` and `stream(messages) -> Generator`. Provider classes (ChatGroq, ChatGemini, ChatOpenAI, etc.) implement this. The `json=True` parameter toggles structured JSON output mode on the provider.

**Tool decorator system.** `src/tool/__init__.py` defines `@tool(name, args_schema)` which attaches `.name`, `.schema`, and `.description` attributes to functions. Tools are extracted/saved/removed from module files by parsing the AST of the source code (not runtime inspection).

**Message types.** `src/message.py` defines `SystemMessage`, `HumanMessage`, `AIMessage`, `ImageMessage`, `ToolMessage` — all inherit from `BaseMessage` and provide `to_dict()` for serialization to the LLM API format.

**Prompt files.** Each agent reads its system prompt from a `.md` file in its `prompt/` subdirectory. These prompts use `{placeholder}` format strings filled at runtime with agent metadata (name, tools, instructions).

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
