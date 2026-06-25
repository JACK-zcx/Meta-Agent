"""τ2-bench Adapter — using native OpenAI function calling, not text prompts."""

import sys, os, json

for candidate in [
    r"F:\desktop\Meta-Agent-with-More-Agents-main\tau2-bench-main\src",
    r"F:\desktop\tau2-bench-main\src",
]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

PROJECT = r"F:\desktop\Meta-Agent-with-More-Agents-main"
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage, ToolCall, UserMessage, ToolMessage, MultiToolMessage,
)
from tau2.environment.tool import Tool as TauTool
from pydantic import BaseModel, Field, create_model
from typing import Any


# ═══════════════════════════════════════════════════════════════
# Convert τ2-bench Tool → our project @tool function
# ═══════════════════════════════════════════════════════════════

def tau_tool_to_project_tool(tau_tool: TauTool):
    """τ2-bench Tool → @tool-decorated function ReactAgent can call."""
    from src.tool import tool as our_tool_decorator
    schema = tau_tool.params.model_json_schema()
    props = schema.get("properties", {})
    required = schema.get("required", [])
    fields = {}
    for name, info in props.items():
        python_type = {"string": str, "integer": int, "number": float, "boolean": bool}.get(info.get("type", "string"), str)
        if name in required:
            fields[name] = (python_type, Field(..., description=info.get("description", "")))
        else:
            fields[name] = (python_type, Field(default=None, description=info.get("description", "")))
    if not fields:
        fields["_"] = (str, Field(default="", description="No params"))
    params_cls = create_model(f"{tau_tool.name}_Params", **fields)

    def _wrapper(**kwargs):
        try:
            import json
            result = tau_tool(**kwargs)
            return json.dumps(str(result))
        except Exception as e:
            return f"Error: {e}"

    _wrapper.__name__ = tau_tool.name
    _wrapper.__doc__ = tau_tool.short_desc or tau_tool.long_desc or tau_tool.name
    return our_tool_decorator(tau_tool.name, args_schema=params_cls)(_wrapper)


# ═══════════════════════════════════════════════════════════════
# OpenAI function calling utility
# ═══════════════════════════════════════════════════════════════

def _build_openai_tools(tau_tools: list[TauTool]) -> list[dict]:
    """Convert τ2-bench tools to OpenAI function-calling format."""
    result = []
    for t in tau_tools:
        schema = t.params.model_json_schema()
        result.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.short_desc or t.long_desc or t.name)[:500],
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        })
    return result


def register_meta_agent():
    from tau2.registry import registry
    def factory(tools, domain_policy, **kwargs):
        return HalfDuplexAdapter(tools=tools, domain_policy=domain_policy)
    registry.register_agent_factory(factory, "meta_agent")


def _generate_for_user_simulator(model, tools, messages, call_name="", **kwargs):
    """Called by patched user_simulator to bypass LiteLLM. Uses our ChatOpenAI."""
    from dotenv import load_dotenv
    load_dotenv()
    from src.inference.openai_compatible import ChatOpenAI
    from src.message import SystemMessage, HumanMessage
    from tau2.data_model.message import AssistantMessage, ToolCall

    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL_ID", "gpt-3.5-turbo"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_BASE_URL", ""),
        temperature=0.7,
    )
    tool_names = {t.name for t in tools} if tools else set()
    lines = []
    for m in messages:
        r = getattr(m, "role", "user"); c = getattr(m, "content", "") or ""
        lines.append(f"[{r.upper()}] {c}")
    prompt = (
        "You are simulating a user. Follow the instructions.\n"
        f"History:\n" + "\n".join(lines[-15:])
    )
    resp = llm.invoke([SystemMessage("You are a user."), HumanMessage(prompt)])
    raw = (resp.content if hasattr(resp, "content") else str(resp)).strip()[:2000]
    # Try tool call first (user can call user tools)
    try:
        cleaned = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "name" in obj and (not tool_names or obj["name"] in tool_names):
            return AssistantMessage(role="assistant", tool_calls=[
                ToolCall(id="c_usr", name=obj["name"], arguments=obj.get("arguments", {}), requestor="assistant")
            ])
    except Exception:
        pass
    return AssistantMessage(role="assistant", content=raw)


# ═══════════════════════════════════════════════════════════════
# The Adapter — uses native function calling via ChatOpenAI.tools
# ═══════════════════════════════════════════════════════════════

class HalfDuplexAdapter(HalfDuplexAgent):

    def __init__(self, tools: list[TauTool], domain_policy: str):
        super().__init__(tools=tools, domain_policy=domain_policy)

        # Convert τ2-bench tools → OpenAI function calling schema
        self._openai_tools = _build_openai_tools(tools)
        self._tool_names = {t.name for t in tools}

        from dotenv import load_dotenv
        load_dotenv()
        from src.inference.openai_compatible import ChatOpenAI

        # Create LLM WITHOUT tools set (we'll set them dynamically per call)
        self._llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL_ID"),
            api_key=os.environ.get("LLM_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL"),
            temperature=0.0,
            tools=[],  # No static tools — we build payload manually
        )
        self._policy = domain_policy

    # ── API helpers ──

    def _call_llm_with_tools(self, messages: list) -> dict:
        """
        Call DeepSeek API with native function calling.
        Returns: {"type": "text", "content": "..."} or {"type": "tool", "name": "...", "args": {...}}
        """
        url = self._llm._get_url()
        headers = {"Authorization": f"Bearer {self._llm.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self._llm.model,
            "messages": messages,
            "temperature": 0.0,
            "tools": self._openai_tools,
            "stream": False,
            "thinking": None,  # disable R1 thinking mode (DeepSeek)
        }
        import httpx
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(url=url, json=payload, headers=headers)
            data = response.json()
            if data.get("error"):
                return {"type": "text", "content": f"Error: {data['error']['message']}"}
            choice = data["choices"][0]
            msg = choice.get("message", {})
            if msg.get("tool_calls"):
                tc = msg["tool_calls"][0]["function"]
                return {"type": "tool", "name": tc["name"], "args": json.loads(tc.get("arguments", "{}"))}
            elif msg.get("content"):
                content = msg["content"]
                # Strip DeepSeek R1 reasoning_content if present (causes API error on next call)
                content = content.split("<｜end▁of▁thinking｜>:")[-1].strip() if " response:" in content else content
                return {"type": "text", "content": content}
            else:
                return {"type": "text", "content": "Unable to process request."}
        except Exception as e:
            return {"type": "text", "content": f"Error: {e}"}

    # ── HalfDuplexAgent API ──

    def get_init_state(self, message_history=None):
        return {"api_messages": [], "n": 0}

    def generate_next_message(self, message, state):
        state["n"] += 1

        # Convert incoming τ2-bench message to OpenAI API message format
        if isinstance(message, UserMessage):
            if not message.content and not getattr(message, "tool_calls", None):
                # Empty user message — silently handle
                msg = AssistantMessage(role="assistant", content="Please let me know how I can help.")
                state.setdefault("api_messages", []).append({"role": "assistant", "content": "Please let me know how I can help."})
                return msg, state

            # Save user message
            user_content = message.content or ""
            api_msg = {"role": "user", "content": user_content}
            state.setdefault("api_messages", []).append(api_msg)

        elif isinstance(message, ToolMessage):
            # τ2-bench executed a tool → add tool result to conversation
            tool_result = message.content or ""
            tool_name = getattr(message, "name", "unknown")
            tool_call_id = getattr(message, "id", "call_0")
            api_msg = {"role": "tool", "tool_call_id": tool_call_id, "name": tool_name,
                       "content": tool_result}
            state.setdefault("api_messages", []).append(api_msg)

        elif isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                api_msg = {"role": "tool", "tool_call_id": getattr(tm, "id", "call_0"),
                           "name": getattr(tm, "name", "unknown"), "content": tm.content or ""}
                state.setdefault("api_messages", []).append(api_msg)

        # Prepare system prompt
        system_msg = {
            "role": "system",
            "content": f"You are a helpful assistant. Follow this policy:\n{self._policy[:1000]}\n\n"
                       f"When you need to perform an action, use the available tools. "
                       f"After a tool returns a result, respond to the user."
        }

        # Call LLM with tools
        call_messages = [system_msg] + state["api_messages"][-20:]
        result = self._call_llm_with_tools(call_messages)

        if result["type"] == "tool":
            # Agent wants to call a tool → return to τ2-bench
            tc = ToolCall(
                id=f"c{state['n']}",
                name=result["name"],
                arguments=result["args"],
                requestor="assistant",
            )
            msg = AssistantMessage(role="assistant", tool_calls=[tc])
            # Track in our message history too
            state["api_messages"].append({"role": "assistant", "tool_calls": [{
                "id": f"c{state['n']}", "type": "function",
                "function": {"name": result["name"], "arguments": json.dumps(result["args"])}
            }]})
        else:
            # Text response
            msg = AssistantMessage(role="assistant", content=result["content"][:2000])
            state["api_messages"].append({"role": "assistant", "content": result["content"][:2000]})

        return msg, state
