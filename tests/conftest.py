"""Shared test configuration and fixtures for evaluating Meta-Agent."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.inference.openai_compatible import ChatOpenAI

# ── LLM instance shared across tests ──
api_key = os.environ.get("LLM_API_KEY")
base_url = os.environ.get("LLM_BASE_URL")
model_id = os.environ.get("LLM_MODEL_ID")

# Default model (fast + cheap for testing)
TEST_MODEL = model_id or "deepseek-v4-flash"
TEST_BASE_URL = base_url or "https://api.deepseek.com/"
TEST_API_KEY = api_key or ""


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Create a fresh LLM instance for testing."""
    return ChatOpenAI(
        model=TEST_MODEL,
        api_key=TEST_API_KEY,
        base_url=TEST_BASE_URL,
        temperature=temperature,
    )


def make_agent(llm=None, tools=None, max_iteration=3):
    """Create a MetaAgent instance with full toolbox."""
    from src.agent.meta import MetaAgent
    from experimental import weather_tool
    from src.tool.prebuilt import (
        rag_add_document, rag_search, rag_ask,
        memory_remember, memory_recall,
        memory_learn, memory_record_event,
    )

    if llm is None:
        llm = make_llm()

    if tools is None:
        tools = [
            weather_tool,
            rag_add_document, rag_search, rag_ask,
            memory_remember, memory_recall,
            memory_learn, memory_record_event,
        ]

    agent_configs = [{
        "name": "Assistant Agent",
        "description": "通用助手，提供天气查询、知识检索、记忆管理等功能。",
        "tools": tools,
    }]

    return MetaAgent(
        agents=agent_configs,
        llm=llm,
        verbose=False,
        max_iteration=max_iteration,
        max_workers=2,
    )
