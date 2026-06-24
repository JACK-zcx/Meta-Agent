from src.agent.tool.utils import read_markdown_file
from src.message import HumanMessage, SystemMessage
from src.inference import BaseInference
import ast


def generate(query: str, llm: BaseInference):
    system_prompt = read_markdown_file('./src/agent/tool/prompt/generate.md')
    messages = [SystemMessage(system_prompt), HumanMessage(f'**Query:**\n`{query}`')]
    tool_data = llm.invoke(messages, json=True)

    # Normalize content — can be dict (json=True) or str (json=False or fallback)
    content = tool_data.content
    if isinstance(content, str):
        import json
        try:
            content = json.loads(content)
        except Exception:
            content = {"tool": content, "name": "Generated Tool", "tool_name": "generated_tool"}
    if not isinstance(content, dict):
        content = {}

    error = ''
    while True:
        try:
            code = content.get('tool', '')
            if code:
                ast.parse(code)
            error = ''
            break
        except Exception as e:
            error = str(e)
            print(f'Error: {error}')
            messages.append(HumanMessage(error))
            retry_data = llm.invoke(messages, json=True)
            content = retry_data.content
            if isinstance(content, str):
                import json
                try:
                    content = json.loads(content)
                except Exception:
                    content = {"tool": content, "name": "Generated Tool", "tool_name": "generated_tool"}
            if not isinstance(content, dict):
                content = {}

    return {
        'name': content.get('name') or '',
        'tool_name': content.get('tool_name') or '',
        'tool': content.get('tool') or '',
    }
