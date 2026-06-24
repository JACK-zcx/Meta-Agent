from src.message import HumanMessage, SystemMessage
from src.router.utils import read_markdown_file
from src.inference import BaseInference
from json import dumps, loads


class LLMRouter:
    def __init__(self, instructions: list[str] = None, routes: list[dict] = None,
                 llm: BaseInference = None, verbose=False):
        self.system_prompt = read_markdown_file('./src/router/prompt.md')
        self.instructions = self.__get_instructions(instructions or [])
        self.routes = dumps(routes or [], indent=2)
        self.llm = llm
        self.verbose = verbose

    def __get_instructions(self, instructions):
        if not instructions:
            return ''
        return '\n'.join(f'{i+1}. {instr}' for i, instr in enumerate(instructions))

    def invoke(self, query: str) -> str:
        parameters = {'instructions': self.instructions, 'routes': self.routes}
        messages = [SystemMessage(self.system_prompt.format(**parameters)), HumanMessage(query or '')]
        response = self.llm.invoke(messages, json=True)

        content = response.content
        if isinstance(content, str):
            try:
                content = loads(content)
            except Exception:
                content = {}
        if not isinstance(content, dict):
            content = {}

        route = content.get('route', '') or 'generate'
        if self.verbose:
            print(f"Going to {route.upper()} route")
        return route
