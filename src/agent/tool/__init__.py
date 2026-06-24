from src.agent.tool.utils import (extract_tools_from_module,
                                           update_tool_to_module,
                                           save_tool_to_module,
                                           remove_tool_from_module,
                                           read_markdown_file)
from langchain_core.runnables.graph import MermaidDrawMethod
from src.message import HumanMessage, SystemMessage
from src.agent.tool.state import AgentState
from langgraph.graph import StateGraph, END
from importlib import import_module, reload
from IPython.display import display, Image
from src.inference import BaseInference
from src.router import LLMRouter
from src.agent import BaseAgent
from src.events import emitter as global_emitter, EventEmitter
from termcolor import colored
from subprocess import run
import json as _json
import ast
import os


def _safe_content(llm_output) -> dict:
    """Normalize LLM output (AIMessage content) to a dict. Handles str/dict/None."""
    raw = getattr(llm_output, 'content', None) if not isinstance(llm_output, (str, dict)) else llm_output
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        # Strip markdown code fences
        if s.startswith('```'):
            lines = s.split('\n')
            if len(lines) > 1:
                s = '\n'.join(lines[1:])
                if s.endswith('```'):
                    s = s[:-3].strip()
        try:
            return _json.loads(s)
        except (_json.JSONDecodeError, TypeError):
            return {"tool": raw, "name": "Generated Tool", "tool_name": "generated_tool"}
    return {}


class ToolAgent(BaseAgent):
    def __init__(self, location: str = '', llm: BaseInference = None,
                 verbose=False, json=False, emitter: EventEmitter = None):
        self.name = 'Tool Agent'
        self.location = location
        self.llm = llm
        self.verbose = verbose
        self.emitter = emitter or global_emitter
        self.graph = self.create_graph()

    def create_module(self):
        if not os.path.exists(self.location):
            content = (
                'from src.tool import tool\n'
                'from pydantic import BaseModel, Field\n'
                'from dotenv import load_dotenv\n'
                'from typing import *\n'
                'import os\n'
                'load_dotenv()\n\n'
            )
            with open(self.location, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'{self.location} has been created successfully.')

    # ── generate_tool ──
    def generate_tool(self, state: AgentState):
        self.emitter.emit('tool_router', agent='Tool', route='generate',
                          query=state.get('input'))
        system_prompt = read_markdown_file('./src/agent/tool/prompt/generate.md')
        system_message = SystemMessage(system_prompt)
        human_message = HumanMessage(f'**Query:**\n`{state.get("input", "")}`')
        tool_data = self.llm.invoke([system_message, human_message], json=True)
        content = _safe_content(tool_data)

        code = str(content.get('tool', ''))
        error = ''
        try:
            ast.parse(code)
            save_tool_to_module(self.location, code)
            self.emitter.emit('tool_code', agent='Tool', code=code,
                              tool_name=content.get('name'), route='generate')
            if self.verbose:
                print(f'{content.get("name")} has been saved to {self.location} successfully.')
        except Exception as e:
            error = e
            self.emitter.emit('tool_error', agent='Tool', error=str(e), route='generate')
            print(f'Error: {error}')
        return {**state, 'tool_data': content, 'error': error}

    # ── update_tool ──
    def update_tool(self, state: AgentState):
        existing_tool = self.find_the_tool(state.get('input') or '')
        tool_def = existing_tool.get('tool', '') if existing_tool else ''
        self.emitter.emit('tool_router', agent='Tool', route='update',
                          query=state.get('input'),
                          tool_name=(existing_tool or {}).get('tool_name', ''))

        system_prompt = read_markdown_file('./src/agent/tool/prompt/update.md')
        human_message = HumanMessage(
            f'Use the following inputs to guide the tool update:\n\n'
            f'**Tool Definition (Existing):**\n`{tool_def}`\n'
            f'**Query (Modification Required):**\n`{state.get("input", "")}`')
        system_message = SystemMessage(system_prompt)

        updated_tool_data = self.llm.invoke([system_message, human_message], json=True)
        content = _safe_content(updated_tool_data)

        code = str(content.get('tool', ''))
        error = ''
        try:
            ast.parse(code)
            update_tool_to_module(self.location, code)
            self.emitter.emit('tool_code', agent='Tool', code=code,
                              tool_name=content.get('name'), route='update')
            if self.verbose:
                print(f'Updated {content.get("name")} and saved to {self.location} successfully.')
        except Exception as e:
            error = e
            self.emitter.emit('tool_error', agent='Tool', error=str(e), route='update')
            print(f'Error: {error}')
        return {**state, 'tool_data': content, 'error': error}

    # ── debug_tool ──
    def debug_tool(self, state: AgentState):
        route = state.get('route', '')
        error_msg = ''
        tool_data_def = {}

        if route == 'debug':
            error_msg = state.get('input') or ''
            existing = self.find_the_tool(error_msg) or {}
            tool_data_def = existing
        elif route in ('generate', 'update'):
            existing = state.get('tool_data') or {}
            tool_data_def = _safe_content(existing) if not isinstance(existing, dict) else existing
            error_msg = str(state.get('error', ''))

        tool_str = str(tool_data_def.get('tool', ''))

        iteration = 0
        max_iteration = 5
        system_prompt = read_markdown_file('./src/agent/tool/prompt/debug.md')
        system_message = SystemMessage(system_prompt)
        human_message = HumanMessage(
            f'Use the following inputs to guide the tool debugging:\n\n'
            f'**Tool Definition:**\n`{tool_str}`\n'
            f'**Error Message:**\n`{error_msg}`')
        messages = [system_message, human_message]

        dbg_content = {}
        while error_msg and iteration < max_iteration:
            debug_tool_data = self.llm.invoke(messages, json=True)
            dbg_content = _safe_content(debug_tool_data)
            debug_code = str(dbg_content.get('tool', ''))

            try:
                ast.parse(debug_code)
                error_msg = ''
                if route == 'debug':
                    update_tool_to_module(self.location, debug_code)
                elif route in ('generate', 'update'):
                    save_tool_to_module(self.location, debug_code)
                self.emitter.emit('tool_code', agent='Tool', code=debug_code,
                                  tool_name=dbg_content.get('name'), route='debug',
                                  debug_iteration=iteration + 1)
                if self.verbose:
                    print(f'Fixed {dbg_content.get("name")} and saved to {self.location} successfully.')
            except Exception as e:
                err_str = str(e)
                error_msg = err_str  # keep looping
                messages.append(HumanMessage(err_str))
                self.emitter.emit('tool_error', agent='Tool', error=err_str,
                                  route='debug', debug_iteration=iteration + 1)
                print(f'Error: {err_str}')
                iteration += 1

        return {**state, 'tool_data': dbg_content, 'error': error_msg}

    # ── delete_tool ──
    def delete_tool(self, state: AgentState):
        existing = self.find_the_tool(state.get('input') or '')
        if existing is None:
            return {**state, 'tool_data': {}, 'error': 'Tool not found'}
        tool_data = {
            'name': existing.get('tool_name', ''),
            'tool_name': existing.get('func_name', ''),
            'tool': existing.get('tool', '')
        }
        remove_tool_from_module(self.location, tool_data)
        self.emitter.emit('tool_router', agent='Tool', route='delete',
                          tool_name=tool_data.get('tool_name'))
        if self.verbose:
            print(f'{tool_data.get("tool_name")} has been removed from {self.location} successfully.')
        return {**state, 'tool_data': tool_data}

    # ── reloader ──
    def reloader(self, state: AgentState):
        route = (state.get('route') or 'generate').lower()
        tool_data = state.get('tool_data') or {}
        if isinstance(tool_data, str):
            tool_data = _safe_content(tool_data)

        func_name = tool_data.get('tool_name', '')
        tool_name = tool_data.get('name', '')
        output = ''

        try:
            module = import_module(self.location.split('.')[0])
            reload(module)
            if func_name and hasattr(module, func_name):
                tool = getattr(module, func_name)
                tool_args = tool.schema if getattr(tool, 'schema', None) and tool.schema.get('properties') else {}
            else:
                tool_args = {}
        except Exception as e:
            tool_args = {'error': str(e)}

        labels = {'update': 'updated', 'debug': 'debugged', 'generate': 'generated', 'delete': 'deleted'}
        label = labels.get(route, 'processed')
        output = f'Tool Name: {tool_name}\nTool Input: {tool_args}\nTool has been {label} successfully.'
        self.emitter.emit('tool_reload', agent='Tool', route=route, tool_name=tool_name)
        return {**state, "output": output}

    # ── find_the_tool ──
    def find_the_tool(self, query: str):
        if not query:
            return None
        tools = extract_tools_from_module(self.location)
        for tool in tools:
            if tool.get('tool_name') and tool['tool_name'] in query:
                return tool
        return None

    # ── package_installer ──
    def package_installer(self, state: AgentState):
        system_prompt = read_markdown_file('./src/agent/tool/prompt/package_installer.md')
        llm_response = self.llm.invoke(
            [SystemMessage(system_prompt.format(query=state.get('input', '')))], json=True)
        response_content = _safe_content(llm_response)
        cmd = response_content.get('command', '')
        if not cmd:
            return {**state, 'output': 'No command generated.'}

        self.emitter.emit('tool_router', agent='Tool', route='package', command=cmd)
        try:
            process = run(cmd.split(' '), text=True, capture_output=True)
            output = 'Package installed successfully.' if process.returncode == 0 else f'Package installation failed: {process.stderr.strip()}'
        except Exception as e:
            output = f'Package installation error: {e}'
        return {**state, 'output': output}

    # ── router ──
    def router(self, state: AgentState):
        routes = [
            {'name': 'update', 'description': 'This route is used if the query is to update or modify an existing tool.'},
            {'name': 'debug', 'description': 'This route is used if the query is to fix errors or bugs in a tool.'},
            {'name': 'generate', 'description': 'This route is used if the query is to create or generate a new tool.'},
            {'name': 'delete', 'description': 'This route is used if the query is to delete an existing tool.'},
            {'name': 'package', 'description': 'This route is used if the query is to install/update/remove Python packages.'},
        ]
        llm_router = LLMRouter(routes=routes, llm=self.llm, verbose=False)
        route = llm_router.invoke(state.get('input') or '') or 'generate'
        return {**state, 'route': route}

    # ── controllers ──
    def controller(self, state: AgentState):
        return state.get('route') or 'generate'

    def sub_controller(self, state: AgentState):
        return 'debug' if state.get('error') else 'reloader'

    # ── graph ──
    def create_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node('router', self.router)
        workflow.add_node('package', self.package_installer)
        workflow.add_node('generate', self.generate_tool)
        workflow.add_node('update', self.update_tool)
        workflow.add_node('debug', self.debug_tool)
        workflow.add_node('delete', self.delete_tool)
        workflow.add_node('reloader', self.reloader)

        workflow.set_entry_point('router')
        workflow.add_conditional_edges('router', self.controller)
        workflow.add_conditional_edges('generate', self.sub_controller)
        workflow.add_conditional_edges('update', self.sub_controller)
        workflow.add_edge('debug', 'reloader')
        workflow.add_edge('package', END)
        workflow.add_edge('reloader', END)
        workflow.add_edge('delete', END)

        return workflow.compile(debug=False)

    def invoke(self, input: str) -> dict[str, str]:
        if self.verbose:
            print(f'Entering ' + colored(self.name, 'black', 'on_white'))
            print(colored(f'Query: {input}', color='grey', attrs=['bold']))
        self.emitter.emit('agent_start', agent='Tool', query=input)
        self.create_module()

        state = {'input': input, 'route': '', 'tool_data': {}, 'error': '', 'output': ''}
        llm_response = self.graph.invoke(state)

        tool_data = llm_response.get('tool_data') or {}
        if isinstance(tool_data, str):
            tool_data = _safe_content(tool_data)
        route = llm_response.get('route') or ''

        result = {
            'route': route,
            'func_name': tool_data.get('tool_name', ''),
            'tool_name': tool_data.get('name', ''),
            'tool': tool_data.get('tool', ''),
            'output': llm_response.get('output') or '',
        }
        self.emitter.emit('agent_end', agent='Tool', route=route,
                          tool_name=result['tool_name'])
        return result

    def stream(self, input: str):
        pass

    def get_mermaid_graph(self) -> str:
        return """graph TD
    START((Start)) --> Router[Router]
    Router -->|generate| Generate[Generate Tool]
    Router -->|update| Update[Update Tool]
    Router -->|debug| Debug[Debug Tool]
    Router -->|delete| Delete[Delete Tool]
    Router -->|package| Package[Package Installer]
    Generate -->|error| Debug
    Generate -->|success| Reloader[Reloader]
    Update -->|error| Debug
    Update -->|success| Reloader
    Debug --> Reloader
    Delete --> ENDD((End))
    Package --> ENDD
    Reloader --> ENDD"""

    def plot_mermaid(self):
        plot = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        return display(Image(plot))
