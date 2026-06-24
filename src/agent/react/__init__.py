from src.agent.react.utils import extract_llm_response, read_markdown_file
from src.message import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from src.tool.prebuilt import user_interface_tool
from src.agent.react.state import AgentState
from importlib import reload, import_module
from IPython.display import display, Image
from src.inference import BaseInference
from langgraph.graph import StateGraph
from src.agent.tool import ToolAgent
from src.agent import BaseAgent
from src.events import emitter as global_emitter, EventEmitter
from termcolor import colored
from platform import system
from getpass import getuser
from os import getcwd
import json


class ReactAgent(BaseAgent):
    def __init__(self, name: str = '', description: str = '',
                 instructions: list[str] = None, tools: list = None,
                 llm: BaseInference = None, max_iterations=10,
                 dynamic_tools_file: str = 'experimental.py', json=False,
                 verbose=False, emitter: EventEmitter = None):
        self.name = name
        self.description = description
        self.instructions = self.get_instructions(instructions or [])
        self.system_prompt = read_markdown_file('./src/agent/react/prompt.md')
        self.max_iterations = max_iterations
        self.tool_names = []
        self.tools_description = []
        self.tools = {}
        self.iteration = 0
        self.dynamic_tools_file = dynamic_tools_file
        self.dynamic_tools_module = import_module(dynamic_tools_file.split('.')[0])
        self.llm = llm
        self.verbose = verbose
        self.emitter = emitter or global_emitter
        self.graph = self.create_graph()
        self.add_tools_to_toolbox([user_interface_tool, *(tools or [])])

    def reason(self, state: AgentState):
        self.emitter.emit('node_start', agent='React', node='reason',
                          iteration=self.iteration + 1, agent_name=self.name)
        message = self.llm.invoke(state['messages'])
        response = extract_llm_response(message.content or '')
        thought = response.get('Thought', '')
        self.emitter.emit('react_thought', agent='React', thought=thought,
                          agent_name=self.name, iteration=self.iteration + 1)
        if self.verbose:
            print(colored(f'Thought: {thought}', color='green', attrs=['bold']))
        return {**state, 'messages': [message]}

    def get_instructions(self, instructions):
        if not instructions:
            return ''
        return '\n'.join(f'{i + 1}. {instr}' for i, instr in enumerate(instructions))

    def add_tools_to_toolbox(self, tools):
        for tool in tools:
            if not hasattr(tool, 'name'):
                continue
            self.tool_names.append(tool.name)
            self.tools_description.append(json.dumps({
                'Tool Name': tool.name,
                'Tool Input': tool.schema if hasattr(tool, 'schema') and tool.schema.get('properties') else {},
                'Tool Description': getattr(tool, 'description', ''),
            }, indent=2))
            self.tools[tool.name] = tool

    def update_tool_in_toolbox(self, tool):
        if not hasattr(tool, 'name'):
            return
        for index, tool_name in enumerate(self.tool_names):
            if tool_name == tool.name:
                self.tools_description[index] = json.dumps({
                    'Tool Name': tool.name,
                    'Tool Input': tool.schema if hasattr(tool, 'schema') and tool.schema.get('properties') else {},
                    'Tool Description': getattr(tool, 'description', ''),
                }, indent=2)
                self.tools[tool.name] = tool
                break

    def remove_tool_from_toolbox(self, _tool_name):
        for index, tool_name in enumerate(self.tool_names):
            if tool_name == _tool_name:
                self.tool_names.pop(index)
                self.tools_description.pop(index)
                self.tools.pop(_tool_name, None)
                break

    def action(self, state: AgentState):
        message = state['messages'][-1]
        response = extract_llm_response(message.content or '')
        thought = response.get('Thought', '')
        action_name = response.get('Action Name', '')
        action_input = response.get('Action Input')
        route = response.get('Route', '')

        # Ensure action_input is a dict
        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except (json.JSONDecodeError, TypeError):
                action_input = {}
        if not isinstance(action_input, dict):
            action_input = {}

        self.emitter.emit('react_action', agent='React', tool_name=action_name,
                          tool_input=action_input, agent_name=self.name)
        if self.verbose:
            print(colored(f'Action Name: {action_name}', color='cyan', attrs=['bold']))
            print(colored(f'Action Input: {json.dumps(action_input, indent=2)}', color='cyan', attrs=['bold']))

        if action_name not in self.tool_names:
            observation = "This tool is not available in the tool box."
        else:
            tool = self.tools[action_name]
            try:
                observation = tool(**action_input)
            except Exception as e:
                observation = str(e)

        self.emitter.emit('react_observation', agent='React',
                          observation=str(observation)[:500], agent_name=self.name)
        if self.verbose:
            print(colored(f'Observation: {observation}', color='magenta', attrs=['bold']))

        state['messages'].pop()
        messages = [
            AIMessage(f'<Thought>{thought}</Thought>\n<Action Name>{action_name}</Action Name>\n<Action Input>{json.dumps(action_input, indent=2)}</Action Input>\n<Route>{route}</Route>'),
            HumanMessage(f'<Observation>{observation}</Observation>')]
        return {**state, 'messages': messages}

    def tool_agent(self, state: AgentState):
        message = state['messages'][-1]
        response = extract_llm_response(message.content or '')
        query = response.get('Query', '')
        self.emitter.emit('tool_agent_invoke', agent='React', query=query, agent_name=self.name)

        generator = ToolAgent(location=self.dynamic_tools_file, llm=self.llm,
                              verbose=self.verbose, json=True, emitter=self.emitter)
        tool_info = generator.invoke(query or '')
        tool_name = tool_info.get('tool_name', '')
        func_name = tool_info.get('func_name', '')
        output = tool_info.get('output', '')
        route = tool_info.get('route', '')

        reload(self.dynamic_tools_module)

        if route == 'delete':
            self.remove_tool_from_toolbox(tool_name)
            content = f'{output} Now the tool is removed from the tool box.'
        else:
            try:
                tool = getattr(self.dynamic_tools_module, func_name) if func_name else None
            except Exception as e:
                print(f'getattr error: {e}')
                tool = None

            if tool is not None:
                if route == 'generate':
                    self.add_tools_to_toolbox([tool])
                elif route in ('update', 'debug'):
                    self.update_tool_in_toolbox(tool)
            content = f'{output} Now the tool is available in the tool box and ready for use.'

        if self.verbose:
            print(colored(content, color='cyan', attrs=['bold']))
        return {**state, 'messages': [HumanMessage(content)]}

    def final(self, state: AgentState):
        if self.max_iterations > self.iteration:
            message = state['messages'][-1]
            response = extract_llm_response(message.content or '')
            final_answer = response.get('Final Answer') or response.get('Thought') or ''
        else:
            final_answer = "The maximum number of iterations has been reached."
        self.emitter.emit('final_answer', agent='React', answer=final_answer, agent_name=self.name)
        if self.verbose:
            print(colored(f'Answer: {final_answer}', color='blue', attrs=['bold']))
        return {**state, 'output': final_answer}

    def controller(self, state: AgentState):
        if self.max_iterations <= self.iteration:
            return 'final'
        self.iteration += 1
        message = state['messages'][-1]
        response = extract_llm_response(message.content or '')
        route = (response.get('Route') or 'final').lower()
        return route

    def create_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node('reason', self.reason)
        workflow.add_node('action', self.action)
        workflow.add_node('final', self.final)
        workflow.add_node('tool', self.tool_agent)
        workflow.set_entry_point('reason')
        workflow.add_conditional_edges('reason', self.controller)
        workflow.add_edge('tool', 'reason')
        workflow.add_edge('action', 'reason')
        workflow.set_finish_point('final')
        return workflow.compile(debug=False)

    def get_mermaid_graph(self) -> str:
        return """graph TD
    START((Start)) --> Reason[Reason]
    Reason -->|Action| Action[Action]
    Reason -->|Tool| ToolAgent[Tool Agent]
    Reason -->|Final| Final[Final Answer]
    Action --> Reason
    ToolAgent --> Reason
    Final --> ENDD((End))"""

    def plot_mermaid(self):
        plot = self.graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        return display(Image(plot))

    def invoke(self, input: str) -> str:
        if self.verbose:
            print(f'Entering ' + colored(self.name, 'black', 'on_white'))
        self.emitter.emit('agent_start', agent='React', query=input, agent_name=self.name)
        parameters = {
            'name': self.name,
            'description': self.description,
            'instructions': self.instructions,
            'tools': f'[{", ".join(self.tools_description)}]',
            'tool_names': self.tool_names,
        }
        system_prompt = self.system_prompt.format(**parameters)
        user_prompt = f"Question:{input}\n Operating System:{system()}\nUser:{getuser()}\nCWD:{getcwd()}"
        state = {
            'input': input,
            'messages': [SystemMessage(system_prompt), HumanMessage(user_prompt)],
            'output': '',
        }
        response = self.graph.invoke(state)
        output = response.get('output', '')
        self.emitter.emit('agent_end', agent='React', output=output, agent_name=self.name)
        return output

    def stream(self, input: str):
        parameters = {
            'name': self.name,
            'description': self.description,
            'instructions': self.instructions,
            'tools': f'[{", ".join(self.tools_description)}]',
            'tool_names': self.tool_names,
        }
        system_prompt = self.system_prompt.format(**parameters)
        user_prompt = f"Question:{input}\n Operating System:{system()}\nUser:{getuser()}\nCWD:{getcwd()}"
        state = {
            'input': input,
            'messages': [SystemMessage(system_prompt), HumanMessage(user_prompt)],
            'output': '',
        }
        return self.graph.stream(state, stream_mode='values', output_keys=['output'])
