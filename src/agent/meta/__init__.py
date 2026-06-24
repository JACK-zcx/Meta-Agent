from src.agent.meta.utils import read_markdown_file
from langchain_core.runnables.graph import MermaidDrawMethod
from src.message import SystemMessage, HumanMessage
from src.agent.meta.state import AgentState
from langgraph.graph import StateGraph, END
from IPython.display import display, Image
from src.inference import BaseInference
from src.agent.react import ReactAgent
from src.agent.cot import COTAgent
from src.agent import BaseAgent
from src.events import emitter as global_emitter, EventEmitter
from termcolor import colored
from typing import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class MetaAgent(BaseAgent):
    def __init__(self, llm: BaseInference = None, agents: list[dict] = [],
                 tools: list = [], max_iteration=10, verbose=False,
                 emitter: EventEmitter = None, max_workers: int = 4):
        self.name = 'Meta Agent'
        self.description = 'This agent orchestrates problem-solving by decomposing queries and dispatching sub-agents.'
        self.llm = llm
        self.max_iteration = max_iteration
        self.iteration = 0
        self.emitter = emitter or global_emitter
        self.max_workers = max_workers
        self._dispatch_history: set = set()  # prevent duplicate dispatches
        self.agents = []
        for agent in agents:
            tool_names_str = ", ".join([tool.name for tool in agent["tools"]])
            self.agents.append(
                f"Name: {agent['name']}\nDescription: {agent['description']}\nTools: {tool_names_str}")
        self.tools = tools or [tool for agent in agents for tool in agent['tools']]
        self.tool_names = [tool.name for tool in tools]
        self.system_prompt = read_markdown_file('./src/agent/meta/prompt.md')
        self.graph = self.create_graph()
        self.verbose = verbose

    def _parse_llm_response(self, raw: str) -> dict:
        """Try to parse LLM output as JSON. Falls back gracefully."""
        raw = (raw or '').strip()
        # Remove markdown code fences if present
        if raw.startswith('```'):
            lines = raw.split('\n')
            raw = '\n'.join(lines[1:]) if len(lines) > 1 else ''
            if raw.endswith('```'):
                raw = raw[:-3]
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON embedded in text
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            return {"action": "answer", "answer": raw}

    def meta_expert(self, state: AgentState):
        self.emitter.emit('node_start', agent='Meta', node='meta_expert',
                          iteration=self.iteration + 1)
        llm_response = self.llm.invoke(state['messages'], json=True)

        # Normalize content: can be dict (json=True) or str (json=False)
        raw_content = llm_response.content
        if isinstance(raw_content, dict):
            parsed = raw_content
        elif isinstance(raw_content, str):
            parsed = self._parse_llm_response(raw_content)
        else:
            parsed = {"action": "answer", "answer": str(raw_content or '')}

        action = parsed.get('action', 'answer')
        answer = parsed.get('answer', '')
        agents_list = parsed.get('agents', [])

        if action == 'answer' or not agents_list:
            # Direct answer (or fallback)
            content = answer or (llm_response.content if isinstance(llm_response.content, str) else str(llm_response.content))
            print_stmt = colored(f'Answer: {content[:200]}', color='cyan', attrs=['bold'])
            self.emitter.emit('final_answer', agent='Meta', answer=content,
                              iteration=self.iteration + 1)
            if self.verbose:
                print(print_stmt)
            return {**state, 'current_agent': self.name,
                    'agent_data': {'action': 'answer', 'answer': content},
                    'messages': [HumanMessage(content)]}

        # Dispatch path
        if len(agents_list) > 1:
            # Parallel: multiple agents
            names = [a.get('name', '?') for a in agents_list]
            content = f'Parallel Agents: {names}'
            print_stmt = colored(content, color='yellow', attrs=['bold'])
            self.emitter.emit('meta_decision', agent='Meta', route='Parallel',
                              sub_agent=', '.join(names),
                              tasks=[], parallel_count=len(agents_list),
                              iteration=self.iteration + 1)
        else:
            # Single agent
            a = agents_list[0]
            name = a.get('name', 'Assistant Agent')
            tool = a.get('tool', None)
            route = 'React' if tool else 'COT'
            tasks = a.get('tasks', [])
            content = f'Agent: {name} | Tasks: {tasks}'
            print_stmt = colored(content, color='yellow', attrs=['bold'])
            self.emitter.emit('meta_decision', agent='Meta', route=route,
                              sub_agent=name, tasks=tasks, tool=tool,
                              iteration=self.iteration + 1)

        if self.verbose:
            print(print_stmt)

        # Store the full parsed result (with agents list) in agent_data
        return {**state, 'current_agent': agents_list[0].get('name', ''),
                'agent_data': parsed,
                'messages': [HumanMessage(json.dumps(parsed, ensure_ascii=False))]}

    def _get_first_agent(self, parsed: dict) -> dict:
        """Extract the first agent info from a dispatch payload."""
        agents = parsed.get('agents', [])
        if not agents:
            return {}
        a = agents[0]
        return {
            'name': a.get('name', 'Assistant Agent'),
            'query': a.get('query', ''),
            'description': a.get('description', ''),
            'tasks': a.get('tasks', []),
            'tool': a.get('tool'),
        }

    def _run_single_agent(self, agent_info: dict) -> dict:
        """Execute a single sub-agent. agent_info keys: name, query, tasks, tool, description."""
        name = agent_info.get('name', 'Unknown')
        query = agent_info.get('query', '')
        instructions = agent_info.get('tasks', [])
        tool = agent_info.get('tool')
        description = agent_info.get('description', '')

        if tool:
            self.emitter.emit('agent_delegate', agent='Meta', sub_agent=name,
                              sub_type='React', query=query)
            agent = ReactAgent(name=name, description=description,
                               instructions=instructions, tools=self.tools,
                               llm=self.llm, verbose=self.verbose, emitter=self.emitter)
        else:
            self.emitter.emit('agent_delegate', agent='Meta', sub_agent=name,
                              sub_type='COT', query=query)
            agent = COTAgent(name=name, description=description,
                             instructions=instructions, llm=self.llm,
                             verbose=self.verbose, emitter=self.emitter)

        try:
            response = agent.invoke(f'Query: {query}')
            resp_preview = response[:200] + '...' if len(response) > 200 else response
            self.emitter.emit('agent_return', agent='Meta', sub_agent=name,
                              response=resp_preview)
            return {'name': name, 'response': response, 'error': None}
        except Exception as e:
            self.emitter.emit('agent_return', agent='Meta', sub_agent=name,
                              response=f'Error: {e}')
            return {'name': name, 'response': f'Error: {e}', 'error': str(e)}

    def parallel_expert(self, state: AgentState):
        """Execute multiple sub-agents concurrently."""
        parsed = state.get('agent_data') or {}
        agents_list = parsed.get('agents', [])

        self.emitter.emit('node_start', agent='Meta', node='parallel_expert',
                          parallel_count=len(agents_list))

        responses = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(agents_list))) as executor:
            futures = {executor.submit(self._run_single_agent, a): a for a in agents_list}
            for future in as_completed(futures):
                result = future.result()
                responses.append(result)

        combined = '\n\n'.join(
            f"Name: {r['name']}\nResponse: {r['response']}" for r in responses)
        if self.verbose:
            print(colored(f'Parallel results:\n{combined}', color='yellow', attrs=['bold']))

        return {**state, 'messages': [HumanMessage(combined)], 'agent_data': None}

    def react_expert(self, state: AgentState):
        """Handle single React agent dispatch."""
        parsed = state.get('agent_data') or {}
        info = self._get_first_agent(parsed)
        name = info['name']
        query = info['query']
        description = info['description']
        instructions = info['tasks']

        self.emitter.emit('agent_delegate', agent='Meta', sub_agent=name,
                          sub_type='React', query=query)
        agent = ReactAgent(name=name, description=description,
                           instructions=instructions, tools=self.tools,
                           llm=self.llm, verbose=self.verbose, emitter=self.emitter)
        agent_response = agent.invoke(f'Query: {query}')

        # Wrap with clear completion marker so MetaAgent's LLM recognizes it
        completed_msg = (
            f'[TASK COMPLETED — DO NOT DISPATCH AGAIN]\n'
            f'Agent: {name}\n'
            f'Result: {agent_response}'
        )

        resp_preview = agent_response[:200] + '...' if len(agent_response) > 200 else agent_response
        self.emitter.emit('agent_return', agent='Meta', sub_agent=name,
                          response=resp_preview)
        # Return None agent_data so controller knows this dispatch is done
        return {**state,
                'messages': [HumanMessage(completed_msg)],
                'agent_data': None,  # Signal: dispatch complete
                '_dispatched': name}  # Track what was dispatched

    def cot_expert(self, state: AgentState):
        """Handle single COT agent dispatch."""
        parsed = state.get('agent_data') or {}
        info = self._get_first_agent(parsed)
        name = info['name']
        query = info['query']
        description = info['description']
        instructions = info['tasks']

        self.emitter.emit('agent_delegate', agent='Meta', sub_agent=name,
                          sub_type='COT', query=query)
        agent = COTAgent(name=name, description=description,
                         instructions=instructions, llm=self.llm,
                         verbose=self.verbose, emitter=self.emitter)
        agent_response = agent.invoke(f'Query: {query}')

        completed_msg = (
            f'[TASK COMPLETED — DO NOT DISPATCH AGAIN]\n'
            f'Agent: {name}\n'
            f'Result: {agent_response}'
        )

        resp_preview = agent_response[:200] + '...' if len(agent_response) > 200 else agent_response
        self.emitter.emit('agent_return', agent='Meta', sub_agent=name,
                          response=resp_preview)
        return {**state,
                'messages': [HumanMessage(completed_msg)],
                'agent_data': None,
                '_dispatched': name}

    def final(self, state: AgentState):
        if self.max_iteration > self.iteration:
            output = state['messages'][-1].content
        else:
            output = 'Iteration limit reached'
        return {**state, 'output': output, 'current_agent': ''}

    def controller(self, state: AgentState):
        """Route based on parsed JSON action field. Prevents duplicate dispatches."""
        if self.max_iteration <= self.iteration:
            return 'Answer'
        self.iteration += 1

        agent_data = state.get('agent_data') or {}

        # Safety 1: If previous dispatch returned None for agent_data, it means task completed
        if not agent_data:
            return 'Answer'

        # Safety 2: agent_data already cleared → previous dispatch finished
        action = agent_data.get('action', 'answer')
        if action == 'answer':
            return 'Answer'

        # Safety 3: Dispatch path — check for duplicates
        if action == 'dispatch':
            agents_list = agent_data.get('agents', [])
            if not agents_list:
                return 'Answer'

            # Dedup: create a fingerprint of this dispatch (name + task)
            fp = tuple((a.get('name', ''), a.get('query', '')) for a in agents_list)
            if fp in self._dispatch_history:
                # Same dispatch already executed → stop
                return 'Answer'
            self._dispatch_history.add(fp)

            if len(agents_list) > 1:
                return 'Parallel'
            if len(agents_list) == 1:
                a = agents_list[0]
                return 'React' if a.get('tool') else 'COT'

        # Fallback: anything unknown → answer
        return 'Answer'

    def create_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node('Meta', self.meta_expert)
        graph.add_node('React', self.react_expert)
        graph.add_node('COT', self.cot_expert)
        graph.add_node('Parallel', self.parallel_expert)
        graph.add_node('Answer', self.final)

        graph.set_entry_point('Meta')
        graph.add_conditional_edges('Meta', self.controller)
        graph.add_edge('React', 'Meta')
        graph.add_edge('COT', 'Meta')
        graph.add_edge('Parallel', 'Meta')
        graph.add_edge('Answer', END)

        return graph.compile(debug=False)

    def get_mermaid_graph(self) -> str:
        return """graph TD
    START((Start)) --> Meta[Meta Expert]
    Meta -->|React| React[React Expert]
    Meta -->|COT| COT[COT Expert]
    Meta -->|Parallel| Parallel[Parallel Expert<br/>ThreadPoolExecutor]
    Meta -->|Answer| Answer[Final Answer]
    React --> Meta
    COT --> Meta
    Parallel --> Meta
    Answer --> ENDD((End))"""

    def plot_mermaid(self):
        plot = self.graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API)
        return display(Image(plot))

    def invoke(self, input: str, memory_context: str = "") -> str:
        if self.verbose:
            print(f'Entering ' + colored(self.name, 'black', 'on_white'))
        self.emitter.emit('agent_start', agent='Meta', query=input)
        self._dispatch_history.clear()
        self.iteration = 0
        system_prompt = self.system_prompt.format(
            agents='\n\n'.join(self.agents), tool_names=self.tool_names)
        if memory_context:
            system_prompt += f"\n\n## Recent Memory Context\n{memory_context}\nUse this context if relevant."
        user_prompt = f'User Query: {input}'
        state = {
            'input': input,
            'messages': [SystemMessage(system_prompt), HumanMessage(user_prompt)],
            'agent_data': {},
            'current_agent': self.name,
            'output': '',
        }
        response = self.graph.invoke(state)
        output = response['output']
        self.emitter.emit('agent_end', agent='Meta', output=output)
        print('=' * 100)
        return output

    def stream(self, input: str):
        if self.verbose:
            print(f'Entering ' + colored(self.name, 'black', 'on_white'))
        self.emitter.emit('agent_start', agent='Meta', query=input)
        system_prompt = self.system_prompt.format(
            agents='\n\n'.join(self.agents), tool_names=self.tool_names)
        user_prompt = f'User Query: {input}'
        state = {
            'input': input,
            'messages': [SystemMessage(system_prompt), HumanMessage(user_prompt)],
            'agent_data': {},
            'current_agent': self.name,
            'output': '',
        }
        for chunk in self.graph.stream(state, stream_mode='values',
                                       output_keys=['output']):
            self.emitter.emit('stream_chunk', agent='Meta', chunk=chunk)
            yield chunk
        self.emitter.emit('agent_end', agent='Meta')
