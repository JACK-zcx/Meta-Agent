from src.agent.cot.utils import extract_llm_response, read_markdown_file
from langchain_core.runnables.graph import MermaidDrawMethod
from src.message import SystemMessage, HumanMessage, AIMessage
from src.agent.cot.state import AgentState
from IPython.display import display, Image
from src.inference import BaseInference
from langgraph.graph import StateGraph
from src.agent import BaseAgent
from src.events import emitter as global_emitter, EventEmitter
from termcolor import colored


class COTAgent(BaseAgent):
    def __init__(self, name: str = '', description: str = '',
                 instructions: list[str] = [], llm: BaseInference = None,
                 max_iteration=10, json=False, verbose=False,
                 emitter: EventEmitter = None):
        self.name = name
        self.description = description
        self.instructions = self.get_instructions(instructions)
        self.llm = llm
        self.max_iteration = max_iteration
        self.graph = self.create_graph()
        self.verbose = verbose
        self.iteration = 0
        self.emitter = emitter or global_emitter
        self.system_prompt = read_markdown_file('./src/agent/cot/prompt.md')

    def get_instructions(self, instructions):
        if not instructions:
            return ''
        return '\n'.join(
            [f'{i + 1}. {instruction}' for i, instruction in
             enumerate(instructions)])

    def reason(self, state: AgentState):
        if self.max_iteration > self.iteration:
            self.emitter.emit('node_start', agent='COT', node='reason',
                              iteration=self.iteration + 1,
                              agent_name=self.name)
            llm_response = self.llm.invoke(state['messages'])
            agent_data = extract_llm_response(llm_response.content)
            if self.verbose:
                if agent_data.get('Observation'):
                    thought = agent_data.get('Thought')
                    print(colored(f'Thought: {thought}', color='green',
                                  attrs=['bold']))
                    print(colored(
                        f"Observation: {agent_data.get('Observation')}",
                        color='cyan', attrs=['bold']))
            if agent_data.get('Observation'):
                self.emitter.emit('cot_thought', agent='COT',
                                  thought=agent_data.get('Thought'),
                                  observation=agent_data.get('Observation'),
                                  agent_name=self.name,
                                  iteration=self.iteration + 1)
        else:
            agent_data = {
                'Thought': 'I reached the iteration limit',
                'Final Answer': 'Iteration limit reached'
            }
            thought_text = agent_data['Thought']
            final_answer_text = agent_data['Final Answer']
            llm_response = AIMessage(
                content=f'<Thought>{thought_text}</Thought>\n<Final-Answer>{final_answer_text}</Final-Answer>')
        return {**state, 'messages': [HumanMessage(llm_response.content)],
                'agent_data': agent_data}

    def reflection(self, state: AgentState):
        agent_data = state['agent_data']
        if self.verbose:
            if agent_data.get('Reflection'):
                thought = agent_data.get('Thought')
                reflection = agent_data.get('Reflection')
                print(colored(f'Thought: {thought}', color='green',
                              attrs=['bold']))
                print(colored(f'Reflection: {reflection}', color='magenta',
                              attrs=['bold']))
        if agent_data.get('Reflection'):
            self.emitter.emit('cot_reflection', agent='COT',
                              thought=agent_data.get('Thought'),
                              reflection=agent_data.get('Reflection'),
                              agent_name=self.name,
                              iteration=self.iteration + 1)
        return {**state, 'agent_data': agent_data}

    def controller(self, state: AgentState):
        if self.max_iteration > self.iteration:
            self.iteration += 1
            return state['agent_data']['Route'].lower()
        else:
            return 'answer'

    def final(self, state: AgentState):
        agent_data = state['agent_data']
        if self.verbose:
            if state['agent_data']['Final Answer']:
                thought = agent_data.get('Thought')
                answer = agent_data.get('Final Answer')
                print(colored(f'Thought: {thought}', color='green',
                              attrs=['bold']))
                print(
                    colored(f'Answer: {answer}', color='blue', attrs=['bold']))
        self.emitter.emit('final_answer', agent='COT',
                          answer=agent_data.get("Final Answer"),
                          agent_name=self.name)
        return {**state, 'output': agent_data.get("Final Answer")}

    def create_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node('reason', self.reason)
        graph.add_node('answer', self.final)
        graph.add_node('reflection', self.reflection)
        graph.set_entry_point('reason')
        graph.add_conditional_edges('reason', self.controller)
        graph.add_edge('reflection', 'reason')
        graph.set_finish_point('answer')

        return graph.compile(debug=False)

    def get_mermaid_graph(self) -> str:
        """Return mermaid diagram source for the COTAgent graph."""
        return """graph TD
    START((Start)) --> Reason[Reason]
    Reason -->|reflection| Reflection[Reflection]
    Reason -->|answer| Answer[Final Answer]
    Reflection --> Reason
    Answer --> ENDD((End))"""

    def plot_mermaid(self):
        plot = self.graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API)
        return display(Image(plot))

    def invoke(self, input: str):
        self.emitter.emit('agent_start', agent='COT', query=input,
                          agent_name=self.name)
        parameters = {
            'name': self.name,
            'description': self.description,
            'instructions': self.instructions,
        }
        system_prompt = self.system_prompt.format(**parameters)
        user_prompt = f'Query: {input}'
        state = {
            'input': input,
            'messages': [SystemMessage(system_prompt),
                         HumanMessage(user_prompt)],
            'output': '',
        }
        graph_response = self.graph.invoke(state)
        output = graph_response['output']
        self.emitter.emit('agent_end', agent='COT', output=output,
                          agent_name=self.name)
        return output

    def stream(self, input: str):
        pass
