from src.inference.openai_compatible import ChatOpenAI
from src.agent.meta import MetaAgent
from dotenv import load_dotenv
from os import environ
from experimental import weather_tool

load_dotenv()
api_key = environ.get('LLM_API_KEY')
base_url = environ.get('LLM_BASE_URL')
model_id = environ.get('LLM_MODEL_ID')

query = input('Enter the query: ')
llm = ChatOpenAI(model=model_id, api_key=api_key, base_url=base_url, temperature=0.7)
agents=[
    {
        'name': 'Weather Agent',
        'description': 'This agent is responsible for providing informationn the given location.',
        'tools': [weather_tool]
    },
]
agent=MetaAgent(agents=agents,llm=llm,verbose=True)
# agent_response=agent.invoke(query)
# print(agent_response)

for chunk in agent.stream(query):
    print(chunk)