from src.inference.openai_compatible import ChatOpenAI
from src.agent.meta import MetaAgent
from dotenv import load_dotenv
from os import environ
from experimental import weather_tool
from src.tool.prebuilt import rag_add_document, rag_search, rag_ask

load_dotenv()
api_key = environ.get('LLM_API_KEY')
base_url = environ.get('LLM_BASE_URL')
model_id = environ.get('LLM_MODEL_ID')

query = input('Enter the query: ')
llm = ChatOpenAI(model=model_id, api_key=api_key, base_url=base_url, temperature=0.7)
agents = [
    {
        'name': 'Assistant Agent',
        'description': '通用助手，提供天气查询、知识检索、文档分析等功能。',
        'tools': [weather_tool, rag_add_document, rag_search, rag_ask]
    },
]
agent = MetaAgent(agents=agents, llm=llm, verbose=True)

for chunk in agent.stream(query):
    print(chunk)