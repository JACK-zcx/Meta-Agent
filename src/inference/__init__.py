from abc import ABC,abstractmethod
from src.message import AIMessage

class BaseInference(ABC):
    def __init__(self,model:str='',api_key:str='',tools:list=[],base_url:str='',temperature:float=0.5):
        self.model=model
        self.api_key=api_key
        self.base_url=base_url
        self.tools=tools
        self.temperature=temperature
        self.headers={'Content-Type': 'application/json'}
        self._trace_session_id: str|None = None  # set by server for tracing
    @abstractmethod
    def invoke(self,messages:list[dict])->AIMessage:
        pass