from src.message import AIMessage, BaseMessage, SystemMessage, ImageMessage, HumanMessage, ToolMessage
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from requests import RequestException, HTTPError, ConnectionError
from src.inference import BaseInference
from httpx import Client
from typing import Generator
from json import loads
import requests
import time
from uuid import uuid4


class ChatOpenAI(BaseInference):
    def _get_url(self):
        if self.base_url:
            base = self.base_url.rstrip('/')
            if base.endswith('/v1'):
                return f'{base}/chat/completions'
            return f'{base}/v1/chat/completions'
        return "https://api.openai.com/v1/chat/completions"

    @retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(RequestException))
    def invoke(self, messages: list[BaseMessage], json: bool = False) -> AIMessage:
        from src.tracing import tracer
        t0 = time.time()
        self.headers.update({'Authorization': f'Bearer {self.api_key}'})
        url = self._get_url()
        contents = []
        for message in messages:
            if isinstance(message, (SystemMessage, HumanMessage, AIMessage)):
                contents.append(message.to_dict())
            if isinstance(message, ImageMessage):
                text, image = message.content
                contents.append({
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': text},
                        {'type': 'image_url', 'image_url': {'url': image}}
                    ]
                })

        payload = {
            "model": self.model,
            "messages": contents,
            "temperature": self.temperature,
            "stream": False,
        }
        if json:
            payload["response_format"] = {"type": "json_object"}
        if self.tools:
            payload["tools"] = [{
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.schema
                }
            } for tool in self.tools]

        result = None
        try:
            with Client(timeout=120) as client:
                response = client.post(url=url, json=payload, headers=self.headers)
            json_object = response.json()
            if json_object.get('error'):
                raise HTTPError(json_object['error']['message'])
            message = json_object['choices'][0]['message']
            raw_content = message.get('content')
            if json:
                if raw_content is None:
                    result = AIMessage('')
                elif isinstance(raw_content, str):
                    try:
                        result = AIMessage(loads(raw_content))
                    except Exception:
                        result = AIMessage(raw_content)  # fallback: keep as string
                else:
                    result = AIMessage(raw_content)
            elif raw_content:
                result = AIMessage(raw_content)
            else:
                tool_calls = message.get('tool_calls')
                if tool_calls:
                    tool_call = tool_calls[0]['function']
                    result = ToolMessage(id=str(uuid4()), name=tool_call['name'], args=tool_call['arguments'])
                else:
                    result = AIMessage('')  # empty response fallback
        except HTTPError as err:
            err_object = loads(err.response.text)
            print(f'\nError: {err_object["error"]["message"]}\nStatus Code: {err.response.status_code}')
            result = AIMessage(f'Error: {err_object["error"]["message"]}')
        except ConnectionError as err:
            print(err)
            result = AIMessage(f'Error: {err}')

        # Record trace
        elapsed = (time.time() - t0) * 1000
        if self._trace_session_id:
            try:
                tracer.record_llm_call(
                    session_id=self._trace_session_id,
                    model=self.model,
                    provider="openai_compatible",
                    messages=contents,
                    output=str(result.content)[:2000] if result else "",
                    latency_ms=elapsed,
                    agent="LLM",
                    node="invoke"
                )
            except Exception:
                pass
        return result

    @retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(RequestException))
    def stream(self, messages: list[BaseMessage], json=False) -> Generator[str, None, None]:
        self.headers.update({'Authorization': f'Bearer {self.api_key}'})
        url = self._get_url()
        messages = [message.to_dict() for message in messages]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {
                "type": "json_object" if json else "text"
            },
            "stream": True,
        }
        try:
            response = requests.post(url=url, json=payload, headers=self.headers)
            response.raise_for_status()
            chunks = response.iter_lines(decode_unicode=True)
            for chunk in chunks:
                chunk = chunk.replace('data: ', '')
                if chunk and chunk != '[DONE]':
                    delta = loads(chunk)['choices'][0]['delta']
                    content = delta.get('content', '')
                    if content:
                        yield content
        except HTTPError as err:
            err_object = loads(err.response.text)
            print(f'\nError: {err_object["error"]["message"]}\nStatus Code: {err.response.status_code}')
        except ConnectionError as err:
            print(err)
        # Don't exit() — let the caller handle it
