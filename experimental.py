from src.tool import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import *
import os
load_dotenv()


class Weather(BaseModel):
    location: str = Field(..., description="The location to get weather information for", example=["London, UK"])


@tool("Weather Tool", args_schema=Weather)
def weather_tool(location: str) -> str:
    """
    Fetches the current weather conditions for a given location using the OpenWeatherMap API.
    """
    import requests
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OpenWeatherMap API key not set."
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    try:
        params = {"q": location, "appid": api_key, "units": "metric"}
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data["cod"] != 200:
            return f"Error: {data['message']}"
        weather_desc = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        return f"Weather in {location}:\nDescription: {weather_desc}\nTemperature: {temperature}C\nHumidity: {humidity}%\nWind Speed: {wind_speed} m/s"
    except Exception as err:
        return f"Error: {err}"
from src.tool import tool
from pydantic import BaseModel, Field
from typing import *
import os

# 全局记忆存储（模拟）
_memory_store = {}  # 实际应用中应替换为持久化存储

class MemoryDeleteTool(BaseModel):
    memory_id: Optional[str] = Field(None, description="记忆标识符", example=['mem_123'])
    content: Optional[str] = Field(None, description="记忆内容", example=['一些记忆内容'])

@tool("Memory Delete Tool", args_schema=MemoryDeleteTool)
def memory_delete_tool(memory_id: Optional[str] = None, content: Optional[str] = None) -> str:
    '''
    根据记忆标识符或内容删除记忆系统中的记忆。至少提供一个参数。
    '''
    try:
        if not memory_id and not content:
            return "错误：必须提供 memory_id 或 content 参数。"

        if memory_id:
            if memory_id in _memory_store:
                del _memory_store[memory_id]
                return f"成功删除 memory_id 为 '{memory_id}' 的记忆。"
            else:
                return f"未找到 memory_id 为 '{memory_id}' 的记忆。"

        if content:
            # 查找内容匹配的项
            matching_ids = [mid for mid, mem in _memory_store.items() if mem.get('content') == content]
            if matching_ids:
                for mid in matching_ids:
                    del _memory_store[mid]
                return f"成功删除内容为 '{content}' 的记忆，共 {len(matching_ids)} 条。"
            else:
                return f"未找到内容为 '{content}' 的记忆。"
    except Exception as err:
        return f"Error: {err}"
    return "删除操作完成。"

from src.tool import tool
from pydantic import BaseModel, Field
from typing import Optional
import os
import requests

class MemoryDelete(BaseModel):
    memory_id: Optional[str] = Field(None, description="The unique identifier of the memory to delete.", example=['mem_12345'])
    content: Optional[str] = Field(None, description="The content of the memory to delete (used if memory_id is not provided).", example=['Remember to buy milk'])

@tool("Memory Delete Tool", args_schema=MemoryDelete)
def memory_delete_tool(memory_id: Optional[str] = None, content: Optional[str] = None) -> str:
    '''
    Deletes a memory from the memory system by either its ID or its content. At least one parameter must be provided.
    '''
    if not memory_id and not content:
        return "Error: Either memory_id or content must be provided."

    api_key = os.environ.get('MEMORY_API_KEY')
    if not api_key:
        return "Error: MEMORY_API_KEY environment variable not set."

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    try:
        if memory_id:
            url = f"http://memory-api.example.com/memories/{memory_id}"
            response = requests.delete(url, headers=headers)
        else:
            url = "http://memory-api.example.com/memories/by-content"
            payload = {"content": content}
            response = requests.delete(url, headers=headers, json=payload)

        if response.status_code == 200:
            return f"Successfully deleted memory. Response: {response.text}"
        elif response.status_code == 404:
            return f"Memory not found: {response.text}"
        else:
            return f"Error deleting memory: {response.status_code} - {response.text}"
    except Exception as err:
        return f"Error: {err}"

from src.tool import tool
from pydantic import BaseModel, Field
from typing import *
import os
from datetime import datetime

class DateDifferenceCalculator(BaseModel):
    date1: str = Field(..., description="First date in YYYY-MM-DD format", example=["2023-01-01"])
    date2: str = Field(..., description="Second date in YYYY-MM-DD format", example=["2023-12-31"])

@tool("Date Difference Calculator Tool", args_schema=DateDifferenceCalculator)
def date_difference_calculator_tool(date1: str, date2: str) -> str:
    '''
    Calculates the absolute number of days between two dates in YYYY-MM-DD format.
    '''
    date_format = "%Y-%m-%d"
    try:
        d1 = datetime.strptime(date1, date_format).date()
        d2 = datetime.strptime(date2, date_format).date()
        diff = abs((d2 - d1).days)
        return str(diff)
    except ValueError as e:
        return f"Error: Invalid date format. Expected YYYY-MM-DD. Details: {e}"

