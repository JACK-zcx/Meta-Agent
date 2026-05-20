from src.tool import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import os
load_dotenv()

class Weather(BaseModel):
    location: str = Field(..., description='The location to get weather information for', example=['London, UK'])

@tool('Weather Tool', args_schema=Weather)
def weather_tool(location: str) -> str:
    """
    Fetches the current weather conditions for a given location using the OpenWeatherMap API.
    Uses the OPENWEATHER_API_KEY environment variable for authentication.
    If the key is not set, returns an error.
    """
    import requests
    import os
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        return 'Error: OpenWeatherMap API key not set. Set the OPENWEATHER_API_KEY environment variable.'
    base_url = 'http://api.openweathermap.org/data/2.5/weather'
    try:
        params = {'q': location, 'appid': api_key, 'units': 'metric'}
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['cod'] != 200:
            return f"Error: {data['message']}"
        weather_desc = data['weather'][0]['description']
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        result = f'Weather in {location}:\nDescription: {weather_desc}\nTemperature: {temperature}°C\nHumidity: {humidity}%\nWind Speed: {wind_speed} m/s'
    except requests.exceptions.RequestException as err:
        return f'Error: {err}'
    except Exception as err:
        return f'Error: {err}'
    return result

