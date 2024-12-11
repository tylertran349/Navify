import openmeteo_requests
import math
import requests_cache
import pandas as pd
from retry_requests import retry

def get_weather_data(longitude, latitude):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 38.54100201651368,
        "longitude": -121.75412639535978,
        "current": ["temperature_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    #print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    #print(f"Elevation {response.Elevation()} m asl")
    #print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    #print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Current values. The order of variables needs to be the same as requested.
    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_precipitation = current.Variables(1).Value()
    current_wind_speed_10m = current.Variables(2).Value()
    current_wind_direction_10m = current.Variables(3).Value()

    #print(f"Current time {current.Time()}")
    #print(f"Current temperature_2m {current_temperature_2m}")
    #print(f"Current precipitation {current_precipitation}")
    #print(f"Current wind_speed_10m {current_wind_speed_10m}")
    #print(f"Current wind_direction_10m {current_wind_direction_10m}")

    return current_wind_speed_10m, current_wind_direction_10m, current_precipitation