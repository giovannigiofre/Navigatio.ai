"""Daily weather forecasts from the Open-Meteo forecast API (no API key needed)."""

import asyncio
from datetime import date
from typing import Self

import httpx
from pydantic import BaseModel, ValidationError, model_validator

from app.tools.http import DEFAULT_RETRY, RetryPolicy, Sleep, UpstreamError, get_json
from app.tools.models import DailyWeather

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SERVICE = "open-meteo"

# Open-Meteo serves forecasts for today plus the next 15 days.
FORECAST_HORIZON_DAYS = 16

DAILY_VARIABLES = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
)

# WMO weather interpretation codes, as documented by Open-Meteo.
WMO_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def describe_weather_code(code: int | None) -> str:
    if code is None:
        return "unknown"
    return WMO_CODES.get(code, f"unknown (WMO code {code})")


class _Daily(BaseModel):
    time: list[date]
    weather_code: list[int | None]
    temperature_2m_max: list[float | None]
    temperature_2m_min: list[float | None]
    precipitation_sum: list[float | None]
    precipitation_probability_max: list[float | None]

    @model_validator(mode="after")
    def _same_length(self) -> Self:
        columns = [getattr(self, name) for name in DAILY_VARIABLES]
        if any(len(column) != len(self.time) for column in columns):
            raise ValueError("daily arrays have different lengths")
        return self


class _ForecastResponse(BaseModel):
    daily: _Daily


class OpenMeteoClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        retry: RetryPolicy = DEFAULT_RETRY,
        url: str = OPEN_METEO_FORECAST_URL,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._http = http
        self._retry = retry
        self._url = url
        self._sleep = sleep

    async def daily_forecast(
        self, latitude: float, longitude: float, start: date, end: date
    ) -> list[DailyWeather]:
        payload = await get_json(
            self._http,
            self._url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": ",".join(DAILY_VARIABLES),
                "timezone": "auto",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
            service=SERVICE,
            retry=self._retry,
            sleep=self._sleep,
        )
        try:
            daily = _ForecastResponse.model_validate(payload).daily
        except ValidationError as exc:
            raise UpstreamError(SERVICE, "unexpected response shape", retryable=False) from exc

        return [
            DailyWeather(
                date=day,
                summary=describe_weather_code(code),
                temperature_max_c=t_max,
                temperature_min_c=t_min,
                precipitation_mm=rain,
                precipitation_probability_pct=None if prob is None else round(prob),
            )
            for day, code, t_max, t_min, rain, prob in zip(
                daily.time,
                daily.weather_code,
                daily.temperature_2m_max,
                daily.temperature_2m_min,
                daily.precipitation_sum,
                daily.precipitation_probability_max,
                strict=True,
            )
        ]
