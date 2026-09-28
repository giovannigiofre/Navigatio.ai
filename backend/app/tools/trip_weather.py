"""The get_trip_weather tool: geocode a destination, then fetch its daily forecast."""

import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import Protocol

import httpx
from langchain_core.tools import BaseTool, StructuredTool

from app.tools.cache import TTLCache
from app.tools.geocoding import NominatimClient
from app.tools.http import UpstreamError
from app.tools.models import (
    DailyWeather,
    Place,
    ToolError,
    TripWeather,
    TripWeatherRequest,
    TripWeatherResult,
)
from app.tools.rate_limit import AsyncRateLimiter
from app.tools.weather import FORECAST_HORIZON_DAYS, OpenMeteoClient

logger = logging.getLogger(__name__)

TOOL_NAME = "get_trip_weather"
TOOL_DESCRIPTION = (
    "Live daily weather forecast (conditions, min/max temperature in °C, precipitation) for a "
    f"destination and trip dates. Forecasts exist only for the next {FORECAST_HORIZON_DAYS} "
    "days; returns a structured error otherwise."
)


class Geocoder(Protocol):
    async def geocode(self, query: str) -> Place | None: ...


class Forecaster(Protocol):
    async def daily_forecast(
        self, latitude: float, longitude: float, start: date, end: date
    ) -> list[DailyWeather]: ...


class TripWeatherProvider(Protocol):
    async def forecast(self, request: TripWeatherRequest) -> TripWeatherResult: ...


class TripWeatherService:
    """Composes geocoding and forecasting; converts every failure into a ToolError."""

    def __init__(
        self,
        geocoder: Geocoder,
        forecaster: Forecaster,
        *,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._geocoder = geocoder
        self._forecaster = forecaster
        self._today = today

    async def forecast(self, request: TripWeatherRequest) -> TripWeatherResult:
        today = self._today()
        horizon_end = today + timedelta(days=FORECAST_HORIZON_DAYS - 1)
        if request.end_date < today:
            return ToolError(kind="out_of_range", message="The trip dates are in the past.")
        if request.start_date > horizon_end:
            return ToolError(
                kind="out_of_range",
                message=(
                    f"Forecasts only cover the next {FORECAST_HORIZON_DAYS} days "
                    f"(until {horizon_end.isoformat()}); the trip starts on "
                    f"{request.start_date.isoformat()}."
                ),
            )
        start = max(request.start_date, today)
        end = min(request.end_date, horizon_end)

        try:
            place = await self._geocoder.geocode(request.destination)
        except UpstreamError as exc:
            logger.warning("geocoding failed", extra={"error": str(exc)})
            return ToolError(
                kind="upstream_unavailable",
                message=f"Geocoding with OpenStreetMap Nominatim failed ({exc.detail}).",
            )
        if place is None:
            return ToolError(
                kind="not_found", message=f"No place found for '{request.destination}'."
            )

        try:
            days = await self._forecaster.daily_forecast(
                place.latitude, place.longitude, start, end
            )
        except UpstreamError as exc:
            logger.warning("forecast failed", extra={"error": str(exc)})
            return ToolError(
                kind="upstream_unavailable",
                message=f"The Open-Meteo forecast request failed ({exc.detail}).",
            )

        note = None
        if (start, end) != (request.start_date, request.end_date):
            note = (
                f"Forecast covers {start.isoformat()} to {end.isoformat()} only; other trip days "
                f"are outside the {FORECAST_HORIZON_DAYS}-day forecast window."
            )
        return TripWeather(place=place, days=days, note=note)


def build_trip_weather_service(
    http: httpx.AsyncClient,
    *,
    geocode_min_interval_s: float = 1.0,
    geocode_cache_ttl_s: float = 24 * 3600,
    today: Callable[[], date] = date.today,
) -> TripWeatherService:
    """Production wiring. The rate limiter and cache live as long as the service."""
    geocoder = NominatimClient(
        http,
        rate_limiter=AsyncRateLimiter(geocode_min_interval_s),
        cache=TTLCache(ttl_s=geocode_cache_ttl_s),
    )
    return TripWeatherService(geocoder, OpenMeteoClient(http), today=today)


def create_trip_weather_tool(provider: TripWeatherProvider) -> BaseTool:
    """LangChain tool. The ToolMessage content is the validated result as JSON (for the
    model); the typed result itself travels as the message artifact (for code)."""

    async def get_trip_weather(
        destination: str, start_date: date, end_date: date
    ) -> tuple[str, TripWeatherResult]:
        request = TripWeatherRequest(
            destination=destination, start_date=start_date, end_date=end_date
        )
        result = await provider.forecast(request)
        return result.model_dump_json(), result

    return StructuredTool.from_function(
        coroutine=get_trip_weather,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=TripWeatherRequest,
        response_format="content_and_artifact",
    )
