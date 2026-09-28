import asyncio
import json
from datetime import date

import httpx
import pytest
from langchain_core.messages import ToolMessage

from app.tools.cache import TTLCache
from app.tools.geocoding import NOMINATIM_SEARCH_URL, NominatimClient
from app.tools.http import RetryPolicy, UpstreamError
from app.tools.models import Place, ToolError, TripWeather, TripWeatherRequest
from app.tools.rate_limit import AsyncRateLimiter
from app.tools.trip_weather import TripWeatherService, create_trip_weather_tool
from app.tools.weather import OPEN_METEO_FORECAST_URL, OpenMeteoClient

TODAY = date(2026, 9, 28)
USER_AGENT = "Navigatio.ai-tests/0.1"
RETRY = RetryPolicy(max_attempts=3, initial_backoff_s=0.5, max_backoff_s=4.0, jitter_s=0.5)

LISBON_HIT = [
    {
        "place_id": 1,
        "lat": "38.7077507",
        "lon": "-9.1365919",
        "display_name": "Lisbon, Portugal",
        "address": {"city": "Lisbon", "country": "Portugal", "country_code": "pt"},
    }
]
FORECAST = {
    "latitude": 38.71,
    "longitude": -9.14,
    "timezone": "Europe/Lisbon",
    "daily": {
        "time": ["2026-10-03", "2026-10-04"],
        "weather_code": [3, 61],
        "temperature_2m_max": [24.1, 21.0],
        "temperature_2m_min": [16.2, 15.8],
        "precipitation_sum": [0.0, 4.2],
        "precipitation_probability_max": [5, 70],
    },
}
TRIP = TripWeatherRequest(
    destination="Lisbon", start_date=date(2026, 10, 3), end_date=date(2026, 10, 4)
)


class FakeClock:
    """Monotonic clock whose sleep() advances time instantly and records the delay."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
async def http():
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        yield client


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_service(http: httpx.AsyncClient, clock: FakeClock) -> TripWeatherService:
    geocoder = NominatimClient(
        http,
        rate_limiter=AsyncRateLimiter(1.0, clock=clock, sleep=clock.sleep),
        cache=TTLCache(ttl_s=3600, clock=clock),
        retry=RETRY,
        sleep=clock.sleep,
    )
    forecaster = OpenMeteoClient(http, retry=RETRY, sleep=clock.sleep)
    return TripWeatherService(geocoder, forecaster, today=lambda: TODAY)


# --- happy path -------------------------------------------------------------------------


async def test_forecast_success_returns_typed_data(http, clock, respx_mock):
    geo = respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=LISBON_HIT)
    wx = respx_mock.get(OPEN_METEO_FORECAST_URL).respond(200, json=FORECAST)

    result = await make_service(http, clock).forecast(TRIP)

    assert isinstance(result, TripWeather)
    assert result.place == Place(
        name="Lisbon, Portugal", latitude=38.7077507, longitude=-9.1365919, country_code="pt"
    )
    assert [d.summary for d in result.days] == ["overcast", "slight rain"]
    assert result.days[1].precipitation_probability_pct == 70
    assert result.note is None

    geo_request = geo.calls.last.request
    assert geo_request.headers["User-Agent"] == USER_AGENT
    assert geo_request.url.params["format"] == "jsonv2"
    assert geo_request.url.params["limit"] == "1"
    wx_params = wx.calls.last.request.url.params
    assert (wx_params["start_date"], wx_params["end_date"]) == ("2026-10-03", "2026-10-04")
    assert wx_params["latitude"] == "38.7077507"


# --- retries ----------------------------------------------------------------------------


async def test_timeout_is_retried_then_succeeds(http, clock, respx_mock):
    attempts: list[float] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(clock())
        if len(attempts) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=LISBON_HIT)

    respx_mock.get(NOMINATIM_SEARCH_URL).mock(side_effect=flaky)
    respx_mock.get(OPEN_METEO_FORECAST_URL).respond(200, json=FORECAST)

    result = await make_service(http, clock).forecast(TRIP)

    assert isinstance(result, TripWeather)
    assert len(attempts) == 2
    # Backoff is 0.5s plus up to 0.5s of jitter; the rate limiter then tops the gap up
    # to 1s, so the retry also respects Nominatim's 1 request/second policy.
    backoff = clock.sleeps[0]
    assert 0.5 <= backoff <= 1.0
    assert attempts == [0.0, pytest.approx(1.0)]


async def test_server_errors_exhaust_retries_and_return_a_structured_error(http, clock, respx_mock):
    respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=LISBON_HIT)
    wx = respx_mock.get(OPEN_METEO_FORECAST_URL).respond(503)

    result = await make_service(http, clock).forecast(TRIP)

    assert wx.call_count == RETRY.max_attempts
    assert isinstance(result, ToolError)
    assert result.kind == "upstream_unavailable"
    assert "HTTP 503" in result.message


async def test_client_errors_are_not_retried(http, clock, respx_mock):
    respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=LISBON_HIT)
    wx = respx_mock.get(OPEN_METEO_FORECAST_URL).respond(
        400, json={"error": True, "reason": "Invalid date"}
    )

    result = await make_service(http, clock).forecast(TRIP)

    assert wx.call_count == 1
    assert isinstance(result, ToolError)
    assert result.kind == "upstream_unavailable"
    assert "HTTP 400" in result.message


async def test_unexpected_payload_is_an_error_not_a_retry(http, clock, respx_mock):
    respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=LISBON_HIT)
    wx = respx_mock.get(OPEN_METEO_FORECAST_URL).respond(200, json={"daily": {"time": "?"}})

    result = await make_service(http, clock).forecast(TRIP)

    assert wx.call_count == 1
    assert isinstance(result, ToolError)
    assert result.kind == "upstream_unavailable"


async def test_upstream_error_is_raised_by_the_client(http, clock, respx_mock):
    respx_mock.get(OPEN_METEO_FORECAST_URL).respond(500)
    client = OpenMeteoClient(http, retry=RetryPolicy(max_attempts=2), sleep=clock.sleep)

    with pytest.raises(UpstreamError) as excinfo:
        await client.daily_forecast(38.7, -9.1, TODAY, TODAY)

    assert excinfo.value.retryable
    assert excinfo.value.status_code == 500


# --- geocoding: cache, not found, rate limit ---------------------------------------------


async def test_geocoding_results_are_cached(http, clock, respx_mock):
    geo = respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=LISBON_HIT)
    respx_mock.get(OPEN_METEO_FORECAST_URL).respond(200, json=FORECAST)
    service = make_service(http, clock)

    await service.forecast(TRIP)
    await service.forecast(TRIP.model_copy(update={"destination": "  LISBON "}))

    assert geo.call_count == 1


async def test_unknown_place_returns_not_found(http, clock, respx_mock):
    respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=[])
    wx = respx_mock.get(OPEN_METEO_FORECAST_URL).respond(200, json=FORECAST)

    result = await make_service(http, clock).forecast(TRIP)

    assert isinstance(result, ToolError)
    assert result.kind == "not_found"
    assert wx.call_count == 0


async def test_nominatim_requests_are_spaced_by_the_rate_limiter(http, clock, respx_mock):
    limiter = AsyncRateLimiter(1.0, clock=clock, sleep=clock.sleep)
    geocoder = NominatimClient(
        http, rate_limiter=limiter, cache=TTLCache(ttl_s=60, clock=clock), sleep=clock.sleep
    )
    started_at: list[float] = []
    respx_mock.get(NOMINATIM_SEARCH_URL).side_effect = lambda request: (
        started_at.append(clock()) or httpx.Response(200, json=LISBON_HIT)
    )

    await asyncio.gather(*(geocoder.geocode(city) for city in ("Lisbon", "Porto", "Faro")))

    assert started_at == [0.0, 1.0, 2.0]


async def test_rate_limiter_only_waits_when_calls_are_too_close(clock):
    limiter = AsyncRateLimiter(1.0, clock=clock, sleep=clock.sleep)

    await limiter.acquire()
    clock.now += 5.0
    await limiter.acquire()
    await limiter.acquire()

    assert clock.sleeps == [1.0]


def test_ttl_cache_expires_and_evicts(clock):
    cache: TTLCache[str, int] = TTLCache(ttl_s=10, maxsize=2, clock=clock)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None  # evicted: oldest entry past maxsize
    assert cache.get("b") == 2

    clock.now += 10
    assert cache.get("b") is None  # expired


# --- forecast window --------------------------------------------------------------------


async def test_dates_beyond_the_forecast_window_fail_without_calling_apis(http, clock, respx_mock):
    far = TripWeatherRequest(
        destination="Lisbon", start_date=date(2026, 12, 20), end_date=date(2026, 12, 27)
    )

    result = await make_service(http, clock).forecast(far)

    assert isinstance(result, ToolError)
    assert result.kind == "out_of_range"
    assert "2026-10-13" in result.message
    assert len(respx_mock.calls) == 0


async def test_trip_partly_inside_the_window_is_clipped_with_a_note(http, clock, respx_mock):
    respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=LISBON_HIT)
    wx = respx_mock.get(OPEN_METEO_FORECAST_URL).respond(200, json=FORECAST)
    long_trip = TripWeatherRequest(
        destination="Lisbon", start_date=date(2026, 10, 10), end_date=date(2026, 10, 20)
    )

    result = await make_service(http, clock).forecast(long_trip)

    assert isinstance(result, TripWeather)
    assert wx.calls.last.request.url.params["end_date"] == "2026-10-13"
    assert result.note is not None and "2026-10-13" in result.note


# --- LangChain tool ---------------------------------------------------------------------


async def test_tool_returns_json_content_and_the_typed_result_as_artifact(http, clock, respx_mock):
    respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=LISBON_HIT)
    respx_mock.get(OPEN_METEO_FORECAST_URL).respond(200, json=FORECAST)
    tool = create_trip_weather_tool(make_service(http, clock))

    message = await tool.ainvoke(
        {
            "type": "tool_call",
            "id": "call-1",
            "name": tool.name,
            "args": {"destination": "Lisbon", "start_date": "2026-10-03", "end_date": "2026-10-04"},
        }
    )

    assert isinstance(message, ToolMessage)
    assert isinstance(message.artifact, TripWeather)
    assert TripWeather.model_validate_json(message.content) == message.artifact


def test_tool_input_rejects_end_before_start():
    with pytest.raises(ValueError, match="end_date"):
        TripWeatherRequest(
            destination="Lisbon", start_date=date(2026, 10, 5), end_date=date(2026, 10, 3)
        )


# --- degradation beyond HTTP errors -----------------------------------------------------


async def test_a_hanging_lookup_is_cut_off_by_the_deadline():
    class HangingGeocoder:
        async def geocode(self, query: str) -> Place | None:
            await asyncio.sleep(10)
            return None

    service = TripWeatherService(
        HangingGeocoder(),
        OpenMeteoClient(httpx.AsyncClient()),
        today=lambda: TODAY,
        deadline_s=0.05,
    )

    result = await service.forecast(TRIP)

    assert isinstance(result, ToolError)
    assert result.kind == "upstream_unavailable"
    assert "did not finish within 0.05 s" in result.message


async def test_invalid_coordinates_from_nominatim_are_an_upstream_error(http, clock, respx_mock):
    respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=[{**LISBON_HIT[0], "lat": "123.4"}])

    result = await make_service(http, clock).forecast(TRIP)

    assert isinstance(result, ToolError)
    assert result.kind == "upstream_unavailable"  # not blamed on the model's arguments


async def test_nan_in_the_forecast_is_an_upstream_error(http, clock, respx_mock):
    respx_mock.get(NOMINATIM_SEARCH_URL).respond(200, json=LISBON_HIT)
    daily = {**FORECAST["daily"], "precipitation_probability_max": [float("nan"), 70]}
    body = json.dumps({**FORECAST, "daily": daily})  # json.loads accepts the NaN token
    respx_mock.get(OPEN_METEO_FORECAST_URL).respond(
        200, content=body, headers={"Content-Type": "application/json"}
    )

    result = await make_service(http, clock).forecast(TRIP)

    assert isinstance(result, ToolError)
    assert result.kind == "upstream_unavailable"
