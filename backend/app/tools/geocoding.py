"""Place-name geocoding with OpenStreetMap Nominatim.

Usage policy (https://operations.osmfoundation.org/policies/nominatim/): an
identifying User-Agent, at most 1 request per second, and caching of results.
"""

import asyncio
import logging

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.tools.cache import TTLCache
from app.tools.http import DEFAULT_RETRY, RetryPolicy, Sleep, UpstreamError, get_json
from app.tools.models import Place
from app.tools.rate_limit import AsyncRateLimiter

logger = logging.getLogger(__name__)

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
SERVICE = "nominatim"


class _NominatimAddress(BaseModel):
    country_code: str | None = None


class _NominatimHit(BaseModel):
    lat: float  # Nominatim returns coordinates as strings; pydantic coerces them
    lon: float
    display_name: str
    address: _NominatimAddress | None = None


_HITS = TypeAdapter(list[_NominatimHit])


def _cache_key(query: str) -> str:
    return " ".join(query.casefold().split())


class NominatimClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        rate_limiter: AsyncRateLimiter,
        cache: TTLCache[str, Place],
        retry: RetryPolicy = DEFAULT_RETRY,
        url: str = NOMINATIM_SEARCH_URL,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._http = http
        self._rate_limiter = rate_limiter
        self._cache = cache
        self._retry = retry
        self._url = url
        self._sleep = sleep

    async def geocode(self, query: str) -> Place | None:
        """Best match for `query`, or None if Nominatim knows no such place."""
        key = _cache_key(query)
        cached = self._cache.get(key)
        if cached is not None:
            logger.info("geocode cache hit", extra={"service": SERVICE, "outcome": "cache_hit"})
            return cached

        payload = await get_json(
            self._http,
            self._url,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
                "accept-language": "en",
            },
            service=SERVICE,
            retry=self._retry,
            before_attempt=self._rate_limiter.acquire,
            sleep=self._sleep,
        )
        try:
            hits = _HITS.validate_python(payload)
            if not hits:
                return None
            hit = hits[0]
            place = Place(
                name=hit.display_name,
                latitude=hit.lat,
                longitude=hit.lon,
                country_code=hit.address.country_code if hit.address else None,
            )
        except ValidationError as exc:
            raise UpstreamError(SERVICE, "unexpected response data", retryable=False) from exc
        self._cache.set(key, place)
        return place
