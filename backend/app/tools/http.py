"""HTTP plumbing for external APIs: explicit timeouts, bounded retries, call logging."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)

# Explicit per-phase timeouts: fail fast on connect, allow a slower response body.
DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=5.0)

Sleep = Callable[[float], Awaitable[None]]


class UpstreamError(Exception):
    """An external API call failed. `retryable` marks transient failures (5xx, timeouts)."""

    def __init__(
        self, service: str, message: str, *, status_code: int | None = None, retryable: bool
    ) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service
        self.detail = message
        self.status_code = status_code
        self.retryable = retryable


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter, retrying only 5xx responses and timeouts."""

    max_attempts: int = 3
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 4.0
    jitter_s: float = 0.5


DEFAULT_RETRY = RetryPolicy()


def create_http_client(user_agent: str) -> httpx.AsyncClient:
    """One shared client per process: connection pooling plus default timeouts."""
    return httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers={"User-Agent": user_agent})


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, UpstreamError) and exc.retryable


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, str | int | float],
    service: str,
    retry: RetryPolicy = DEFAULT_RETRY,
    before_attempt: Callable[[], Awaitable[None]] | None = None,
    sleep: Sleep = asyncio.sleep,
) -> Any:
    """GET a JSON document, retrying transient failures.

    Raises UpstreamError once the call has failed for good: a 4xx or invalid
    body immediately, a 5xx or timeout after `retry.max_attempts` attempts.
    `before_attempt` runs before every attempt, e.g. to wait for a rate limiter.
    """
    retrying = AsyncRetrying(
        stop=stop_after_attempt(retry.max_attempts),
        wait=wait_exponential_jitter(
            initial=retry.initial_backoff_s, max=retry.max_backoff_s, jitter=retry.jitter_s
        ),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        sleep=sleep,
    )
    async for attempt in retrying:
        with attempt:
            if before_attempt is not None:
                await before_attempt()
            return await _attempt_get(
                client, url, params, service, attempt.retry_state.attempt_number
            )
    raise AssertionError("unreachable: tenacity either returns or re-raises")


async def _attempt_get(
    client: httpx.AsyncClient,
    url: str,
    params: Mapping[str, str | int | float],
    service: str,
    attempt: int,
) -> Any:
    started = time.perf_counter()

    def log(outcome: str, level: int = logging.INFO, **fields: Any) -> None:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.log(
            level,
            "external call",
            extra={
                "service": service,
                "attempt": attempt,
                "outcome": outcome,
                "latency_ms": latency_ms,
                **fields,
            },
        )

    try:
        response = await client.get(url, params=params)
    except httpx.TimeoutException as exc:
        log("timeout", logging.WARNING, error=type(exc).__name__)
        raise UpstreamError(service, f"timed out ({type(exc).__name__})", retryable=True) from exc
    except httpx.HTTPError as exc:
        log("transport_error", logging.WARNING, error=type(exc).__name__)
        raise UpstreamError(
            service, f"request failed ({type(exc).__name__})", retryable=False
        ) from exc

    status = response.status_code
    if status >= 500:
        log("server_error", logging.WARNING, status_code=status)
        raise UpstreamError(service, f"HTTP {status}", status_code=status, retryable=True)
    if status >= 400:
        log("client_error", logging.WARNING, status_code=status)
        raise UpstreamError(service, f"HTTP {status}", status_code=status, retryable=False)

    try:
        payload = response.json()
    except ValueError as exc:
        log("invalid_json", logging.WARNING, status_code=status)
        raise UpstreamError(service, "response is not valid JSON", retryable=False) from exc

    log("ok", status_code=status)
    return payload
