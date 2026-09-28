import asyncio
import time
from collections.abc import Awaitable, Callable


class AsyncRateLimiter:
    """Spaces calls at least `min_interval_s` apart. Waiters are served in FIFO order.

    State is per process: with several worker processes, each gets its own budget.
    """

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._next_allowed = float("-inf")

    async def acquire(self) -> None:
        async with self._lock:
            delay = self._next_allowed - self._clock()
            if delay > 0:
                await self._sleep(delay)
            self._next_allowed = max(self._clock(), self._next_allowed) + self.min_interval_s
