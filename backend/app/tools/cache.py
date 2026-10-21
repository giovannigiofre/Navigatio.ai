import time
from collections import OrderedDict
from collections.abc import Callable, Hashable


class TTLCache[K: Hashable, V]:
    """Small in-memory cache: entries expire after `ttl_s`; the oldest is evicted past `maxsize`."""

    def __init__(
        self, ttl_s: float, maxsize: int = 1024, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl_s = ttl_s
        self.maxsize = maxsize
        self._clock = clock
        self._entries: OrderedDict[K, tuple[float, V]] = OrderedDict()

    def get(self, key: K) -> V | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._entries[key]
            return None
        return value

    def set(self, key: K, value: V) -> None:
        self._entries.pop(key, None)
        self._entries[key] = (self._clock() + self.ttl_s, value)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
