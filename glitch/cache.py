"""Small thread-safe TTL + LRU cache for model results.

Keys are built from SHA-256 fingerprints, so cached text is never held in
plain form beyond the values a feature chooses to store.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict

_registry: list["TTLCache"] = []


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TTLCache:
    def __init__(self, size: int, ttl_s: float):
        self._size = size
        self._ttl = ttl_s
        self._data: OrderedDict[tuple, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()
        _registry.append(self)

    def get(self, key: tuple):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if time.monotonic() - entry[0] > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return entry[1]

    def put(self, key: tuple, value) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def clear_all() -> None:
    for cache in _registry:
        cache.clear()
