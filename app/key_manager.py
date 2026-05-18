import asyncio
import itertools
from typing import List


class KeyManager:
    """Rotates Gemini API keys automatically when a quota error occurs."""

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("API key list cannot be empty.")
        self.keys = keys
        self._key_iterator = itertools.cycle(keys)
        self._current_key = next(self._key_iterator)
        self._lock = asyncio.Lock()
        print(f"KeyManager ready with {len(self.keys)} key(s).")

    def get_current_key(self) -> str:
        return self._current_key

    async def rotate(self) -> str:
        async with self._lock:
            prev = self._current_key
            self._current_key = next(self._key_iterator)
            if prev != self._current_key:
                print(f"Quota exceeded — rotating to key ...{self._current_key[-4:]}")
            return self._current_key
