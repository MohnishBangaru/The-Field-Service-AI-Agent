"""Retry policy constants for OpenAI calls."""

from __future__ import annotations

from typing import Final

RETRY_ATTEMPTS: Final[int] = 3
RETRY_WAIT_MULTIPLIER: Final[float] = 0.5
RETRY_WAIT_MAX: Final[float] = 4.0
