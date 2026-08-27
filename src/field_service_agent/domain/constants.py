"""Domain-wide enumerations and named constants."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class RecordingMode(StrEnum):
    """How microphone capture is started and stopped."""

    PUSH_TO_TALK = "ptt"
    TIMED = "timed"


class TravelMode(StrEnum):
    """Means of travel used for routing requests."""

    DRIVING = "driving"
    WALKING = "walking"
    BICYCLING = "bicycling"
    TRANSIT = "transit"


class EstimateStatus(StrEnum):
    """Outcome of a single origin/destination travel estimate."""

    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    ZERO_RESULTS = "ZERO_RESULTS"
    UNKNOWN = "UNKNOWN"


EARTH_RADIUS_METERS: Final[float] = 6_371_000.0
MONO_CHANNELS: Final[int] = 1
PCM_SAMPLE_WIDTH_BYTES: Final[int] = 2
METERS_PER_KILOMETER: Final[int] = 1_000
SECONDS_PER_MINUTE: Final[int] = 60
DEFAULT_SPEECH_SEGMENT_LIMIT: Final[int] = 800
MINIMUM_ROUTE_STOPS: Final[int] = 2
