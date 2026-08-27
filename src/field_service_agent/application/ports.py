"""Interfaces the application depends on; implemented by infrastructure and adapters."""

from __future__ import annotations

from datetime import datetime, tzinfo
from typing import Protocol

from field_service_agent.domain.constants import TravelMode
from field_service_agent.domain.schemas import (
    AddressVerdict,
    AudioClip,
    Coordinate,
    DirectionStep,
    Exchange,
    Location,
    Place,
    RouteSummary,
    SearchResult,
    TravelEstimate,
)


class Clock(Protocol):
    """Source of the current time."""

    def now(self, *, zone: tzinfo) -> datetime:
        """Current time in the given zone."""
        ...


class Console(Protocol):
    """Line-oriented user interaction channel."""

    def say(self, *, message: str) -> None:
        """Show a message to the user."""
        ...

    def prompt(self, *, message: str) -> str:
        """Show a message and wait for a line of input."""
        ...


class Recorder(Protocol):
    """Captures microphone audio."""

    def record_timed(self, *, duration: float, rate: int) -> AudioClip:
        """Record for a fixed number of seconds."""
        ...

    def record_until_stopped(self, *, rate: int) -> AudioClip:
        """Record until the user signals stop."""
        ...


class Player(Protocol):
    """Plays audio to the speaker."""

    def play(self, *, clip: AudioClip) -> None:
        """Play a clip to completion."""
        ...


class Transcriber(Protocol):
    """Speech-to-text service."""

    def transcribe(self, *, clip: AudioClip) -> str:
        """Return the spoken text in a clip."""
        ...


class Synthesizer(Protocol):
    """Text-to-speech service."""

    def synthesize(self, *, text: str) -> AudioClip:
        """Return spoken audio for text."""
        ...


class Assistant(Protocol):
    """Language model that answers a user message given prior exchanges."""

    def reply(self, *, message: str, history: tuple[Exchange, ...]) -> str:
        """Produce the assistant reply."""
        ...


class Geocoder(Protocol):
    """Resolves address text to a coordinate."""

    def geocode(self, *, address: str) -> Location | None:
        """Best matching location, or None when unresolved."""
        ...


class PlaceFinder(Protocol):
    """Searches points of interest around a center."""

    def search(self, *, query: str, center: Coordinate | None, radius: int, limit: int) -> tuple[Place, ...]:
        """Places matching the query, nearest or most relevant first."""
        ...


class RoutePlanner(Protocol):
    """Computes routes and travel estimates."""

    def optimize_order(self, *, origin: Location, destination: Location, waypoints: tuple[Location, ...]) -> tuple[int, ...]:
        """Indexes into waypoints in the recommended visiting order."""
        ...

    def summarize(self, *, stops: tuple[Location, ...], mode: TravelMode) -> RouteSummary:
        """Total distance and duration for stops visited in order."""
        ...

    def directions(self, *, origin: str, destination: str, mode: TravelMode) -> tuple[DirectionStep, ...]:
        """Turn-by-turn steps between two addresses."""
        ...

    def estimates(self, *, origins: tuple[str, ...], destinations: tuple[str, ...], mode: TravelMode) -> tuple[TravelEstimate, ...]:
        """Travel estimates for every origin/destination pair."""
        ...


class AddressValidator(Protocol):
    """Checks and normalizes postal addresses."""

    def validate(self, *, address: str) -> AddressVerdict:
        """Verdict for the address."""
        ...


class WebSearcher(Protocol):
    """General web search."""

    def search(self, *, query: str, limit: int) -> tuple[SearchResult, ...]:
        """Top results for the query."""
        ...


class PageFetcher(Protocol):
    """Retrieves readable text from a URL."""

    def fetch(self, *, url: str, limit: int) -> str:
        """Visible page text truncated to limit characters."""
        ...
