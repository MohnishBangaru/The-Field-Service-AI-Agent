"""In-memory port implementations for application tests."""

from __future__ import annotations

from field_service_agent.domain.constants import TravelMode
from field_service_agent.domain.errors import TranscriptionError
from field_service_agent.domain.schemas import (
    AudioClip,
    Coordinate,
    DirectionStep,
    Exchange,
    Location,
    Place,
    RouteSummary,
    TravelEstimate,
)


class ScriptedConsole:
    """Replays prompt answers and records everything said."""

    def __init__(self, *, answers: list[str]) -> None:
        self.__answers = list(answers)
        self.said: list[str] = []

    def say(self, *, message: str) -> None:
        self.said.append(message)

    def prompt(self, *, message: str) -> str:
        self.said.append(message)
        return self.__answers.pop(0) if self.__answers else "q"


class FixedRecorder:
    """Returns a preset clip for every recording call."""

    def __init__(self, *, clip: AudioClip) -> None:
        self.__clip = clip
        self.timed_calls: list[float] = []
        self.stopped_calls = 0

    def record_timed(self, *, duration: float, rate: int) -> AudioClip:
        self.timed_calls.append(duration)
        return self.__clip

    def record_until_stopped(self, *, rate: int) -> AudioClip:
        self.stopped_calls += 1
        return self.__clip


class CapturingPlayer:
    """Stores played clips."""

    def __init__(self) -> None:
        self.played: list[AudioClip] = []

    def play(self, *, clip: AudioClip) -> None:
        self.played.append(clip)


class FixedTranscriber:
    """Returns preset text, or raises when text is None."""

    def __init__(self, *, text: str | None) -> None:
        self.__text = text

    def transcribe(self, *, clip: AudioClip) -> str:
        if self.__text is None:
            raise TranscriptionError("service unavailable")
        return self.__text


class EchoSynthesizer:
    """Encodes the text bytes as the clip payload so tests can inspect it."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(self, *, text: str) -> AudioClip:
        self.texts.append(text)
        return AudioClip(pcm=text.encode(), rate=1)


class EchoAssistant:
    """Replies with a fixed template and records the history it saw."""

    def __init__(self, *, template: str = "You said: {message}") -> None:
        self.__template = template
        self.histories: list[tuple[Exchange, ...]] = []

    def reply(self, *, message: str, history: tuple[Exchange, ...]) -> str:
        self.histories.append(history)
        return self.__template.format(message=message)


class TableGeocoder:
    """Resolves only addresses present in its table."""

    def __init__(self, *, table: dict[str, Coordinate]) -> None:
        self.__table = table
        self.requested: list[str] = []

    def geocode(self, *, address: str) -> Location | None:
        self.requested.append(address)
        point = self.__table.get(address)
        return Location(address=address, coordinate=point) if point else None


class ReversingPlanner:
    """Recommends visiting waypoints in reverse and sums fixed per-leg costs."""

    def __init__(self, *, order: tuple[int, ...] | None = None) -> None:
        self.__order = order
        self.optimize_calls = 0

    def optimize_order(self, *, origin: Location, destination: Location, waypoints: tuple[Location, ...]) -> tuple[int, ...]:
        self.optimize_calls += 1
        return self.__order if self.__order is not None else tuple(reversed(range(len(waypoints))))

    def summarize(self, *, stops: tuple[Location, ...], mode: TravelMode) -> RouteSummary:
        legs = len(stops) - 1
        return RouteSummary(stops=tuple(stop.address for stop in stops), distance=legs * 1_000, duration=legs * 60)

    def directions(self, *, origin: str, destination: str, mode: TravelMode) -> tuple[DirectionStep, ...]:
        return ()

    def estimates(self, *, origins: tuple[str, ...], destinations: tuple[str, ...], mode: TravelMode) -> tuple[TravelEstimate, ...]:
        return ()


class TablePlaceFinder:
    """Returns preset places and records the center it was given."""

    def __init__(self, *, places: tuple[Place, ...]) -> None:
        self.__places = places
        self.centers: list[Coordinate | None] = []

    def search(self, *, query: str, center: Coordinate | None, radius: int, limit: int) -> tuple[Place, ...]:
        self.centers.append(center)
        return self.__places[:limit]
