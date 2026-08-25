"""Immutable value objects exchanged across layer boundaries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from field_service_agent.domain.constants import EstimateStatus


class Coordinate(BaseModel):
    """Geographic point in WGS84 degrees."""

    model_config = ConfigDict(frozen=True)

    latitude: float = Field(ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(ge=-180.0, le=180.0, description="Longitude in decimal degrees")


class Location(BaseModel):
    """Address text resolved to a coordinate."""

    model_config = ConfigDict(frozen=True)

    address: str = Field(min_length=1, description="Address text as supplied by the caller")
    coordinate: Coordinate = Field(description="Resolved geographic point")


class Place(BaseModel):
    """Point of interest returned by a place search."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Display name of the place")
    address: str = Field(default="", description="Formatted street address when known")
    url: str = Field(default="", description="Canonical link to the place")
    category: str = Field(default="", description="Primary type or category")
    rating: float | None = Field(default=None, description="Average user rating when known")
    reviews: int | None = Field(default=None, description="Number of user ratings when known")
    distance: int | None = Field(default=None, description="Distance from the search center in meters")


class SearchResult(BaseModel):
    """Single hit from a web search."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Result title")
    url: str = Field(description="Result link")


class DirectionStep(BaseModel):
    """One navigation instruction within a route leg."""

    model_config = ConfigDict(frozen=True)

    instruction: str = Field(description="Plain-text maneuver instruction")
    distance: str = Field(default="", description="Human-readable distance of the step")
    duration: str = Field(default="", description="Human-readable duration of the step")


class TravelEstimate(BaseModel):
    """Travel time and distance for one origin/destination pair."""

    model_config = ConfigDict(frozen=True)

    origin: str = Field(description="Origin address text")
    destination: str = Field(description="Destination address text")
    status: EstimateStatus = Field(description="Whether an estimate was available")
    distance: str = Field(default="", description="Human-readable distance")
    duration: str = Field(default="", description="Human-readable duration")


class RouteSummary(BaseModel):
    """Ordered stops with total distance and duration."""

    model_config = ConfigDict(frozen=True)

    stops: tuple[str, ...] = Field(description="Addresses in visiting order")
    distance: int = Field(ge=0, description="Total route length in meters")
    duration: int = Field(ge=0, description="Total travel time in seconds")


class AddressVerdict(BaseModel):
    """Result of validating a postal address."""

    model_config = ConfigDict(frozen=True)

    valid: bool = Field(description="Whether the address is deliverable as understood")
    formatted: str = Field(default="", description="Normalized address text")


class AudioClip(BaseModel):
    """Mono 16-bit PCM audio in memory."""

    model_config = ConfigDict(frozen=True)

    pcm: bytes = Field(description="Little-endian signed 16-bit mono samples")
    rate: int = Field(gt=0, description="Samples per second")

    @property
    def empty(self) -> bool:
        """True when the clip holds no samples."""
        return len(self.pcm) == 0


class Exchange(BaseModel):
    """One completed user/assistant turn."""

    model_config = ConfigDict(frozen=True)

    user: str = Field(description="What the user said")
    assistant: str = Field(description="What the assistant replied")
