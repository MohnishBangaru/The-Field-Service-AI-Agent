"""Tests for boundary schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from field_service_agent.domain.schemas import AudioClip, Coordinate, RouteSummary


class TestCoordinate:
    """Coordinate rejects out-of-range degrees and is immutable."""

    def test_latitude_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Coordinate(latitude=91.0, longitude=0.0)

    def test_instances_are_frozen(self) -> None:
        point = Coordinate(latitude=1.0, longitude=2.0)
        with pytest.raises(ValidationError):
            point.latitude = 3.0  # type: ignore[misc]


class TestAudioClip:
    """AudioClip reports emptiness."""

    def test_empty_when_no_samples(self) -> None:
        assert AudioClip(pcm=b"", rate=16_000).empty

    def test_not_empty_with_samples(self) -> None:
        assert not AudioClip(pcm=b"\x00\x01", rate=16_000).empty


class TestRouteSummary:
    """RouteSummary rejects negative totals."""

    def test_negative_distance_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RouteSummary(stops=("a", "b"), distance=-1, duration=0)
