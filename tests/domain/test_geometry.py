"""Tests for geographic distance."""

from __future__ import annotations

from field_service_agent.domain.geometry import GreatCircle
from field_service_agent.domain.schemas import Coordinate


class TestGreatCircle:
    """GreatCircle.distance matches known city separations."""

    def test_same_point_is_zero(self) -> None:
        point = Coordinate(latitude=37.7749, longitude=-122.4194)
        assert GreatCircle.distance(origin=point, target=point) == 0.0

    def test_san_francisco_to_los_angeles(self) -> None:
        san_francisco = Coordinate(latitude=37.7749, longitude=-122.4194)
        los_angeles = Coordinate(latitude=34.0522, longitude=-118.2437)
        distance = GreatCircle.distance(origin=san_francisco, target=los_angeles)
        assert 558_000 < distance < 561_000
