"""Pure geographic computations."""

from __future__ import annotations

import math

from field_service_agent.domain.constants import EARTH_RADIUS_METERS
from field_service_agent.domain.schemas import Coordinate


class GreatCircle:
    """Computes surface distances on a spherical Earth."""

    @staticmethod
    def distance(*, origin: Coordinate, target: Coordinate) -> float:
        """Haversine distance between two coordinates in meters."""
        origin_latitude = math.radians(origin.latitude)
        target_latitude = math.radians(target.latitude)
        delta_latitude = math.radians(target.latitude - origin.latitude)
        delta_longitude = math.radians(target.longitude - origin.longitude)
        chord = (
            math.sin(delta_latitude / 2) ** 2 + math.cos(origin_latitude) * math.cos(target_latitude) * math.sin(delta_longitude / 2) ** 2
        )
        arc = 2 * math.atan2(math.sqrt(chord), math.sqrt(1 - chord))
        return EARTH_RADIUS_METERS * arc
