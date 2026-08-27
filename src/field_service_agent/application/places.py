"""Place discovery use case."""

from __future__ import annotations

from field_service_agent.application.ports import Geocoder, PlaceFinder
from field_service_agent.domain.errors import RoutingError
from field_service_agent.domain.schemas import Place


class PlaceDiscovery:
    """Finds places for a query near a free-text location."""

    def __init__(self, *, geocoder: Geocoder, finder: PlaceFinder) -> None:
        self.__geocoder = geocoder
        self.__finder = finder

    def search(self, *, query: str, near: str | None, radius: int, limit: int) -> tuple[Place, ...]:
        """Places for the query; when near is given it must geocode."""
        center = None
        if near:
            location = self.__geocoder.geocode(address=near)
            if location is None:
                raise RoutingError(f"No location found for '{near}'")
            center = location.coordinate
        return self.__finder.search(query=query, center=center, radius=radius, limit=limit)
