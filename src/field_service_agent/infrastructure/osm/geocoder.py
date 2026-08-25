"""OpenStreetMap Nominatim geocoding."""

from __future__ import annotations

from typing import Final

from field_service_agent.domain.schemas import Coordinate, Location
from field_service_agent.infrastructure.web.http import HttpGateway


class NominatimGeocoder:
    """Resolves addresses through the Nominatim API."""

    __URL: Final[str] = "https://nominatim.openstreetmap.org/search"

    def __init__(self, *, http: HttpGateway) -> None:
        self.__http = http

    def geocode(self, *, address: str) -> Location | None:
        """Top Nominatim match, or None."""
        matches = self.__http.get_json_array(url=self.__URL, params={"q": address, "format": "jsonv2", "limit": 1})
        if not matches:
            return None
        first = matches[0]
        return Location(address=address, coordinate=Coordinate(latitude=float(str(first["lat"])), longitude=float(str(first["lon"]))))
