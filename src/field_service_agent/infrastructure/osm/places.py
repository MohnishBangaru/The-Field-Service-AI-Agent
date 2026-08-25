"""OpenStreetMap Overpass place search."""

from __future__ import annotations

from typing import Final

from field_service_agent.domain.geometry import GreatCircle
from field_service_agent.domain.schemas import Coordinate, Place
from field_service_agent.infrastructure.web.http import HttpGateway


class OverpassPlaceFinder:
    """Finds named features around a point through the Overpass API."""

    __URL: Final[str] = "https://overpass-api.de/api/interpreter"
    __SITE: Final[str] = "https://www.openstreetmap.org"
    __TIMEOUT: Final[float] = 30.0

    def __init__(self, *, http: HttpGateway) -> None:
        self.__http = http

    def search(self, *, query: str, center: Coordinate | None, radius: int, limit: int) -> tuple[Place, ...]:
        """Nearest matching features; a center is required."""
        if center is None:
            return ()
        payload = self.__http.post_json(url=self.__URL, data={"data": self.__query(query=query, center=center, radius=radius)})
        elements = payload.get("elements", [])
        places: list[Place] = []
        for element in elements if isinstance(elements, list) else []:
            place = self.__place(element=element, center=center)
            if place is not None:
                places.append(place)
        places.sort(key=lambda place: place.distance or 0)
        return tuple(places[: max(1, limit)])

    @staticmethod
    def __query(*, query: str, center: Coordinate, radius: int) -> str:
        safe = query.replace('"', "")
        around = f"around:{int(radius)},{center.latitude},{center.longitude}"
        return (
            "[out:json][timeout:25];("
            f'node({around})["name"~"{safe}",i];way({around})["name"~"{safe}",i];rel({around})["name"~"{safe}",i];'
            f'node({around})["amenity"~"{safe}",i];node({around})["shop"~"{safe}",i];'
            ");out center;"
        )

    def __place(self, *, element: object, center: Coordinate) -> Place | None:
        if not isinstance(element, dict):
            return None
        raw_tags = element.get("tags")
        tags: dict[str, object] = raw_tags if isinstance(raw_tags, dict) else {}
        point = element if "lat" in element else element.get("center") or {}
        if not isinstance(point, dict) or "lat" not in point or "lon" not in point:
            return None
        coordinate = Coordinate(latitude=float(point["lat"]), longitude=float(point["lon"]))
        kind = str(element.get("type", "node"))
        return Place(
            name=str(tags.get("name") or tags.get("amenity") or tags.get("shop") or "(unnamed)"),
            category=str(tags.get("amenity") or tags.get("shop") or ""),
            url=f"{self.__SITE}/{kind}/{element.get('id')}",
            distance=int(GreatCircle.distance(origin=center, target=coordinate)),
        )
