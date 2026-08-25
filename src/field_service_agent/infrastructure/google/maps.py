"""Google Maps Platform gateways."""

from __future__ import annotations

import re
from datetime import UTC, timedelta
from typing import Final

from field_service_agent.application.ports import Clock
from field_service_agent.domain.constants import EstimateStatus, TravelMode
from field_service_agent.domain.errors import GatewayError
from field_service_agent.domain.schemas import (
    AddressVerdict,
    Coordinate,
    DirectionStep,
    Location,
    Place,
    RouteSummary,
    TravelEstimate,
)
from field_service_agent.infrastructure.web.http import HttpGateway

DEPARTURE_LEAD: Final[timedelta] = timedelta(minutes=5)


class GoogleGeocoder:
    """Resolves addresses with the Geocoding API."""

    __URL: Final[str] = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(self, *, http: HttpGateway, api_key: str) -> None:
        self.__http = http
        self.__api_key = api_key

    def geocode(self, *, address: str) -> Location | None:
        """Top geocode match, or None."""
        payload = self.__http.get_json(url=self.__URL, params={"address": address, "key": self.__api_key})
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None
        point = results[0]["geometry"]["location"]
        return Location(address=address, coordinate=Coordinate(latitude=float(point["lat"]), longitude=float(point["lng"])))


class GooglePlaceFinder:
    """Text search through the Places API (New)."""

    __URL: Final[str] = "https://places.googleapis.com/v1/places:searchText"
    __FIELDS: Final[str] = (
        "places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.primaryType,places.googleMapsUri"
    )

    def __init__(self, *, http: HttpGateway, api_key: str) -> None:
        self.__http = http
        self.__api_key = api_key

    def search(self, *, query: str, center: Coordinate | None, radius: int, limit: int) -> tuple[Place, ...]:
        """Places for the text query, biased toward the center when given."""
        body: dict[str, object] = {"textQuery": query}
        if center is not None:
            body["locationBias"] = {
                "circle": {"center": {"latitude": center.latitude, "longitude": center.longitude}, "radius": int(radius)}
            }
        payload = self.__http.post_json(
            url=self.__URL, body=body, headers={"X-Goog-Api-Key": self.__api_key, "X-Goog-FieldMask": self.__FIELDS}
        )
        raw = payload.get("places", [])
        places = raw[: max(1, limit)] if isinstance(raw, list) else []
        return tuple(self.__place(item=item) for item in places if isinstance(item, dict))

    @staticmethod
    def __place(*, item: dict[str, object]) -> Place:
        display = item.get("displayName")
        name = display.get("text", "(unnamed)") if isinstance(display, dict) else "(unnamed)"
        rating = item.get("rating")
        reviews = item.get("userRatingCount")
        return Place(
            name=str(name),
            address=str(item.get("formattedAddress", "")),
            url=str(item.get("googleMapsUri", "")),
            category=str(item.get("primaryType", "")),
            rating=float(rating) if isinstance(rating, (int, float)) else None,
            reviews=int(reviews) if isinstance(reviews, int) else None,
        )


class GoogleRoutePlanner:
    """Routing through the Routes, Directions, Distance Matrix, and Time Zone APIs."""

    __ROUTES_URL: Final[str] = "https://routes.googleapis.com/directions/v2:computeRoutes"
    __DIRECTIONS_URL: Final[str] = "https://maps.googleapis.com/maps/api/directions/json"
    __MATRIX_URL: Final[str] = "https://maps.googleapis.com/maps/api/distancematrix/json"
    __TIMEZONE_URL: Final[str] = "https://maps.googleapis.com/maps/api/timezone/json"
    __TRAVEL_MODES: Final[dict[TravelMode, str]] = {
        TravelMode.DRIVING: "DRIVE",
        TravelMode.WALKING: "WALK",
        TravelMode.BICYCLING: "BICYCLE",
        TravelMode.TRANSIT: "TRANSIT",
    }
    __HTML_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
    __DURATION: Final[re.Pattern[str]] = re.compile(r"^(\d+(?:\.\d+)?)s$")

    def __init__(self, *, http: HttpGateway, api_key: str, clock: Clock) -> None:
        self.__http = http
        self.__api_key = api_key
        self.__clock = clock

    def optimize_order(self, *, origin: Location, destination: Location, waypoints: tuple[Location, ...]) -> tuple[int, ...]:
        """Waypoint indexes in the order recommended by the Routes API."""
        body = {
            "origin": self.__waypoint(location=origin),
            "destination": self.__waypoint(location=destination),
            "intermediates": [self.__waypoint(location=point) for point in waypoints],
            "travelMode": self.__TRAVEL_MODES[TravelMode.DRIVING],
            "optimizeWaypointOrder": True,
        }
        payload = self.__routes(body=body, fields="routes.optimizedIntermediateWaypointIndex")
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            raise GatewayError("Routes API returned no route for waypoint optimization")
        indexes = routes[0].get("optimizedIntermediateWaypointIndex", [])
        if not isinstance(indexes, list) or not all(isinstance(index, int) for index in indexes):
            raise GatewayError("Routes API returned a malformed waypoint order")
        return tuple(indexes)

    def summarize(self, *, stops: tuple[Location, ...], mode: TravelMode) -> RouteSummary:
        """Traffic-aware totals for stops in the given order."""
        body: dict[str, object] = {
            "origin": self.__waypoint(location=stops[0]),
            "destination": self.__waypoint(location=stops[-1]),
            "travelMode": self.__TRAVEL_MODES[mode],
        }
        if len(stops) > 2:
            body["intermediates"] = [self.__waypoint(location=point) for point in stops[1:-1]]
        if mode is TravelMode.DRIVING:
            body["routingPreference"] = "TRAFFIC_AWARE_OPTIMAL"
            body["departureTime"] = self.__departure(coordinate=stops[0].coordinate)
        payload = self.__routes(body=body, fields="routes.distanceMeters,routes.duration,routes.legs")
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            raise GatewayError("Routes API returned no route for the given stops")
        route = routes[0]
        legs = route.get("legs", []) if isinstance(route.get("legs"), list) else []
        distance = route.get("distanceMeters") or sum(int(leg.get("distanceMeters", 0)) for leg in legs)
        duration = self.__seconds(value=route.get("duration")) or sum(self.__seconds(value=leg.get("duration")) for leg in legs)
        return RouteSummary(stops=tuple(stop.address for stop in stops), distance=int(distance), duration=int(duration))

    def directions(self, *, origin: str, destination: str, mode: TravelMode) -> tuple[DirectionStep, ...]:
        """Turn-by-turn steps from the Directions API."""
        payload = self.__http.get_json(
            url=self.__DIRECTIONS_URL,
            params={"origin": origin, "destination": destination, "mode": mode.value, "key": self.__api_key},
        )
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            return ()
        steps: list[DirectionStep] = []
        for leg in routes[0].get("legs", []):
            for step in leg.get("steps", []):
                steps.append(
                    DirectionStep(
                        instruction=self.__HTML_TAG.sub("", step.get("html_instructions", "")).strip(),
                        distance=step.get("distance", {}).get("text", ""),
                        duration=step.get("duration", {}).get("text", ""),
                    )
                )
        return tuple(steps)

    def estimates(self, *, origins: tuple[str, ...], destinations: tuple[str, ...], mode: TravelMode) -> tuple[TravelEstimate, ...]:
        """Pairwise estimates from the Distance Matrix API."""
        payload = self.__http.get_json(
            url=self.__MATRIX_URL,
            params={"origins": "|".join(origins), "destinations": "|".join(destinations), "mode": mode.value, "key": self.__api_key},
        )
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return ()
        estimates: list[TravelEstimate] = []
        for origin, row in zip(origins, rows, strict=False):
            for destination, element in zip(destinations, row.get("elements", []), strict=False):
                status = self.__status(value=element.get("status"))
                estimates.append(
                    TravelEstimate(
                        origin=origin,
                        destination=destination,
                        status=status,
                        distance=element.get("distance", {}).get("text", "") if status is EstimateStatus.OK else "",
                        duration=element.get("duration", {}).get("text", "") if status is EstimateStatus.OK else "",
                    )
                )
        return tuple(estimates)

    def __routes(self, *, body: dict[str, object], fields: str) -> dict[str, object]:
        return self.__http.post_json(
            url=self.__ROUTES_URL, body=body, headers={"X-Goog-Api-Key": self.__api_key, "X-Goog-FieldMask": fields}
        )

    def __departure(self, *, coordinate: Coordinate) -> str:
        """RFC3339 UTC timestamp slightly ahead of local time at the coordinate."""
        now = self.__clock.now(zone=UTC)
        payload = self.__http.get_json(
            url=self.__TIMEZONE_URL,
            params={"location": f"{coordinate.latitude},{coordinate.longitude}", "timestamp": int(now.timestamp()), "key": self.__api_key},
        )
        offset = timedelta(seconds=self.__offset(value=payload.get("rawOffset")) + self.__offset(value=payload.get("dstOffset")))
        departure = (now + offset + DEPARTURE_LEAD) - offset
        return departure.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def __offset(*, value: object) -> int:
        return int(value) if isinstance(value, int | float | str) else 0

    @staticmethod
    def __waypoint(*, location: Location) -> dict[str, object]:
        return {"location": {"latLng": {"latitude": location.coordinate.latitude, "longitude": location.coordinate.longitude}}}

    def __seconds(self, *, value: object) -> int:
        match = self.__DURATION.match(value) if isinstance(value, str) else None
        return int(float(match.group(1))) if match else 0

    @staticmethod
    def __status(*, value: object) -> EstimateStatus:
        try:
            return EstimateStatus(str(value))
        except ValueError:
            return EstimateStatus.UNKNOWN


class GoogleAddressValidator:
    """Address Validation API client."""

    __URL: Final[str] = "https://addressvalidation.googleapis.com/v1:validateAddress"

    def __init__(self, *, http: HttpGateway, api_key: str) -> None:
        self.__http = http
        self.__api_key = api_key

    def validate(self, *, address: str) -> AddressVerdict:
        """Verdict and normalized formatting for the address."""
        payload = self.__http.post_json(
            url=self.__URL, body={"address": {"addressLines": [address]}}, headers={"X-Goog-Api-Key": self.__api_key}
        )
        result = payload.get("result", {})
        result = result if isinstance(result, dict) else {}
        verdict = result.get("verdict", {}) if isinstance(result.get("verdict"), dict) else {}
        formatted = result.get("address", {}) if isinstance(result.get("address"), dict) else {}
        lines = formatted.get("formattedAddress") or " ".join(formatted.get("addressLines", []))
        valid = bool(verdict.get("addressComplete") or verdict.get("hasInferredComponents") or verdict.get("hasReplacedComponents"))
        return AddressVerdict(valid=valid, formatted=str(lines))
