"""Tests for Google Maps response parsing using a recorded HTTP gateway."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, tzinfo

from field_service_agent.domain.constants import EstimateStatus, TravelMode
from field_service_agent.domain.schemas import Coordinate, Location
from field_service_agent.infrastructure.google.maps import GoogleGeocoder, GoogleRoutePlanner


class RecordedHttp:
    """Serves canned JSON per URL and records request bodies."""

    def __init__(self, *, responses: dict[str, dict[str, object]]) -> None:
        self.__responses = responses
        self.bodies: list[Mapping[str, object] | None] = []

    def get_json(
        self, *, url: str, params: Mapping[str, object] | None = None, headers: Mapping[str, str] | None = None
    ) -> dict[str, object]:
        return self.__responses[url]

    def post_json(
        self,
        *,
        url: str,
        body: Mapping[str, object] | None = None,
        data: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        self.bodies.append(body)
        return self.__responses[url]

    def get_json_array(self, *, url: str, params: Mapping[str, object] | None = None) -> list[dict[str, object]]:
        return []

    def get_text(self, *, url: str, timeout: float | None = None) -> str:
        return ""


class FrozenClock:
    """Always returns the same instant."""

    def now(self, *, zone: tzinfo) -> datetime:
        return datetime(2026, 8, 25, 12, 0, tzinfo=UTC).astimezone(zone)


DEPOT = Location(address="Depot", coordinate=Coordinate(latitude=37.70, longitude=-122.40))
CUSTOMER = Location(address="Customer", coordinate=Coordinate(latitude=37.71, longitude=-122.41))
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
TIMEZONE_URL = "https://maps.googleapis.com/maps/api/timezone/json"


class TestGoogleGeocoder:
    """GoogleGeocoder maps the first result to a Location."""

    def test_first_result_is_used(self) -> None:
        http = RecordedHttp(responses={GEOCODE_URL: {"results": [{"geometry": {"location": {"lat": 37.7, "lng": -122.4}}}]}})
        location = GoogleGeocoder(http=http, api_key="k").geocode(address="Depot")  # type: ignore[arg-type]
        assert location == DEPOT

    def test_no_results_returns_none(self) -> None:
        http = RecordedHttp(responses={GEOCODE_URL: {"results": []}})
        assert GoogleGeocoder(http=http, api_key="k").geocode(address="Nowhere") is None  # type: ignore[arg-type]


class TestGoogleRoutePlanner:
    """GoogleRoutePlanner parses Routes and Distance Matrix payloads."""

    def test_summarize_driving_adds_traffic_and_departure(self) -> None:
        http = RecordedHttp(
            responses={
                TIMEZONE_URL: {"rawOffset": -28800, "dstOffset": 3600},
                ROUTES_URL: {"routes": [{"distanceMeters": 4200, "duration": "615s", "legs": []}]},
            }
        )
        planner = GoogleRoutePlanner(http=http, api_key="k", clock=FrozenClock())  # type: ignore[arg-type]
        summary = planner.summarize(stops=(DEPOT, CUSTOMER), mode=TravelMode.DRIVING)
        assert summary.stops == ("Depot", "Customer")
        assert summary.distance == 4200
        assert summary.duration == 615
        body = http.bodies[0]
        assert body is not None
        assert body["routingPreference"] == "TRAFFIC_AWARE_OPTIMAL"
        assert body["departureTime"] == "2026-08-25T12:05:00Z"

    def test_summarize_sums_legs_when_totals_missing(self) -> None:
        http = RecordedHttp(
            responses={
                ROUTES_URL: {"routes": [{"legs": [{"distanceMeters": 100, "duration": "10s"}, {"distanceMeters": 200, "duration": "20s"}]}]}
            }
        )
        planner = GoogleRoutePlanner(http=http, api_key="k", clock=FrozenClock())  # type: ignore[arg-type]
        summary = planner.summarize(stops=(DEPOT, CUSTOMER), mode=TravelMode.WALKING)
        assert (summary.distance, summary.duration) == (300, 30)

    def test_optimize_order_returns_indexes(self) -> None:
        http = RecordedHttp(responses={ROUTES_URL: {"routes": [{"optimizedIntermediateWaypointIndex": [1, 0]}]}})
        planner = GoogleRoutePlanner(http=http, api_key="k", clock=FrozenClock())  # type: ignore[arg-type]
        assert planner.optimize_order(origin=DEPOT, destination=CUSTOMER, waypoints=(DEPOT, CUSTOMER)) == (1, 0)

    def test_estimates_map_statuses(self) -> None:
        http = RecordedHttp(
            responses={
                MATRIX_URL: {
                    "rows": [
                        {
                            "elements": [
                                {"status": "OK", "distance": {"text": "5 km"}, "duration": {"text": "9 mins"}},
                                {"status": "ZERO_RESULTS"},
                            ]
                        }
                    ]
                }
            }
        )
        planner = GoogleRoutePlanner(http=http, api_key="k", clock=FrozenClock())  # type: ignore[arg-type]
        estimates = planner.estimates(origins=("A",), destinations=("B", "C"), mode=TravelMode.DRIVING)
        assert estimates[0].status is EstimateStatus.OK
        assert estimates[0].distance == "5 km"
        assert estimates[1].status is EstimateStatus.ZERO_RESULTS
