"""Tests for route optimization."""

from __future__ import annotations

import pytest

from field_service_agent.application.routing import RouteOptimization
from field_service_agent.domain.constants import TravelMode
from field_service_agent.domain.errors import RoutingError
from field_service_agent.domain.schemas import Coordinate
from tests.application.fakes import ReversingPlanner, TableGeocoder

TABLE = {
    "Depot": Coordinate(latitude=37.70, longitude=-122.40),
    "Customer A": Coordinate(latitude=37.71, longitude=-122.41),
    "Customer B": Coordinate(latitude=37.72, longitude=-122.42),
    "Customer C": Coordinate(latitude=37.73, longitude=-122.43),
}


class TestRouteOptimization:
    """RouteOptimization orders stops and reports totals."""

    def test_driving_route_uses_planner_order_for_middle_stops(self) -> None:
        planner = ReversingPlanner()
        summary = RouteOptimization(geocoder=TableGeocoder(table=TABLE), planner=planner).plan(
            addresses=("Depot", "Customer A", "Customer B", "Customer C"), mode=TravelMode.DRIVING
        )
        assert summary.stops == ("Depot", "Customer B", "Customer A", "Customer C")
        assert summary.distance == 3_000
        assert summary.duration == 180

    def test_explicit_origin_and_destination_pin_endpoints(self) -> None:
        summary = RouteOptimization(geocoder=TableGeocoder(table=TABLE), planner=ReversingPlanner()).plan(
            addresses=("Customer A", "Customer B"), mode=TravelMode.DRIVING, origin="Depot", destination="Customer C"
        )
        assert summary.stops == ("Depot", "Customer B", "Customer A", "Customer C")

    def test_walking_route_keeps_supplied_order(self) -> None:
        planner = ReversingPlanner()
        summary = RouteOptimization(geocoder=TableGeocoder(table=TABLE), planner=planner).plan(
            addresses=("Depot", "Customer A", "Customer B"), mode=TravelMode.WALKING
        )
        assert summary.stops == ("Depot", "Customer A", "Customer B")
        assert planner.optimize_calls == 0

    def test_unresolvable_addresses_are_skipped(self) -> None:
        summary = RouteOptimization(geocoder=TableGeocoder(table=TABLE), planner=ReversingPlanner()).plan(
            addresses=("Depot", "Nowhere Lane", "Customer A"), mode=TravelMode.DRIVING
        )
        assert summary.stops == ("Depot", "Customer A")

    def test_fewer_than_two_resolved_stops_raises(self) -> None:
        with pytest.raises(RoutingError):
            RouteOptimization(geocoder=TableGeocoder(table=TABLE), planner=ReversingPlanner()).plan(
                addresses=("Depot", "Nowhere Lane"), mode=TravelMode.DRIVING
            )

    def test_malformed_planner_order_falls_back_to_supplied_order(self) -> None:
        planner = ReversingPlanner(order=(0, 0))
        summary = RouteOptimization(geocoder=TableGeocoder(table=TABLE), planner=planner).plan(
            addresses=("Depot", "Customer A", "Customer B", "Customer C"), mode=TravelMode.DRIVING
        )
        assert summary.stops == ("Depot", "Customer A", "Customer B", "Customer C")
