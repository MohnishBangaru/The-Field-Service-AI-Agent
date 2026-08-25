"""Tests for tool output formatting."""

from __future__ import annotations

from field_service_agent.adapters.tools.presenter import ToolPresenter
from field_service_agent.domain.constants import EstimateStatus
from field_service_agent.domain.schemas import AddressVerdict, Place, RouteSummary, TravelEstimate


class TestToolPresenter:
    """ToolPresenter renders compact, model-readable lines."""

    def test_route_shows_order_and_rounded_totals(self) -> None:
        text = ToolPresenter().route(summary=RouteSummary(stops=("Depot", "Customer A"), distance=12_600, duration=1_530))
        assert text == "Depot -> Customer A\nTotal: 12 km, 25 min"

    def test_places_include_only_known_fields(self) -> None:
        places = (Place(name="Ace Hardware", distance=120, url="https://osm.org/node/1"), Place(name="Cafe", rating=4.5, reviews=20))
        assert ToolPresenter().places(places=places) == "Ace Hardware - 120m - https://osm.org/node/1\nCafe - 4.5(20)"

    def test_empty_places_message(self) -> None:
        assert ToolPresenter().places(places=()) == "No places found."

    def test_estimates_show_status_when_not_ok(self) -> None:
        estimates = (
            TravelEstimate(origin="A", destination="B", status=EstimateStatus.OK, distance="5 km", duration="10 mins"),
            TravelEstimate(origin="A", destination="C", status=EstimateStatus.ZERO_RESULTS),
        )
        assert ToolPresenter().estimates(estimates=estimates) == "A -> B: 5 km, 10 mins\nA -> C: ZERO_RESULTS"

    def test_verdict(self) -> None:
        assert ToolPresenter().verdict(verdict=AddressVerdict(valid=True, formatted="1 Main St")) == "Valid: True\n1 Main St"
