"""Tests for place discovery."""

from __future__ import annotations

import pytest

from field_service_agent.application.places import PlaceDiscovery
from field_service_agent.domain.errors import RoutingError
from field_service_agent.domain.schemas import Coordinate, Place
from tests.application.fakes import TableGeocoder, TablePlaceFinder

CENTER = Coordinate(latitude=37.76, longitude=-122.42)
PLACES = (Place(name="Ace Hardware", distance=120), Place(name="Home Depot", distance=900))


class TestPlaceDiscovery:
    """PlaceDiscovery geocodes the center before searching."""

    def test_search_passes_geocoded_center(self) -> None:
        finder = TablePlaceFinder(places=PLACES)
        discovery = PlaceDiscovery(geocoder=TableGeocoder(table={"Mission District": CENTER}), finder=finder)
        result = discovery.search(query="hardware", near="Mission District", radius=1_000, limit=10)
        assert result == PLACES
        assert finder.centers == [CENTER]

    def test_search_without_near_uses_no_center(self) -> None:
        finder = TablePlaceFinder(places=PLACES)
        PlaceDiscovery(geocoder=TableGeocoder(table={}), finder=finder).search(query="hardware", near=None, radius=1_000, limit=1)
        assert finder.centers == [None]

    def test_unresolvable_near_raises(self) -> None:
        discovery = PlaceDiscovery(geocoder=TableGeocoder(table={}), finder=TablePlaceFinder(places=PLACES))
        with pytest.raises(RoutingError):
            discovery.search(query="hardware", near="Atlantis", radius=1_000, limit=10)
