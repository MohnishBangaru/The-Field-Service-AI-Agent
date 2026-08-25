"""Multi-stop route planning use case."""

from __future__ import annotations

from field_service_agent.application.ports import Geocoder, RoutePlanner
from field_service_agent.domain.constants import MINIMUM_ROUTE_STOPS, TravelMode
from field_service_agent.domain.errors import RoutingError
from field_service_agent.domain.schemas import Location, RouteSummary


class RouteOptimization:
    """Geocodes stops, orders them efficiently, and summarizes the resulting route."""

    def __init__(self, *, geocoder: Geocoder, planner: RoutePlanner) -> None:
        self.__geocoder = geocoder
        self.__planner = planner

    def plan(
        self,
        *,
        addresses: tuple[str, ...],
        mode: TravelMode,
        origin: str | None = None,
        destination: str | None = None,
    ) -> RouteSummary:
        """Return the best visiting order with totals; raises RoutingError when impossible."""
        start = self.__geocoder.geocode(address=origin) if origin else None
        end = self.__geocoder.geocode(address=destination) if destination else None
        resolved = tuple(self.__resolve(addresses=addresses))
        total = len(resolved) + (1 if start else 0) + (1 if end else 0)
        if total < MINIMUM_ROUTE_STOPS:
            raise RoutingError("Could not geocode at least two stops; refine the addresses and retry")
        first, middle, last = self.__split(resolved=resolved, start=start, end=end)
        ordered = self.__order(first=first, middle=middle, last=last, mode=mode)
        return self.__planner.summarize(stops=ordered, mode=mode)

    def __resolve(self, *, addresses: tuple[str, ...]) -> list[Location]:
        located: list[Location] = []
        for address in addresses:
            location = self.__geocoder.geocode(address=address)
            if location is not None:
                located.append(location)
        return located

    @staticmethod
    def __split(
        *, resolved: tuple[Location, ...], start: Location | None, end: Location | None
    ) -> tuple[Location, tuple[Location, ...], Location]:
        remaining = list(resolved)
        first = start if start is not None else remaining.pop(0)
        last = end if end is not None else remaining.pop(-1)
        return first, tuple(remaining), last

    def __order(self, *, first: Location, middle: tuple[Location, ...], last: Location, mode: TravelMode) -> tuple[Location, ...]:
        if mode is not TravelMode.DRIVING or not middle:
            return (first, *middle, last)
        indexes = self.__planner.optimize_order(origin=first, destination=last, waypoints=middle)
        if sorted(indexes) != list(range(len(middle))):
            return (first, *middle, last)
        return (first, *(middle[index] for index in indexes), last)
