"""Routing, directions, and address tools."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from field_service_agent.adapters.tools.addresses import AddressParser
from field_service_agent.adapters.tools.factory import ToolFactory
from field_service_agent.adapters.tools.presenter import ToolPresenter
from field_service_agent.application.ports import AddressValidator, RoutePlanner
from field_service_agent.application.routing import RouteOptimization
from field_service_agent.domain.constants import MINIMUM_ROUTE_STOPS, TravelMode
from field_service_agent.domain.errors import ToolError


class RoutingTools:
    """Exposes route optimization, directions, travel estimates, and address validation."""

    def __init__(
        self,
        *,
        optimization: RouteOptimization,
        planner: RoutePlanner,
        validator: AddressValidator,
        parser: AddressParser,
        presenter: ToolPresenter,
        factory: ToolFactory,
    ) -> None:
        self.__optimization = optimization
        self.__planner = planner
        self.__validator = validator
        self.__parser = parser
        self.__presenter = presenter
        self.__factory = factory

    def tools(self) -> tuple[BaseTool, ...]:
        """Tool set for this group."""
        return (
            self.__factory.build(
                name="optimize_route",
                description=(
                    "Optimize the visiting order for two or more addresses and return the ordered stops "
                    "with total distance and travel time. Optional origin and destination pin the endpoints."
                ),
                func=self.__optimize,
            ),
            self.__factory.build(
                name="get_directions",
                description="Turn-by-turn directions between two addresses.",
                func=self.__directions,
            ),
            self.__factory.build(
                name="travel_estimates",
                description="Travel distance and time between every origin and destination pair.",
                func=self.__estimates,
            ),
            self.__factory.build(
                name="validate_address",
                description="Validate a postal address and return its normalized form.",
                func=self.__validate,
            ),
        )

    def __optimize(
        self,
        addresses: list[str] | str,
        mode: str = TravelMode.DRIVING.value,
        origin: str | None = None,
        destination: str | None = None,
    ) -> str:
        """Ordered stops with totals."""
        stops = self.__parser.parse(raw=addresses)
        if len(stops) < MINIMUM_ROUTE_STOPS:
            return "Please provide at least two addresses for route optimization."
        summary = self.__optimization.plan(addresses=stops, mode=self.__mode(value=mode), origin=origin, destination=destination)
        return self.__presenter.route(summary=summary)

    def __directions(self, origin: str, destination: str, mode: str = TravelMode.DRIVING.value) -> str:
        """Step list between two addresses."""
        steps = self.__planner.directions(origin=origin, destination=destination, mode=self.__mode(value=mode))
        return self.__presenter.steps(steps=steps)

    def __estimates(self, origins: list[str], destinations: list[str], mode: str = TravelMode.DRIVING.value) -> str:
        """Pairwise travel estimates."""
        estimates = self.__planner.estimates(
            origins=self.__parser.parse(raw=origins), destinations=self.__parser.parse(raw=destinations), mode=self.__mode(value=mode)
        )
        return self.__presenter.estimates(estimates=estimates)

    def __validate(self, address: str) -> str:
        """Validity and normalized address."""
        return self.__presenter.verdict(verdict=self.__validator.validate(address=address))

    @staticmethod
    def __mode(*, value: str) -> TravelMode:
        try:
            return TravelMode(value.lower())
        except ValueError as exception:
            options = ", ".join(mode.value for mode in TravelMode)
            raise ToolError(f"Unsupported travel mode '{value}'; choose one of {options}") from exception
