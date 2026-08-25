"""Place search tools."""

from __future__ import annotations

from typing import Final

from langchain_core.tools import BaseTool

from field_service_agent.adapters.tools.factory import ToolFactory
from field_service_agent.adapters.tools.presenter import ToolPresenter
from field_service_agent.application.places import PlaceDiscovery

DEFAULT_RADIUS: Final[int] = 1_000
DEFAULT_LIMIT: Final[int] = 10


class PlaceTools:
    """Exposes open-data and Google place discovery to the model."""

    def __init__(
        self,
        *,
        open_discovery: PlaceDiscovery,
        google_discovery: PlaceDiscovery | None,
        presenter: ToolPresenter,
        factory: ToolFactory,
    ) -> None:
        self.__open = open_discovery
        self.__google = google_discovery
        self.__presenter = presenter
        self.__factory = factory

    def tools(self) -> tuple[BaseTool, ...]:
        """Tool set; the Google tool is present only when configured."""
        tools = [
            self.__factory.build(
                name="search_nearby_places",
                description=(
                    "Search OpenStreetMap for places matching a query near a location string "
                    "(for example 'coffee' near 'Mission District, San Francisco'). Returns name, distance in meters, and link."
                ),
                func=self.__search_open,
            )
        ]
        if self.__google is not None:
            tools.append(
                self.__factory.build(
                    name="google_places_search",
                    description=(
                        "Search Google Places for a text query, optionally biased near a location string. "
                        "Returns name, rating, address, category, and link."
                    ),
                    func=self.__search_google,
                )
            )
        return tuple(tools)

    def __search_open(self, query: str, near: str, radius: int = DEFAULT_RADIUS, limit: int = DEFAULT_LIMIT) -> str:
        """Nearest OpenStreetMap matches."""
        return self.__presenter.places(places=self.__open.search(query=query, near=near, radius=radius, limit=limit))

    def __search_google(self, query: str, near: str | None = None, radius: int = DEFAULT_RADIUS, limit: int = DEFAULT_LIMIT) -> str:
        """Google Places matches."""
        if self.__google is None:
            return "Google Places is not configured."
        return self.__presenter.places(places=self.__google.search(query=query, near=near, radius=radius, limit=limit))
