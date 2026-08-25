"""Clock tools."""

from __future__ import annotations

from datetime import UTC, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool

from field_service_agent.adapters.tools.factory import ToolFactory
from field_service_agent.application.ports import Clock
from field_service_agent.domain.errors import ToolError


class TimeTools:
    """Exposes the current time to the model."""

    def __init__(self, *, clock: Clock, factory: ToolFactory) -> None:
        self.__clock = clock
        self.__factory = factory

    def tools(self) -> tuple[BaseTool, ...]:
        """Tool set for this group."""
        return (
            self.__factory.build(
                name="current_time",
                description="Return the current time. Optionally pass an IANA timezone like 'UTC' or 'America/Los_Angeles'.",
                func=self.__current_time,
            ),
        )

    def __current_time(self, timezone_name: str | None = None) -> str:
        """ISO 8601 timestamp in the requested zone (UTC by default)."""
        zone: tzinfo = UTC
        if timezone_name:
            try:
                zone = ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError as exception:
                raise ToolError(f"Unknown timezone '{timezone_name}'; use an IANA name such as 'Europe/London'") from exception
        return self.__clock.now(zone=zone).isoformat()
