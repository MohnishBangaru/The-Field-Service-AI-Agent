"""Formats domain results as compact text for the language model."""

from __future__ import annotations

from field_service_agent.domain.constants import METERS_PER_KILOMETER, SECONDS_PER_MINUTE, EstimateStatus
from field_service_agent.domain.schemas import (
    AddressVerdict,
    DirectionStep,
    Place,
    RouteSummary,
    SearchResult,
    TravelEstimate,
)


class ToolPresenter:
    """Renders typed results into newline-separated lines."""

    def places(self, *, places: tuple[Place, ...]) -> str:
        """One line per place with the details that are known."""
        if not places:
            return "No places found."
        return "\n".join(self.__place(place=place) for place in places)

    def search_results(self, *, results: tuple[SearchResult, ...]) -> str:
        """Title and link per result."""
        if not results:
            return "No results found."
        return "\n".join(f"{result.title} - {result.url}" for result in results)

    def route(self, *, summary: RouteSummary) -> str:
        """Visiting order followed by totals."""
        kilometers = summary.distance // METERS_PER_KILOMETER
        minutes = summary.duration // SECONDS_PER_MINUTE
        return " -> ".join(summary.stops) + f"\nTotal: {kilometers} km, {minutes} min"

    def steps(self, *, steps: tuple[DirectionStep, ...]) -> str:
        """Instruction with distance and duration per step."""
        if not steps:
            return "No steps available."
        return "\n".join(f"{step.instruction} ({step.distance}, {step.duration})" for step in steps)

    def estimates(self, *, estimates: tuple[TravelEstimate, ...]) -> str:
        """Origin to destination with distance and duration or status."""
        if not estimates:
            return "No results."
        lines: list[str] = []
        for estimate in estimates:
            detail = f"{estimate.distance}, {estimate.duration}" if estimate.status is EstimateStatus.OK else estimate.status.value
            lines.append(f"{estimate.origin} -> {estimate.destination}: {detail}")
        return "\n".join(lines)

    def verdict(self, *, verdict: AddressVerdict) -> str:
        """Validity flag and normalized address."""
        return f"Valid: {verdict.valid}\n{verdict.formatted}"

    @staticmethod
    def __place(*, place: Place) -> str:
        parts = [place.name]
        if place.rating is not None:
            parts.append(f"{place.rating}({place.reviews or 0})")
        if place.distance is not None:
            parts.append(f"{place.distance}m")
        if place.address:
            parts.append(place.address)
        if place.category:
            parts.append(place.category)
        if place.url:
            parts.append(place.url)
        return " - ".join(parts)
