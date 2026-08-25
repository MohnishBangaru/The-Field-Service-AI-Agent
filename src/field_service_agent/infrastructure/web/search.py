"""Web search and page retrieval."""

from __future__ import annotations

from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from field_service_agent.domain.errors import GatewayError
from field_service_agent.domain.schemas import SearchResult
from field_service_agent.infrastructure.web.http import HttpGateway


class DuckDuckGoSearcher:
    """Searches the web through DuckDuckGo."""

    def search(self, *, query: str, limit: int) -> tuple[SearchResult, ...]:
        """Top results with a link."""
        try:
            with DDGS() as engine:
                hits = engine.text(query, max_results=limit)
        except DDGSException as exception:
            raise GatewayError(f"Web search failed: {exception}") from exception
        results: list[SearchResult] = []
        for hit in hits or ():
            link = str(hit.get("href") or hit.get("link") or "")
            if link:
                results.append(SearchResult(title=str(hit.get("title", "")), url=link))
        return tuple(results)


class HtmlPageFetcher:
    """Downloads a page and extracts its visible text."""

    __NOISE_TAGS = ("script", "style", "noscript")

    def __init__(self, *, http: HttpGateway) -> None:
        self.__http = http

    def fetch(self, *, url: str, limit: int) -> str:
        """Visible text truncated to limit characters."""
        html = self.__http.get_text(url=url)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(self.__NOISE_TAGS):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())[:limit]
