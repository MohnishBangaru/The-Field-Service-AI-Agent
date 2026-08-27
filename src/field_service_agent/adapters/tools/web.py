"""Web search and page reading tools."""

from __future__ import annotations

from typing import Final

from langchain_core.tools import BaseTool

from field_service_agent.adapters.tools.factory import ToolFactory
from field_service_agent.adapters.tools.presenter import ToolPresenter
from field_service_agent.application.ports import PageFetcher, WebSearcher

DEFAULT_RESULT_LIMIT: Final[int] = 5
DEFAULT_PAGE_LIMIT: Final[int] = 5_000


class WebTools:
    """Exposes web search and page retrieval to the model."""

    def __init__(self, *, searcher: WebSearcher, fetcher: PageFetcher, presenter: ToolPresenter, factory: ToolFactory) -> None:
        self.__searcher = searcher
        self.__fetcher = fetcher
        self.__presenter = presenter
        self.__factory = factory

    def tools(self) -> tuple[BaseTool, ...]:
        """Tool set for this group."""
        return (
            self.__factory.build(
                name="web_search",
                description="Search the web and return top result titles and links.",
                func=self.__search,
            ),
            self.__factory.build(
                name="fetch_page",
                description="Fetch a web page and return its visible text, truncated.",
                func=self.__fetch,
            ),
        )

    def __search(self, query: str, limit: int = DEFAULT_RESULT_LIMIT) -> str:
        """Top results for the query."""
        return self.__presenter.search_results(results=self.__searcher.search(query=query, limit=limit))

    def __fetch(self, url: str, limit: int = DEFAULT_PAGE_LIMIT) -> str:
        """Visible text of the page."""
        return self.__fetcher.fetch(url=url, limit=limit)
