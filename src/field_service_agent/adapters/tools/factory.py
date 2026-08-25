"""Wraps bound methods as LangChain tools with uniform error handling."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import ParamSpec

from langchain_core.tools import BaseTool, StructuredTool, ToolException

from field_service_agent.domain.errors import FieldServiceAgentError

Parameters = ParamSpec("Parameters")


class ToolFactory:
    """Builds StructuredTool instances that report domain errors back to the model."""

    def __init__(self, *, logger: logging.Logger) -> None:
        self.__logger = logger

    def build(self, *, name: str, description: str, func: Callable[Parameters, str]) -> BaseTool:
        """Tool that converts FieldServiceAgentError into a model-visible ToolException."""

        @functools.wraps(func)
        def guarded(*args: Parameters.args, **kwargs: Parameters.kwargs) -> str:
            try:
                return func(*args, **kwargs)
            except FieldServiceAgentError as exception:
                self.__logger.warning("tool_failed", extra={"tool": name, "error": str(exception)})
                raise ToolException(f"{name} failed: {exception}") from exception

        return StructuredTool.from_function(func=guarded, name=name, description=description, handle_tool_error=True)
