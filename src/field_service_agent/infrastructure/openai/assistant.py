"""LangChain tool-calling assistant backed by an OpenAI chat model."""

from __future__ import annotations

from typing import Any, Final

from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from field_service_agent.domain.errors import AssistantError
from field_service_agent.domain.schemas import Exchange


class LangChainAssistant:
    """Answers messages using a chat model that may call the supplied tools."""

    __SYSTEM_PROMPT: Final[str] = (
        "You are a concise, helpful voice assistant for field service technicians. "
        "Use tools when helpful. If the user request lacks required details "
        "(for example addresses for route optimization), ask a brief clarifying question before proceeding."
    )

    def __init__(self, *, model: ChatOpenAI, tools: tuple[BaseTool, ...]) -> None:
        self.__agent = create_agent(model=model, tools=list(tools), system_prompt=self.__SYSTEM_PROMPT)

    def reply(self, *, message: str, history: tuple[Exchange, ...]) -> str:
        """Assistant reply text."""
        messages: list[AnyMessage | dict[str, Any]] = [*self.__messages(history=history), HumanMessage(content=message)]
        try:
            result = self.__agent.invoke(InputAgentState(messages=messages))
        except Exception as exception:  # LangChain surfaces heterogeneous provider errors
            raise AssistantError(f"Assistant invocation failed: {exception}") from exception
        final = result["messages"][-1] if result.get("messages") else None
        if not isinstance(final, AIMessage):
            raise AssistantError("Assistant returned no reply message")
        text = self.__text(message=final)
        if not text:
            raise AssistantError("Assistant returned an empty reply")
        return text

    @staticmethod
    def __messages(*, history: tuple[Exchange, ...]) -> list[AnyMessage]:
        messages: list[AnyMessage] = []
        for exchange in history:
            messages.append(HumanMessage(content=exchange.user))
            messages.append(AIMessage(content=exchange.assistant))
        return messages

    @staticmethod
    def __text(*, message: AIMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content.strip()
        parts = [block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)]
        return " ".join(parts).strip()
