from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from .tools import (
	get_time,
	http_get,
	search_nearby_places,
	web_search,
	web_fetch,
	google_places_search,
	google_optimize_route,
	google_directions,
	google_distance_matrix,
	google_validate_address,
 	google_places_aggregate,
)


def build_agent(model: str, api_key: str) -> Runnable:
	llm = ChatOpenAI(model=model, api_key=api_key)
	tools = [
		get_time,
		http_get,
		search_nearby_places,
		web_search,
		web_fetch,
		google_places_search,
		google_optimize_route,
		google_directions,
		google_distance_matrix,
		google_validate_address,
		google_places_aggregate,
	]
	prompt = ChatPromptTemplate.from_messages([
		("system", "You are a concise, helpful voice assistant. Use tools when helpful. If the user request lacks required details (e.g., addresses for route optimization), ask a brief clarifying question before proceeding."),
		MessagesPlaceholder("chat_history"),
		("human", "{input}"),
		MessagesPlaceholder("agent_scratchpad"),
	])
	tool_agent = create_tool_calling_agent(llm, tools, prompt)
	return AgentExecutor(agent=tool_agent, tools=tools, verbose=False)


