"""Shared outbound HTTP client wrapper."""

from __future__ import annotations

from collections.abc import Mapping

import httpx

from field_service_agent.domain.errors import GatewayError


class HttpGateway:
    """Issues JSON and text requests with uniform timeouts, headers, and error mapping."""

    def __init__(self, *, client: httpx.Client, timeout: float, user_agent: str) -> None:
        self.__client = client
        self.__timeout = timeout
        self.__headers = {"User-Agent": user_agent}

    def get_json(
        self, *, url: str, params: Mapping[str, str | int | float] | None = None, headers: Mapping[str, str] | None = None
    ) -> dict[str, object]:
        """GET a JSON object."""
        response = self.__send(method="GET", url=url, params=params, headers=headers)
        return self.__object(response=response)

    def post_json(
        self,
        *,
        url: str,
        body: Mapping[str, object] | None = None,
        data: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        """POST a JSON or form body and parse a JSON object."""
        response = self.__send(method="POST", url=url, json=body, data=data, headers=headers)
        return self.__object(response=response)

    def get_json_array(self, *, url: str, params: Mapping[str, str | int | float] | None = None) -> list[dict[str, object]]:
        """GET a JSON array of objects."""
        response = self.__send(method="GET", url=url, params=params)
        try:
            payload = response.json()
        except ValueError as exception:
            raise GatewayError(f"Non-JSON response from {response.url}") from exception
        if not isinstance(payload, list):
            raise GatewayError(f"Expected a JSON array from {response.url}")
        return [item for item in payload if isinstance(item, dict)]

    def get_text(self, *, url: str, timeout: float | None = None) -> str:
        """GET a text body."""
        return self.__send(method="GET", url=url, timeout=timeout).text

    def __send(
        self,
        *,
        method: str,
        url: str,
        params: Mapping[str, str | int | float] | None = None,
        json: Mapping[str, object] | None = None,
        data: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        merged = {**self.__headers, **(headers or {})}
        try:
            response = self.__client.request(
                method, url, params=params, json=json, data=data, headers=merged, timeout=timeout or self.__timeout
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exception:
            raise GatewayError(f"{method} {url} returned {exception.response.status_code}: {exception.response.text[:200]}") from exception
        except httpx.HTTPError as exception:
            raise GatewayError(f"{method} {url} failed: {exception}") from exception
        return response

    @staticmethod
    def __object(*, response: httpx.Response) -> dict[str, object]:
        try:
            payload = response.json()
        except ValueError as exception:
            raise GatewayError(f"Non-JSON response from {response.url}") from exception
        if not isinstance(payload, dict):
            raise GatewayError(f"Expected a JSON object from {response.url}")
        return payload
