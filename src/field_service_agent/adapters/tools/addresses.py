"""Normalization of address lists supplied by the language model."""

from __future__ import annotations

import re
from typing import Final


class AddressParser:
    """Turns free-form address input into a clean tuple of address strings."""

    __SEPARATOR: Final[re.Pattern[str]] = re.compile(r"[\n;]|\s{2,}|,\s*(?=[A-Za-z])")

    def parse(self, *, raw: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Split a delimited string or clean a list, dropping blanks."""
        items = self.__SEPARATOR.split(raw) if isinstance(raw, str) else list(raw)
        return tuple(item.strip() for item in items if isinstance(item, str) and item.strip())
