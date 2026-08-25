"""Text preparation rules for spoken output."""

from __future__ import annotations

import re
from typing import Final

from field_service_agent.domain.constants import DEFAULT_SPEECH_SEGMENT_LIMIT


class SpeechSanitizer:
    """Strips links and markup that should not be read aloud."""

    __MARKDOWN_LINK: Final[re.Pattern[str]] = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
    __DASHED_URL: Final[re.Pattern[str]] = re.compile(r"\s+[–—-]\s+(https?://\S+)")
    __RAW_URL: Final[re.Pattern[str]] = re.compile(r"(https?://\S+|www\.\S+)")
    __WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

    def sanitize(self, *, text: str) -> str:
        """Return text with URLs removed and whitespace collapsed."""
        if not text:
            return ""
        clean = self.__MARKDOWN_LINK.sub(r"\1", text)
        clean = self.__DASHED_URL.sub("", clean)
        clean = self.__RAW_URL.sub("", clean)
        return self.__WHITESPACE.sub(" ", clean).strip()


class SpeechSegmenter:
    """Splits long text into segments small enough for one synthesis request."""

    def __init__(self, *, limit: int = DEFAULT_SPEECH_SEGMENT_LIMIT) -> None:
        if limit <= 0:
            raise ValueError("Speech segment limit must be a positive character count")
        self.__limit = limit

    def segment(self, *, text: str) -> tuple[str, ...]:
        """Split text on line boundaries so each segment stays within the limit."""
        segments: list[str] = []
        current: list[str] = []
        current_length = 0
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if current and current_length + len(line) + 1 > self.__limit:
                segments.append(" ".join(current))
                current = [line]
                current_length = len(line)
            else:
                current.append(line)
                current_length += len(line) + 1
        if current:
            segments.append(" ".join(current))
        return tuple(segments)
