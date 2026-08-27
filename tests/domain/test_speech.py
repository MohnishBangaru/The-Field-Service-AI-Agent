"""Tests for spoken-text preparation."""

from __future__ import annotations

import pytest

from field_service_agent.domain.speech import SpeechSanitizer, SpeechSegmenter


class TestSpeechSanitizer:
    """SpeechSanitizer removes links the voice should not read."""

    def test_markdown_link_keeps_title(self) -> None:
        result = SpeechSanitizer().sanitize(text="See [the map](https://maps.example.com/x) for details")
        assert result == "See the map for details"

    def test_dashed_url_suffix_is_dropped(self) -> None:
        result = SpeechSanitizer().sanitize(text="Blue Bottle Coffee - https://www.openstreetmap.org/node/1")
        assert result == "Blue Bottle Coffee"

    def test_raw_urls_and_whitespace_are_collapsed(self) -> None:
        result = SpeechSanitizer().sanitize(text="Visit www.example.com   now\n\nplease")
        assert result == "Visit now please"

    def test_empty_text_returns_empty(self) -> None:
        assert SpeechSanitizer().sanitize(text="") == ""


class TestSpeechSegmenter:
    """SpeechSegmenter keeps every segment within the limit."""

    def test_lines_are_grouped_until_limit(self) -> None:
        segments = SpeechSegmenter(limit=20).segment(text="first stop\nsecond stop\nthird stop")
        assert segments == ("first stop", "second stop", "third stop")

    def test_short_lines_share_a_segment(self) -> None:
        segments = SpeechSegmenter(limit=100).segment(text="one\ntwo\n\nthree")
        assert segments == ("one two three",)

    def test_blank_text_yields_no_segments(self) -> None:
        assert SpeechSegmenter().segment(text="\n  \n") == ()

    def test_non_positive_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SpeechSegmenter(limit=0)
