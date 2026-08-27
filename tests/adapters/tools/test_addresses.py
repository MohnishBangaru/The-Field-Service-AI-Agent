"""Tests for address input normalization."""

from __future__ import annotations

from field_service_agent.adapters.tools.addresses import AddressParser


class TestAddressParser:
    """AddressParser accepts the shapes a language model tends to produce."""

    def test_list_is_trimmed_and_blanks_dropped(self) -> None:
        assert AddressParser().parse(raw=[" 1 Main St ", "", "2 Oak Ave"]) == ("1 Main St", "2 Oak Ave")

    def test_newline_and_semicolon_delimited_string(self) -> None:
        assert AddressParser().parse(raw="1 Main St\n2 Oak Ave; 3 Pine Rd") == ("1 Main St", "2 Oak Ave", "3 Pine Rd")

    def test_comma_before_letter_splits_but_zip_stays(self) -> None:
        assert AddressParser().parse(raw="1 Main St, Springfield, 12345") == ("1 Main St", "Springfield, 12345")
