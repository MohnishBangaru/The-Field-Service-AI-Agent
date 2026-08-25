"""Terminal input and output."""

from __future__ import annotations


class TerminalConsole:
    """Standard input/output console."""

    def say(self, *, message: str) -> None:
        """Print a line to standard output."""
        print(message)

    def prompt(self, *, message: str) -> str:
        """Print a line and read a line from standard input."""
        print(message)
        return input()
