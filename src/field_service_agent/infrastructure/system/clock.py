"""Wall-clock time source."""

from __future__ import annotations

from datetime import datetime, tzinfo


class SystemClock:
    """Reads the operating system clock."""

    def now(self, *, zone: tzinfo) -> datetime:
        """Current time in the requested zone."""
        return datetime.now(tz=zone)
