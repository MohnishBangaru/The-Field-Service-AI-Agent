from __future__ import annotations

# Placeholder for future streamed STT; OpenAI Whisper API is request-based.
# This module defines an interface to plug in partial transcripts when available.

from typing import Iterator


def stream_partial_transcripts() -> Iterator[str]:
	# Implement with a streaming provider or local VAD+chunk loop in the future.
	yield from ()


