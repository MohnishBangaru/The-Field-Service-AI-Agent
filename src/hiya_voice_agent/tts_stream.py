from __future__ import annotations

from typing import Iterator
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .errors import TtsError


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
def stream_tts_wav(text: str, model: str, voice: str, api_key: str) -> Iterator[bytes]:
	client = OpenAI(api_key=api_key)
	# Stream chunks (SDK may buffer; this yields once when available)
	try:
		resp = client.audio.speech.with_streaming_response.create(  # type: ignore[attr-defined]
			model=model,
			voice=voice,
			input=text,
			response_format="wav",
		)
		with resp as stream:
			for chunk in stream.iter_bytes():
				yield chunk
	except Exception as e:
		raise TtsError(str(e))


def split_text_for_tts(text: str, max_chars: int = 800) -> list[str]:
	parts: list[str] = []
	current = []
	current_len = 0
	# Prefer newline boundaries first
	for line in (text or "").splitlines():
		line = line.strip()
		if not line:
			continue
		if current_len + len(line) + 1 > max_chars and current:
			parts.append(" ".join(current))
			current = [line]
			current_len = len(line)
		else:
			current.append(line)
			current_len += len(line) + 1
	if current:
		parts.append(" ".join(current))
	# Fallback if everything was empty
	if not parts:
		return ["I couldn't generate any results to speak."]
	return parts


def sanitize_for_speech(text: str) -> str:
	"""Remove URLs and link targets so TTS won't read them out loud.

	- Converts markdown links [title](url) -> title
	- Removes raw URLs (http/https/www)
	- Removes trailing em-dash url segments
	"""
	if not text:
		return ""
	clean = text
	# [title](http://link) -> title
	clean = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1", clean)
	# Remove ' — http://link' and ' - http://link'
	clean = re.sub(r"\s+[–—-]\s+(https?://\S+)", "", clean)
	# Remove raw URLs
	clean = re.sub(r"(https?://\S+|www\.\S+)", "", clean)
	# Collapse extra whitespace
	clean = re.sub(r"\s+", " ", clean).strip()
	return clean


