from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from .errors import TtsError


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
def synthesize_wav(text: str, model: str, voice: str, api_key: str) -> bytes:
	client = OpenAI(api_key=api_key)
	# Request wav output so we can play easily
	try:
		resp = client.audio.speech.create(  # type: ignore[attr-defined]
			model=model,
			voice=voice,
			input=text,
			response_format="wav",
		)
		return resp.content  # type: ignore[return-value]
	except Exception as e:
		raise TtsError(str(e))


