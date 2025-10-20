from __future__ import annotations

import io

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .errors import TranscriptionError


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
def transcribe_wav(wav_bytes: bytes, model: str, api_key: str) -> str:
	client = OpenAI(api_key=api_key)
	# Use file-like object for in-memory bytes
	with io.BytesIO(wav_bytes) as f:
		f.name = "audio.wav"  # hint for content-type
		try:
			resp = client.audio.transcriptions.create(model=model, file=f)  # type: ignore[arg-type]
			return resp.text  # type: ignore[return-value]
		except Exception as e:
			raise TranscriptionError(str(e))


