"""OpenAI text-to-speech."""

from __future__ import annotations

from typing import Final, Literal

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

from field_service_agent.domain.errors import SynthesisError
from field_service_agent.domain.schemas import AudioClip
from field_service_agent.infrastructure.audio.codec import WavCodec
from field_service_agent.infrastructure.openai.retry import RETRY_ATTEMPTS, RETRY_WAIT_MAX, RETRY_WAIT_MULTIPLIER


class OpenAISynthesizer:
    """Synthesizes speech with the OpenAI audio speech endpoint."""

    __FORMAT: Final[Literal["wav"]] = "wav"

    def __init__(self, *, client: OpenAI, model: str, voice: str, codec: WavCodec) -> None:
        self.__client = client
        self.__model = model
        self.__voice = voice
        self.__codec = codec

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX), reraise=True
    )
    def synthesize(self, *, text: str) -> AudioClip:
        """Spoken audio for text, streamed into memory."""
        try:
            with self.__client.audio.speech.with_streaming_response.create(
                model=self.__model, voice=self.__voice, input=text, response_format=self.__FORMAT
            ) as stream:
                data = b"".join(stream.iter_bytes())
        except OpenAIError as exception:
            raise SynthesisError(f"Speech request failed: {exception}") from exception
        return self.__codec.decode(data=data)
