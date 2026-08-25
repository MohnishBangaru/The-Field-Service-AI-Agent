"""OpenAI speech-to-text."""

from __future__ import annotations

import io

from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential

from field_service_agent.domain.errors import TranscriptionError
from field_service_agent.domain.schemas import AudioClip
from field_service_agent.infrastructure.audio.codec import WavCodec
from field_service_agent.infrastructure.openai.retry import RETRY_ATTEMPTS, RETRY_WAIT_MAX, RETRY_WAIT_MULTIPLIER


class OpenAITranscriber:
    """Transcribes clips with the OpenAI audio transcription endpoint."""

    __UPLOAD_NAME = "audio.wav"

    def __init__(self, *, client: OpenAI, model: str, codec: WavCodec) -> None:
        self.__client = client
        self.__model = model
        self.__codec = codec

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_WAIT_MULTIPLIER, max=RETRY_WAIT_MAX), reraise=True
    )
    def transcribe(self, *, clip: AudioClip) -> str:
        """Spoken text in the clip."""
        with io.BytesIO(self.__codec.encode(clip=clip)) as upload:
            upload.name = self.__UPLOAD_NAME
            try:
                response = self.__client.audio.transcriptions.create(model=self.__model, file=upload)
            except OpenAIError as exception:
                raise TranscriptionError(f"Transcription request failed: {exception}") from exception
        return response.text
