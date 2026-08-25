"""WAV container encoding and decoding."""

from __future__ import annotations

import io
import wave

from field_service_agent.domain.constants import MONO_CHANNELS, PCM_SAMPLE_WIDTH_BYTES
from field_service_agent.domain.errors import AudioError
from field_service_agent.domain.schemas import AudioClip


class WavCodec:
    """Converts between AudioClip and WAV bytes."""

    def encode(self, *, clip: AudioClip) -> bytes:
        """WAV file bytes for a clip."""
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as writer:
                writer.setnchannels(MONO_CHANNELS)
                writer.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
                writer.setframerate(clip.rate)
                writer.writeframes(clip.pcm)
            return buffer.getvalue()

    def decode(self, *, data: bytes) -> AudioClip:
        """Clip parsed from WAV bytes; raises AudioError for unsupported layouts."""
        try:
            with io.BytesIO(data) as buffer, wave.open(buffer, "rb") as reader:
                if reader.getnchannels() != MONO_CHANNELS or reader.getsampwidth() != PCM_SAMPLE_WIDTH_BYTES:
                    raise AudioError("Only mono 16-bit WAV audio is supported")
                return AudioClip(pcm=reader.readframes(reader.getnframes()), rate=reader.getframerate())
        except wave.Error as exception:
            raise AudioError(f"Invalid WAV payload: {exception}") from exception
