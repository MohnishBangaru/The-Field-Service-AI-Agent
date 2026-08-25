"""Tests for WAV encoding."""

from __future__ import annotations

import pytest

from field_service_agent.domain.errors import AudioError
from field_service_agent.domain.schemas import AudioClip
from field_service_agent.infrastructure.audio.codec import WavCodec


class TestWavCodec:
    """WavCodec round-trips mono 16-bit PCM."""

    def test_encode_then_decode_preserves_samples_and_rate(self) -> None:
        clip = AudioClip(pcm=bytes(range(0, 200, 2)), rate=8_000)
        decoded = WavCodec().decode(data=WavCodec().encode(clip=clip))
        assert decoded == clip

    def test_garbage_bytes_raise_audio_error(self) -> None:
        with pytest.raises(AudioError):
            WavCodec().decode(data=b"not a wav file")
