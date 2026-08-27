"""Microphone capture and speaker playback via sounddevice."""

from __future__ import annotations

import numpy as np
import sounddevice

from field_service_agent.application.ports import Console
from field_service_agent.domain.constants import MONO_CHANNELS
from field_service_agent.domain.errors import AudioError
from field_service_agent.domain.schemas import AudioClip


class SoundDeviceRecorder:
    """Captures mono 16-bit PCM from the default input device."""

    def __init__(self, *, console: Console) -> None:
        self.__console = console

    def record_timed(self, *, duration: float, rate: int) -> AudioClip:
        """Record a fixed number of seconds."""
        try:
            samples = sounddevice.rec(int(duration * rate), samplerate=rate, channels=MONO_CHANNELS, dtype=np.int16)
            sounddevice.wait()
        except sounddevice.PortAudioError as exception:
            raise AudioError(f"Microphone capture failed: {exception}") from exception
        return AudioClip(pcm=samples.reshape(-1).tobytes(), rate=rate)

    def record_until_stopped(self, *, rate: int) -> AudioClip:
        """Record until the console receives an empty line."""
        frames: list[np.ndarray] = []

        def collect(indata: np.ndarray, frame_count: int, timing: object, status: object) -> None:
            frames.append(indata.copy())

        try:
            with sounddevice.InputStream(samplerate=rate, channels=MONO_CHANNELS, dtype=np.int16, callback=collect):
                self.__console.prompt(message="")
        except sounddevice.PortAudioError as exception:
            raise AudioError(f"Microphone capture failed: {exception}") from exception
        if not frames:
            return AudioClip(pcm=b"", rate=rate)
        return AudioClip(pcm=np.concatenate(frames, axis=0).reshape(-1).tobytes(), rate=rate)


class SoundDevicePlayer:
    """Plays clips through the default output device."""

    def play(self, *, clip: AudioClip) -> None:
        """Block until playback finishes; Ctrl+C stops playback early."""
        samples = np.frombuffer(clip.pcm, dtype=np.int16)
        try:
            sounddevice.play(samples, samplerate=clip.rate)
            try:
                sounddevice.wait()
            except KeyboardInterrupt:
                sounddevice.stop()
        except sounddevice.PortAudioError as exception:
            raise AudioError(f"Playback failed: {exception}") from exception
