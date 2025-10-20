from __future__ import annotations

import io
import wave
from typing import Tuple

import numpy as np
import sounddevice as sd
from .errors import AudioError
from typing import List


def record_pcm(duration_seconds: float, sample_rate: int) -> np.ndarray:
	try:
		frames = int(duration_seconds * sample_rate)
		data = sd.rec(frames, samplerate=sample_rate, channels=1, dtype=np.int16)
		sd.wait()
		return data.reshape(-1)
	except Exception as e:
		raise AudioError(str(e))


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
	with io.BytesIO() as buffer:
		with wave.open(buffer, "wb") as wf:
			wf.setnchannels(1)
			wf.setsampwidth(2)  # int16
			wf.setframerate(sample_rate)
			wf.writeframes(pcm.tobytes())
		return buffer.getvalue()


def wav_bytes_to_pcm(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
	with io.BytesIO(wav_bytes) as buffer:
		with wave.open(buffer, "rb") as wf:
			sample_rate = wf.getframerate()
			frames = wf.readframes(wf.getnframes())
	pcm = np.frombuffer(frames, dtype=np.int16)
	return pcm, sample_rate


def play_wav_bytes(wav_bytes: bytes) -> None:
	try:
		pcm, sample_rate = wav_bytes_to_pcm(wav_bytes)
		sd.play(pcm, samplerate=sample_rate)
		try:
			sd.wait()
		except KeyboardInterrupt:
			# Stop playback gracefully on Ctrl+C
			sd.stop()
	except Exception as e:
		raise AudioError(str(e))


def record_until_enter(sample_rate: int) -> np.ndarray:
	"""Record from microphone until the user presses Enter to stop.

	Returns mono int16 PCM samples.
	"""
	frames: List[np.ndarray] = []

	def _callback(indata, frames_count, time, status):  # type: ignore[no-untyped-def]
		# Copy to detach from sounddevice's buffer
		frames.append(indata.copy())

	try:
		with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16, callback=_callback):
			# Block until user presses Enter in the calling terminal
			input()
	except Exception as e:
			raise AudioError(str(e))

	if not frames:
		return np.zeros((0,), dtype=np.int16)

	pcm = np.concatenate(frames, axis=0).reshape(-1)
	return pcm


