from __future__ import annotations


class VoiceAgentError(Exception):
	pass


class TranscriptionError(VoiceAgentError):
	pass


class TtsError(VoiceAgentError):
	pass


class AudioError(VoiceAgentError):
	pass


class ToolError(VoiceAgentError):
	pass


