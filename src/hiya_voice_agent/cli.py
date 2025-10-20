from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from .settings import load_settings
from langchain_core.messages import HumanMessage, AIMessage
from .audio_io import record_pcm, pcm_to_wav_bytes, play_wav_bytes, record_until_enter
from .stt import transcribe_wav
from .tts import synthesize_wav
from .tts_stream import stream_tts_wav, split_text_for_tts, sanitize_for_speech
from .agent import build_agent
from .errors import VoiceAgentError, TranscriptionError, TtsError, AudioError, ToolError


def main() -> None:
	parser = argparse.ArgumentParser(prog="hiya-agent", description="OpenAI-only LangChain voice agent")
	parser.add_argument("--duration", type=float, default=None, help="Recording duration seconds (overrides settings)")
	parser.add_argument("--mode", type=str, choices=["ptt", "timed"], default="ptt", help="Recording mode: push-to-talk or fixed duration")
	args = parser.parse_args()

	# Optional .env
	if os.path.exists(".env"):
		load_dotenv()

	settings = load_settings()
	record_seconds = args.duration if args.duration is not None else settings.push_to_talk_duration_s

	chat_history = []
	agent = build_agent(settings.openai_model, settings.openai_api_key)

	while True:
		print("Press Enter to start speaking (or type q to quit):")
		cmd = input().strip().lower()
		if cmd in {"q", "quit", "exit"}:
			break

		print("Recording...")
		try:
			if args.mode == "ptt":
				print("Press Enter again to stop recording.")
				pcm = record_until_enter(settings.sample_rate)
			else:
				print(f"Recording for {record_seconds:.1f}s...")
				pcm = record_pcm(record_seconds, settings.sample_rate)
			wav_bytes = pcm_to_wav_bytes(pcm, settings.sample_rate)
		except AudioError as e:
			print(f"Audio error: {e}")
			continue

		print("Transcribing...")
		try:
			text = transcribe_wav(wav_bytes, settings.openai_whisper_model, settings.openai_api_key)
		except TranscriptionError as e:
			print(f"STT error: {e}")
			continue
		print(f"You: {text}")

		print("Thinking...")
		try:
			response = agent.invoke({"input": text, "chat_history": chat_history})
		except Exception as e:
			print(f"Agent error: {e}")
			continue
		assistant_text = response.get("output", str(response)) if isinstance(response, dict) else str(response)
		print(f"Assistant: {assistant_text}")

		chat_history.append(HumanMessage(content=text))
		chat_history.append(AIMessage(content=assistant_text))

		print("Speaking...")
		# Stream TTS for lower latency
		try:
			spoken_text = sanitize_for_speech(assistant_text)
			for segment in split_text_for_tts(spoken_text):
				chunks = list(stream_tts_wav(segment, settings.openai_tts_model, settings.voice, settings.openai_api_key))
				audio = b"".join(chunks)
				play_wav_bytes(audio)
		except TtsError as e:
			print(f"TTS error: {e}")
			continue

	print("Goodbye.")


if __name__ == "__main__":
	main()


