from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=None, env_prefix="", extra="ignore")

	openai_api_key: str
	openai_model: str = "gpt-4o-mini"
	openai_whisper_model: str = "whisper-1"
	openai_tts_model: str = "gpt-4o-mini-tts"
	voice: str = "alloy"
	sample_rate: int = 16000
	push_to_talk_duration_s: float = 5.0
	google_maps_api_key: str | None = None


def load_settings() -> Settings:
	# Optional .env load handled in CLI; here we just construct from env
	return Settings()  # type: ignore[call-arg]


