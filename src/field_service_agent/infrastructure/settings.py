"""Environment-backed configuration validated at process start."""

from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseModel):
    """OpenAI credentials and model selection."""

    api_key: SecretStr = Field(description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="Chat model used for the assistant")
    transcription_model: str = Field(default="whisper-1", description="Speech-to-text model")
    speech_model: str = Field(default="gpt-4o-mini-tts", description="Text-to-speech model")
    voice: str = Field(default="alloy", description="Text-to-speech voice name")


class GoogleSettings(BaseModel):
    """Google Maps Platform credentials."""

    api_key: SecretStr | None = Field(default=None, description="Google Maps Platform API key")


class AudioSettings(BaseModel):
    """Microphone and playback parameters."""

    rate: int = Field(default=16_000, gt=0, description="Sample rate in samples per second")
    duration: float = Field(default=5.0, gt=0, description="Recording length in seconds for timed mode")


class HttpSettings(BaseModel):
    """Outbound HTTP client parameters."""

    timeout: float = Field(default=20.0, gt=0, description="Request timeout in seconds")
    user_agent: str = Field(default="field-service-agent/0.1", description="User-Agent header for outbound requests")


class WebSettings(BaseModel):
    """Local HTTP server binding."""

    host: str = Field(default="127.0.0.1", description="Interface to bind")
    port: int = Field(default=8000, ge=1, le=65_535, description="TCP port to bind")


class Settings(BaseSettings):
    """Root configuration; nested fields map to OPENAI__API_KEY style variables."""

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    openai: OpenAISettings = Field(description="OpenAI configuration")
    google: GoogleSettings = Field(default_factory=GoogleSettings, description="Google Maps configuration")
    audio: AudioSettings = Field(default_factory=AudioSettings, description="Audio device configuration")
    http: HttpSettings = Field(default_factory=HttpSettings, description="Outbound HTTP configuration")
    web: WebSettings = Field(default_factory=WebSettings, description="Local web server configuration")


class SettingsLoader:
    """Builds Settings from the environment and an optional dotenv file."""

    @staticmethod
    def load() -> Settings:
        """Load and validate settings; raises on missing required values."""
        return Settings()  # type: ignore[call-arg]
