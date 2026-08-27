"""Domain-specific exception hierarchy."""

from __future__ import annotations


class FieldServiceAgentError(Exception):
    """Base class for every error raised by the agent."""


class ConfigurationError(FieldServiceAgentError):
    """Raised when required configuration is missing or invalid."""


class AudioError(FieldServiceAgentError):
    """Raised when capturing, encoding, or playing audio fails."""


class TranscriptionError(FieldServiceAgentError):
    """Raised when speech-to-text fails."""


class SynthesisError(FieldServiceAgentError):
    """Raised when text-to-speech fails."""


class AssistantError(FieldServiceAgentError):
    """Raised when the language model fails to produce a reply."""


class ToolError(FieldServiceAgentError):
    """Raised when an agent tool cannot complete its request."""


class GatewayError(FieldServiceAgentError):
    """Raised when an external service returns an error or malformed payload."""


class RoutingError(FieldServiceAgentError):
    """Raised when a route cannot be planned from the supplied stops."""
