"""Command-line entry point for the voice agent."""

from __future__ import annotations

import argparse
import logging
import sys

from pydantic import ValidationError

from field_service_agent.application.conversation import ConversationOptions
from field_service_agent.composition import Assembler
from field_service_agent.domain.constants import RecordingMode
from field_service_agent.domain.errors import ConfigurationError
from field_service_agent.infrastructure.settings import SettingsLoader


class CommandLine:
    """Parses arguments, assembles the agent, and runs the conversation loop."""

    PROGRAM = "field-agent"

    @classmethod
    def run(cls) -> None:
        """Console-script entry point."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        arguments = cls.__parser().parse_args()
        try:
            settings = SettingsLoader.load()
        except ValidationError as exception:
            raise ConfigurationError(f"Invalid configuration; check .env or environment variables: {exception}") from exception
        options = ConversationOptions(
            mode=RecordingMode(arguments.mode),
            duration=arguments.duration if arguments.duration is not None else settings.audio.duration,
            rate=settings.audio.rate,
        )
        try:
            Assembler(settings=settings).conversation(options=options).run()
        except KeyboardInterrupt:
            sys.exit(0)

    @classmethod
    def __parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog=cls.PROGRAM, description="Voice assistant for field service technicians")
        parser.add_argument("--duration", type=float, default=None, help="Recording length in seconds for timed mode")
        parser.add_argument(
            "--mode",
            choices=[mode.value for mode in RecordingMode],
            default=RecordingMode.PUSH_TO_TALK.value,
            help="Recording mode: push-to-talk or fixed duration",
        )
        return parser
