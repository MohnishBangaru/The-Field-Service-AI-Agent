"""Voice conversation loop: capture, transcribe, reply, speak."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from field_service_agent.application.ports import Assistant, Console, Player, Recorder, Synthesizer, Transcriber
from field_service_agent.domain.constants import RecordingMode
from field_service_agent.domain.errors import AssistantError, AudioError, SynthesisError, TranscriptionError
from field_service_agent.domain.schemas import Exchange
from field_service_agent.domain.speech import SpeechSanitizer, SpeechSegmenter


class ConversationOptions(BaseModel):
    """Runtime choices for a conversation session."""

    model_config = ConfigDict(frozen=True)

    mode: RecordingMode = Field(default=RecordingMode.PUSH_TO_TALK, description="How recording starts and stops")
    duration: float = Field(default=5.0, gt=0, description="Seconds to record in timed mode")
    rate: int = Field(default=16_000, gt=0, description="Microphone sample rate in samples per second")


class VoiceConversation:
    """Runs repeated spoken turns between the user and the assistant."""

    __QUIT_COMMANDS: Final[frozenset[str]] = frozenset({"q", "quit", "exit"})

    def __init__(
        self,
        *,
        recorder: Recorder,
        player: Player,
        transcriber: Transcriber,
        synthesizer: Synthesizer,
        assistant: Assistant,
        console: Console,
        sanitizer: SpeechSanitizer,
        segmenter: SpeechSegmenter,
        options: ConversationOptions,
    ) -> None:
        self.__recorder = recorder
        self.__player = player
        self.__transcriber = transcriber
        self.__synthesizer = synthesizer
        self.__assistant = assistant
        self.__console = console
        self.__sanitizer = sanitizer
        self.__segmenter = segmenter
        self.__options = options
        self.__history: tuple[Exchange, ...] = ()

    @property
    def history(self) -> tuple[Exchange, ...]:
        """Completed exchanges so far."""
        return self.__history

    def run(self) -> None:
        """Loop over turns until the user asks to quit."""
        while True:
            command = self.__console.prompt(message="Press Enter to start speaking (or type q to quit):").strip().lower()
            if command in self.__QUIT_COMMANDS:
                break
            self.turn()
        self.__console.say(message="Goodbye.")

    def turn(self) -> Exchange | None:
        """Run one full spoken exchange; returns None when a stage failed."""
        try:
            spoken = self.__capture()
        except AudioError as exception:
            self.__console.say(message=f"Audio error: {exception}")
            return None
        if spoken is None:
            return None
        try:
            reply = self.__assistant.reply(message=spoken, history=self.__history)
        except AssistantError as exception:
            self.__console.say(message=f"Assistant error: {exception}")
            return None
        self.__console.say(message=f"Assistant: {reply}")
        exchange = Exchange(user=spoken, assistant=reply)
        self.__history = (*self.__history, exchange)
        self.__speak(text=reply)
        return exchange

    def __capture(self) -> str | None:
        self.__console.say(message="Recording...")
        if self.__options.mode is RecordingMode.PUSH_TO_TALK:
            self.__console.say(message="Press Enter again to stop recording.")
            clip = self.__recorder.record_until_stopped(rate=self.__options.rate)
        else:
            self.__console.say(message=f"Recording for {self.__options.duration:.1f}s...")
            clip = self.__recorder.record_timed(duration=self.__options.duration, rate=self.__options.rate)
        if clip.empty:
            self.__console.say(message="No audio captured.")
            return None
        self.__console.say(message="Transcribing...")
        try:
            text = self.__transcriber.transcribe(clip=clip).strip()
        except TranscriptionError as exception:
            self.__console.say(message=f"Transcription error: {exception}")
            return None
        if not text:
            self.__console.say(message="Nothing was heard.")
            return None
        self.__console.say(message=f"You: {text}")
        return text

    def __speak(self, *, text: str) -> None:
        self.__console.say(message="Speaking...")
        spoken = self.__sanitizer.sanitize(text=text)
        segments = self.__segmenter.segment(text=spoken) or ("I could not produce anything to say.",)
        try:
            for segment in segments:
                self.__player.play(clip=self.__synthesizer.synthesize(text=segment))
        except (SynthesisError, AudioError) as exception:
            self.__console.say(message=f"Speech error: {exception}")
