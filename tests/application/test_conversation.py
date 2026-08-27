"""Tests for the voice conversation loop."""

from __future__ import annotations

from field_service_agent.application.conversation import ConversationOptions, VoiceConversation
from field_service_agent.domain.constants import RecordingMode
from field_service_agent.domain.schemas import AudioClip
from field_service_agent.domain.speech import SpeechSanitizer, SpeechSegmenter
from tests.application.fakes import (
    CapturingPlayer,
    EchoAssistant,
    EchoSynthesizer,
    FixedRecorder,
    FixedTranscriber,
    ScriptedConsole,
)

SPEECH = AudioClip(pcm=b"\x01\x00" * 100, rate=16_000)


class TestVoiceConversation:
    """VoiceConversation drives one turn through every port."""

    def __build(
        self,
        *,
        answers: list[str],
        clip: AudioClip = SPEECH,
        transcript: str | None = "Route me to 1 Main St and 2 Oak Ave",
        mode: RecordingMode = RecordingMode.PUSH_TO_TALK,
    ) -> tuple[VoiceConversation, ScriptedConsole, CapturingPlayer, EchoSynthesizer, EchoAssistant, FixedRecorder]:
        console = ScriptedConsole(answers=answers)
        player = CapturingPlayer()
        synthesizer = EchoSynthesizer()
        assistant = EchoAssistant()
        recorder = FixedRecorder(clip=clip)
        conversation = VoiceConversation(
            recorder=recorder,
            player=player,
            transcriber=FixedTranscriber(text=transcript),
            synthesizer=synthesizer,
            assistant=assistant,
            console=console,
            sanitizer=SpeechSanitizer(),
            segmenter=SpeechSegmenter(limit=40),
            options=ConversationOptions(mode=mode, duration=2.0, rate=16_000),
        )
        return conversation, console, player, synthesizer, assistant, recorder

    def test_turn_records_transcribes_replies_and_speaks(self) -> None:
        conversation, console, player, synthesizer, _, _ = self.__build(answers=[])
        exchange = conversation.turn()
        assert exchange is not None
        assert exchange.user == "Route me to 1 Main St and 2 Oak Ave"
        assert exchange.assistant == "You said: Route me to 1 Main St and 2 Oak Ave"
        assert synthesizer.texts == ["You said: Route me to 1 Main St and 2 Oak Ave"]
        assert len(player.played) == 1
        assert "You: Route me to 1 Main St and 2 Oak Ave" in console.said

    def test_history_accumulates_and_is_passed_to_assistant(self) -> None:
        conversation, _, _, _, assistant, _ = self.__build(answers=[])
        conversation.turn()
        conversation.turn()
        assert len(conversation.history) == 2
        assert assistant.histories[0] == ()
        assert assistant.histories[1] == (conversation.history[0],)

    def test_timed_mode_uses_configured_duration(self) -> None:
        conversation, _, _, _, _, recorder = self.__build(answers=[], mode=RecordingMode.TIMED)
        conversation.turn()
        assert recorder.timed_calls == [2.0]
        assert recorder.stopped_calls == 0

    def test_empty_recording_skips_turn(self) -> None:
        conversation, console, _, synthesizer, _, _ = self.__build(answers=[], clip=AudioClip(pcm=b"", rate=16_000))
        assert conversation.turn() is None
        assert "No audio captured." in console.said
        assert synthesizer.texts == []

    def test_transcription_failure_is_reported_not_raised(self) -> None:
        conversation, console, _, _, _, _ = self.__build(answers=[], transcript=None)
        assert conversation.turn() is None
        assert any(message.startswith("Transcription error:") for message in console.said)

    def test_run_stops_on_quit_command(self) -> None:
        conversation, console, _, _, _, _ = self.__build(answers=["", "quit"])
        conversation.run()
        assert len(conversation.history) == 1
        assert console.said[-1] == "Goodbye."
