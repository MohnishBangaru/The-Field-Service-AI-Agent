"""Minimal HTTP front end for text-to-speech."""

from __future__ import annotations

from http import HTTPStatus
from typing import Final

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, Response

from field_service_agent.application.ports import Synthesizer
from field_service_agent.composition import Assembler
from field_service_agent.domain.errors import SynthesisError
from field_service_agent.infrastructure.audio.codec import WavCodec
from field_service_agent.infrastructure.settings import SettingsLoader


class SpeechService:
    """Builds a FastAPI app that turns posted text into WAV audio."""

    __PAGE: Final[str] = """
    <!doctype html>
    <html><head><meta charset="utf-8"><title>Field Service Agent</title></head>
    <body>
    <h3>Field Service Agent - Text to Speech</h3>
    <form method="post" action="/tts">
        <input name="text" placeholder="Type something" size="50"/>
        <button type="submit">Speak</button>
    </form>
    <audio id="audio" controls></audio>
    <script>
      document.querySelector('form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const text = new FormData(event.target).get('text');
        const response = await fetch('/tts', { method: 'POST', body: new URLSearchParams({text}) });
        document.getElementById('audio').src = URL.createObjectURL(await response.blob());
      });
    </script>
    </body></html>
    """

    def __init__(self, *, synthesizer: Synthesizer, codec: WavCodec) -> None:
        self.__synthesizer = synthesizer
        self.__codec = codec

    def build(self) -> FastAPI:
        """FastAPI application with the index page and /tts endpoint."""
        app = FastAPI(title="Field Service Agent")
        app.get("/", response_class=HTMLResponse)(self.__index)
        app.post("/tts")(self.__speak)
        return app

    def __index(self) -> str:
        return self.__PAGE

    def __speak(self, text: str = Form(...)) -> Response:
        cleaned = text.strip()
        if not cleaned:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Text must not be empty")
        try:
            clip = self.__synthesizer.synthesize(text=cleaned)
        except SynthesisError as exception:
            raise HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail=f"Speech synthesis failed: {exception}") from exception
        return Response(content=self.__codec.encode(clip=clip), media_type="audio/wav")


class WebServer:
    """Runs the speech service under uvicorn."""

    @staticmethod
    def run() -> None:
        """Console-script entry point."""
        import uvicorn

        settings = SettingsLoader.load()
        assembler = Assembler(settings=settings)
        app = SpeechService(synthesizer=assembler.synthesizer(), codec=assembler.codec()).build()
        uvicorn.run(app, host=settings.web.host, port=settings.web.port)
