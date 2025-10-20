from __future__ import annotations

import os
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from dotenv import load_dotenv

from .settings import load_settings
from .tts import synthesize_wav
from .errors import TtsError


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
	return (
		"""
		<!doctype html>
		<html><head><meta charset="utf-8"><title>Hiya Voice</title></head>
		<body>
		<h3>Hiya Voice - Simple TTS</h3>
		<form method="post" action="/tts">
			<input name="text" placeholder="Type something" size="50"/>
			<button type="submit">Speak</button>
		</form>
		<audio id="audio" controls></audio>
		<script>
		  document.querySelector('form').addEventListener('submit', async (e) => {
		    e.preventDefault();
		    const text = new FormData(e.target).get('text');
		    const resp = await fetch('/tts', { method: 'POST', body: new URLSearchParams({text}) });
		    const blob = await resp.blob();
		    const url = URL.createObjectURL(blob);
		    document.getElementById('audio').src = url;
		  });
		</script>
		</body></html>
		"""
	)


@app.post("/tts")
def tts(text: str = Form(...)):
	if os.path.exists('.env'):
		load_dotenv()
	settings = load_settings()
	text = (text or "").strip()
	if not text:
		raise HTTPException(status_code=400, detail="Text must not be empty")
	try:
		wav = synthesize_wav(text, settings.openai_tts_model, settings.voice, settings.openai_api_key)
		return StreamingResponse(iter([wav]), media_type="audio/wav")
	except TtsError as e:
		raise HTTPException(status_code=502, detail=f"TTS error: {e}")


def run() -> None:
	import uvicorn
	uvicorn.run(app, host="127.0.0.1", port=8000)


