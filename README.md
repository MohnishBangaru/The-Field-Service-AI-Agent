Hiya Voice Agent
=================

OpenAI-only LangChain voice agent with push-to-talk mic capture, Whisper STT, LangChain LLM response, and OpenAI TTS playback.

Quickstart
----------

1) Create and activate a virtualenv, then install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

2) Create a `.env` by copying `.env.example` and set `OPENAI_API_KEY`.

3) Run the agent:

```bash
hiya-agent --duration 5
```

Notes
-----
- Uses OpenAI models by default: `gpt-4o-mini` for LLM, `whisper-1` for STT, and `gpt-4o-mini-tts` for TTS.
- Push-to-talk: press Enter to record for the configured duration.
- Requires microphone/speaker access and `sounddevice`.


