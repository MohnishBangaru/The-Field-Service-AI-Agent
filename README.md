Field Service Agent
===================

Voice assistant for field service technicians. Push-to-talk microphone capture, OpenAI speech-to-text,
a LangChain tool-calling assistant (routing, places, directions, address validation, web search), and
OpenAI text-to-speech playback.

Quickstart
----------

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[web,dev]"
cp .env.example .env   # set OPENAI__API_KEY (and GOOGLE__API_KEY for routing tools)
field-agent --mode ptt
field-agent --mode timed --duration 5
field-web              # local text-to-speech page at http://127.0.0.1:8000
```

Configuration
-------------

Settings are read from the environment or `.env` using `SECTION__FIELD` names (see `.env.example`).
Google Maps tools (`optimize_route`, `get_directions`, `travel_estimates`, `validate_address`,
`google_places_search`) are registered only when `GOOGLE__API_KEY` is set; OpenStreetMap place search
and web tools are always available.

Architecture
------------

Hexagonal layout under `src/field_service_agent/`:

| Layer            | Role                                                                 |
|------------------|----------------------------------------------------------------------|
| `domain/`        | Constants, errors, immutable Pydantic schemas, pure text/geo logic   |
| `application/`   | Ports (Protocols) and use cases: conversation loop, routing, places  |
| `infrastructure/`| OpenAI, Google Maps, OpenStreetMap, DuckDuckGo, audio devices, settings |
| `adapters/`      | LangChain tool bindings, CLI, HTTP front end                          |
| `composition.py` | Composition root wiring settings into use cases                       |

Dependencies point inward only; every external system sits behind a port and is swappable.

Development
-----------

```bash
pytest
ruff check src tests
mypy src
```
