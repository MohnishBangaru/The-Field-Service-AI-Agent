# Field Service Agent

A voice assistant for field service technicians. Speak a request, and the agent transcribes it,
reasons with a tool-calling language model, and answers out loud. It can plan multi-stop routes,
give turn-by-turn directions, estimate travel times, validate addresses, find nearby suppliers,
and look things up on the web.

Built as a hexagonal, fully typed Python application: every external system sits behind a port and
can be swapped without touching business logic.

## Features

- **Push-to-talk or timed voice capture** from the default microphone.
- **Speech-to-text and text-to-speech** through OpenAI, with retry policies and links stripped
  from spoken replies.
- **Tool-calling assistant** on LangChain 1.x `create_agent`, keeping multi-turn history.
- **Field-service tools**: route optimization with traffic-aware totals, directions, distance
  matrix, address validation, Google and OpenStreetMap place search, web search and page reading.
- **Graceful degradation**: Google tools register only when a key is configured; tool failures
  are reported back to the model instead of crashing the session.
- **Strict engineering baseline**: Pydantic schemas at every boundary, Protocol ports, no global
  state, `mypy --strict`, ruff, and a class-based test suite.

## How a turn works

```
 microphone ──> Recorder ──> AudioClip ──> Transcriber ──> text
                                                            │
                                     Assistant (LLM + tools) <──── history
                                                            │
 speaker <── Player <── AudioClip <── Synthesizer <── SpeechSanitizer / Segmenter
```

1. `VoiceConversation` asks the `Recorder` for an `AudioClip` (press Enter to stop, or a fixed
   duration).
2. The `Transcriber` turns it into text; empty or failed captures end the turn with a message.
3. The `Assistant` receives the text plus prior `Exchange`s and may call any registered tool.
4. The reply is sanitized (URLs removed), split into speakable segments, synthesized, and played.

## Tools available to the assistant

| Tool | Backend | Needs `GOOGLE__API_KEY` |
|---|---|---|
| `current_time` | System clock, IANA timezones | no |
| `web_search` | DuckDuckGo | no |
| `fetch_page` | HTTP + readable-text extraction | no |
| `search_nearby_places` | OpenStreetMap Nominatim + Overpass | no |
| `google_places_search` | Google Places (New) text search | yes |
| `optimize_route` | Google Routes API (waypoint optimization, traffic-aware totals) | yes |
| `get_directions` | Google Directions API | yes |
| `travel_estimates` | Google Distance Matrix API | yes |
| `validate_address` | Google Address Validation API | yes |

Example requests:

- "Plan the fastest order to visit 12 Elm St, 40 Pine Ave, and 7 Harbor Rd, starting from the depot."
- "How long from the warehouse to 220 Market St in traffic?"
- "Find a plumbing supply store within a kilometer of Mission District."
- "Is 1600 Amphitheatre Parkway a valid address?"

## Quickstart

Requirements: Python 3.11+, a working microphone and speaker, an OpenAI API key. A Google Maps
Platform key enables the routing tools.

```bash
git clone https://github.com/MohnishBangaru/The-Field-Service-AI-Agent.git
cd The-Field-Service-AI-Agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[web,dev]"
cp .env.example .env        # fill in OPENAI__API_KEY (and GOOGLE__API_KEY)

field-agent                 # push-to-talk: Enter to start, Enter to stop, q to quit
field-agent --mode timed --duration 5
field-web                   # browser text-to-speech demo at http://127.0.0.1:8000
```

## Configuration

Settings are loaded from the environment or `.env` with `SECTION__FIELD` names and validated at
startup; a missing required value fails fast with a clear error.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI__API_KEY` | required | OpenAI credentials |
| `OPENAI__MODEL` | `gpt-4o-mini` | Chat model for the assistant |
| `OPENAI__TRANSCRIPTION_MODEL` | `whisper-1` | Speech-to-text model |
| `OPENAI__SPEECH_MODEL` | `gpt-4o-mini-tts` | Text-to-speech model |
| `OPENAI__VOICE` | `alloy` | Text-to-speech voice |
| `GOOGLE__API_KEY` | unset | Enables Google Maps tools |
| `AUDIO__RATE` | `16000` | Microphone sample rate |
| `AUDIO__DURATION` | `5` | Recording length in seconds for timed mode |
| `HTTP__TIMEOUT` | `20` | Outbound request timeout in seconds |
| `HTTP__USER_AGENT` | `field-service-agent/0.1` | User-Agent for outbound requests |
| `WEB__HOST` / `WEB__PORT` | `127.0.0.1` / `8000` | Binding for `field-web` |

## Architecture

```
src/field_service_agent/
├── domain/          constants, errors, frozen Pydantic schemas, geometry, speech text rules
├── application/     ports (Protocols) + use cases: conversation, routing, places
├── infrastructure/  openai/ google/ osm/ web/ audio/ system/ settings.py
├── adapters/        tools/ (LangChain bindings, presenter, parser)  cli/  http/
└── composition.py   composition root wiring settings into use cases
```

Dependency direction is strictly inward: `adapters` and `infrastructure` depend on `application`
ports, `application` depends on `domain`, and `domain` depends on nothing. The domain layer has no
HTTP, SDK, file system, or numpy imports and is tested without any infrastructure.

Key design choices:

- **Ports over providers.** `Transcriber`, `Synthesizer`, `Assistant`, `Geocoder`,
  `RoutePlanner`, `PlaceFinder`, and friends are `Protocol`s. Google and OpenStreetMap implement
  the same `Geocoder`/`PlaceFinder` contracts and are interchangeable.
- **Typed boundaries.** External payloads are validated into immutable schemas (`Place`,
  `RouteSummary`, `TravelEstimate`, `AudioClip`) before reaching use cases.
- **Deterministic core.** Time comes from an injected `Clock`; console I/O from an injected
  `Console`; nothing reads the environment except `Settings`.
- **Tools as classes.** Each tool group (`TimeTools`, `WebTools`, `PlaceTools`, `RoutingTools`)
  exposes methods that a `ToolFactory` wraps as LangChain `StructuredTool`s, converting domain
  errors into messages the model can recover from.

## Extending

**Add a tool.** Create a method on an existing group (or a new class with a `tools()` method),
register it through `ToolFactory.build(name=..., description=..., func=...)`, and add the group in
`Assembler.tools()`. Keep provider calls in `infrastructure/` behind a port.

**Swap a provider.** Implement the relevant port (for example a local Whisper `Transcriber` or a
Mapbox `RoutePlanner`) and change one line in `composition.py`. No use case or tool code changes.

## Development

```bash
pytest                 # class-based tests mirroring the source tree
ruff check src tests   # lint and import order
ruff format src tests
mypy src               # strict type checking
```

Tests exercise real scenarios through in-memory ports: route ordering with pinned endpoints,
unresolvable stops, Google Routes and Distance Matrix payload parsing, WAV round-trips, and full
conversation turns.

## Troubleshooting

- **`Invalid configuration`** on start: `OPENAI__API_KEY` is missing from `.env`/environment.
- **No Google tools listed**: set `GOOGLE__API_KEY` and enable the Geocoding, Places (New),
  Routes, Directions, Distance Matrix, Time Zone, and Address Validation APIs on the key.
- **`Microphone capture failed`**: sounddevice could not open the default input; check OS
  permissions and that a device is selected.

## License

MIT. See [LICENSE](LICENSE).
