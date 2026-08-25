"""Composition root wiring infrastructure into application use cases."""

from __future__ import annotations

import logging

import httpx
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import OpenAI

from field_service_agent.adapters.tools.addresses import AddressParser
from field_service_agent.adapters.tools.factory import ToolFactory
from field_service_agent.adapters.tools.places import PlaceTools
from field_service_agent.adapters.tools.presenter import ToolPresenter
from field_service_agent.adapters.tools.routing import RoutingTools
from field_service_agent.adapters.tools.time import TimeTools
from field_service_agent.adapters.tools.web import WebTools
from field_service_agent.application.conversation import ConversationOptions, VoiceConversation
from field_service_agent.application.places import PlaceDiscovery
from field_service_agent.application.ports import Synthesizer
from field_service_agent.application.routing import RouteOptimization
from field_service_agent.domain.speech import SpeechSanitizer, SpeechSegmenter
from field_service_agent.infrastructure.audio.codec import WavCodec
from field_service_agent.infrastructure.audio.device import SoundDevicePlayer, SoundDeviceRecorder
from field_service_agent.infrastructure.google.maps import GoogleAddressValidator, GoogleGeocoder, GooglePlaceFinder, GoogleRoutePlanner
from field_service_agent.infrastructure.openai.assistant import LangChainAssistant
from field_service_agent.infrastructure.openai.synthesis import OpenAISynthesizer
from field_service_agent.infrastructure.openai.transcription import OpenAITranscriber
from field_service_agent.infrastructure.osm.geocoder import NominatimGeocoder
from field_service_agent.infrastructure.osm.places import OverpassPlaceFinder
from field_service_agent.infrastructure.settings import Settings
from field_service_agent.infrastructure.system.clock import SystemClock
from field_service_agent.infrastructure.system.console import TerminalConsole
from field_service_agent.infrastructure.web.http import HttpGateway
from field_service_agent.infrastructure.web.search import DuckDuckGoSearcher, HtmlPageFetcher


class Assembler:
    """Creates fully wired objects from validated settings."""

    def __init__(self, *, settings: Settings) -> None:
        self.__settings = settings
        self.__codec = WavCodec()
        self.__console = TerminalConsole()
        self.__clock = SystemClock()
        self.__openai = OpenAI(api_key=settings.openai.api_key.get_secret_value())
        self.__http = HttpGateway(client=httpx.Client(), timeout=settings.http.timeout, user_agent=settings.http.user_agent)
        self.__factory = ToolFactory(logger=logging.getLogger("field_service_agent.tools"))
        self.__presenter = ToolPresenter()

    def codec(self) -> WavCodec:
        """Shared WAV codec."""
        return self.__codec

    def synthesizer(self) -> Synthesizer:
        """OpenAI text-to-speech."""
        openai = self.__settings.openai
        return OpenAISynthesizer(client=self.__openai, model=openai.speech_model, voice=openai.voice, codec=self.__codec)

    def tools(self) -> tuple[BaseTool, ...]:
        """Every tool available to the assistant given current configuration."""
        tools: list[BaseTool] = [
            *TimeTools(clock=self.__clock, factory=self.__factory).tools(),
            *WebTools(
                searcher=DuckDuckGoSearcher(), fetcher=HtmlPageFetcher(http=self.__http), presenter=self.__presenter, factory=self.__factory
            ).tools(),
        ]
        open_discovery = PlaceDiscovery(geocoder=NominatimGeocoder(http=self.__http), finder=OverpassPlaceFinder(http=self.__http))
        google_discovery = None
        key = self.__settings.google.api_key
        if key is not None:
            api_key = key.get_secret_value()
            geocoder = GoogleGeocoder(http=self.__http, api_key=api_key)
            planner = GoogleRoutePlanner(http=self.__http, api_key=api_key, clock=self.__clock)
            google_discovery = PlaceDiscovery(geocoder=geocoder, finder=GooglePlaceFinder(http=self.__http, api_key=api_key))
            tools.extend(
                RoutingTools(
                    optimization=RouteOptimization(geocoder=geocoder, planner=planner),
                    planner=planner,
                    validator=GoogleAddressValidator(http=self.__http, api_key=api_key),
                    parser=AddressParser(),
                    presenter=self.__presenter,
                    factory=self.__factory,
                ).tools()
            )
        tools.extend(
            PlaceTools(
                open_discovery=open_discovery, google_discovery=google_discovery, presenter=self.__presenter, factory=self.__factory
            ).tools()
        )
        return tuple(tools)

    def conversation(self, *, options: ConversationOptions) -> VoiceConversation:
        """Interactive voice loop with all dependencies attached."""
        openai = self.__settings.openai
        model = ChatOpenAI(model=openai.model, api_key=openai.api_key)
        return VoiceConversation(
            recorder=SoundDeviceRecorder(console=self.__console),
            player=SoundDevicePlayer(),
            transcriber=OpenAITranscriber(client=self.__openai, model=openai.transcription_model, codec=self.__codec),
            synthesizer=self.synthesizer(),
            assistant=LangChainAssistant(model=model, tools=self.tools()),
            console=self.__console,
            sanitizer=SpeechSanitizer(),
            segmenter=SpeechSegmenter(),
            options=options,
        )
