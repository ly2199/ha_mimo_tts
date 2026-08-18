"""Support for Mimo text-to-speech service."""
from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from homeassistant.components.tts import (
    TTSAudioRequest,
    TTSAudioResponse,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    CONF_API_KEY,
    MIMO_API_BASE,
    MIMO_TTS_MODEL,
    SUPPORTED_VOICES,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Mimo TTS entity from a config entry."""
    _LOGGER.info("Setting up Mimo TTS entity")
    async_add_entities([MimoTTSEntity(hass, config_entry)])


class MimoTTSEntity(TextToSpeechEntity):
    """Mimo TTS entity."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        self.hass = hass
        self._config_entry = config_entry
        self._api_key: str = config_entry.data[CONF_API_KEY]
        self._client: AsyncOpenAI | None = None

        self._attr_name = "Mimo Text-to-Speech"
        self._attr_unique_id = f"{config_entry.entry_id}_tts"
        self._default_voice = DEFAULT_VOICE
        self._default_language = DEFAULT_LANGUAGE

        _LOGGER.debug("Mimo TTS entity initialized")

    async def _async_get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = self._api_key

            def _create_client():
                return AsyncOpenAI(api_key=api_key, base_url=MIMO_API_BASE)

            self._client = await self.hass.async_add_executor_job(_create_client)
        return self._client

    @property
    def default_language(self) -> str | None:
        return self._default_language

    @property
    def supported_languages(self) -> list[str]:
        return SUPPORTED_LANGUAGES

    @property
    def supported_options(self) -> list[str] | None:
        return ["voice", "style"]

    @property
    def default_options(self) -> dict[str, Any] | None:
        return {"voice": self._default_voice}

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        if language not in SUPPORTED_LANGUAGES:
            return None
        return [
            Voice(voice_id=voice_id, name=name)
            for voice_id, name in SUPPORTED_VOICES.items()
        ]

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType | None:
        """单次合成（非流式，用于简短文本来回调）。"""
        if language not in SUPPORTED_LANGUAGES:
            language = self.default_language

        voice = self._default_voice
        if options and "voice" in options:
            voice = options["voice"]
            if voice not in SUPPORTED_VOICES:
                voice = self._default_voice

        style = options.get("style", "") if options else ""

        client = await self._async_get_client()
        messages = []
        if style:
            messages.append({"role": "user", "content": style})
        messages.append({"role": "assistant", "content": message})

        try:
            completion = await client.chat.completions.create(
                model=MIMO_TTS_MODEL,
                messages=messages,
                audio={"format": "wav", "voice": voice},
            )
            if (
                completion.choices
                and hasattr(completion.choices[0].message, "audio")
                and completion.choices[0].message.audio
            ):
                audio_b64 = completion.choices[0].message.audio.data
                if audio_b64:
                    return ("wav", base64.b64decode(audio_b64))
            return None
        except Exception as err:
            _LOGGER.exception("TTS 1-shot failed: %s", err)
            return None

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse | None:
        """真正的流式 TTS：使用 Mimo stream=True，实时 yield 音频块。"""
        _LOGGER.debug("Stream TTS: language=%s, options=%s", request.language, request.options)

        # 收集完整消息（Mimo 不支持文本流式输入）
        message_parts = []
        async for chunk in request.message_gen:
            message_parts.append(chunk)
        message = "".join(message_parts)

        if not message:
            _LOGGER.error("Empty message received")
            return None

        if request.language not in SUPPORTED_LANGUAGES:
            _LOGGER.error("Unsupported language: %s", request.language)
            return None

        voice = self._default_voice
        if request.options and "voice" in request.options:
            voice = request.options["voice"]
            if voice not in SUPPORTED_VOICES:
                _LOGGER.error("Unsupported voice: %s", voice)
                voice = self._default_voice

        style = request.options.get("style", "") if request.options else ""

        client = await self._async_get_client()
        messages = []
        if style:
            messages.append({"role": "user", "content": style})
        messages.append({"role": "assistant", "content": message})

        async def stream_generator() -> AsyncGenerator[bytes]:
            """生成流式音频数据（优先使用 MP3 格式）。"""
            try:
                # 使用 MP3 格式流式输出（如果 Mimo 支持）
                stream = await client.chat.completions.create(
                    model=MIMO_TTS_MODEL,
                    messages=messages,
                    audio={"format": "mp3", "voice": voice},
                    stream=True,
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    audio_data = getattr(delta, "audio", None)
                    if audio_data is not None:
                        if isinstance(audio_data, dict):
                            audio_b64 = audio_data.get("data")
                        else:
                            audio_b64 = getattr(audio_data, "data", None)
                        if audio_b64:
                            yield base64.b64decode(audio_b64)
            except Exception as err:
                _LOGGER.exception("MP3 streaming failed, falling back to 1-shot: %s", err)
                # 降级到非流式
                result = await self.async_get_tts_audio(message, request.language, request.options)
                if result:
                    _, data = result
                    yield data
                else:
                    _LOGGER.error("Fallback TTS failed")
                    return

        return TTSAudioResponse("mp3", stream_generator())
