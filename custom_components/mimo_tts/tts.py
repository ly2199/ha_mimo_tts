"""Support for Mimo text-to-speech service."""
from __future__ import annotations

import asyncio
import base64
import logging
import struct
from collections.abc import AsyncGenerator
from typing import Any

import httpx
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

# 音频缓冲大小（字节），适中的大小可减少卡顿且保持低延迟
AUDIO_BUFFER_SIZE = 32768

# Mimo PCM16 音频参数（24kHz mono 16-bit，来自官方文档）
MIMO_SAMPLE_RATE = 24000
MIMO_CHANNELS = 1
MIMO_SAMPLE_WIDTH = 2  # 16-bit


def _create_wav_header(data_size: int = 0xFFFFFFFF) -> bytes:
    """创建流式 WAV 文件头（长度未知时使用 0xFFFFFFFF）。"""
    byte_rate = MIMO_SAMPLE_RATE * MIMO_CHANNELS * MIMO_SAMPLE_WIDTH
    block_align = MIMO_CHANNELS * MIMO_SAMPLE_WIDTH
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM format
        MIMO_CHANNELS,
        MIMO_SAMPLE_RATE,
        byte_rate,
        block_align,
        MIMO_SAMPLE_WIDTH * 8,
        b"data",
        0xFFFFFFFF,
    )


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
                return AsyncOpenAI(
                    api_key=api_key,
                    base_url=MIMO_API_BASE,
                    timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
                )

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
        """流式 TTS：使用 Mimo stream=True，输出带 WAV 头的 PCM16 流。"""
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
            """生成带 WAV 头的 PCM16 流式音频，缓冲后输出。"""
            header_sent = False
            buffer = bytearray()
            try:
                stream = await client.chat.completions.create(
                    model=MIMO_TTS_MODEL,
                    messages=messages,
                    audio={"format": "pcm16", "voice": voice},
                    stream=True,
                )

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    audio_data = getattr(delta, "audio", None)
                    if audio_data is None:
                        continue
                    if isinstance(audio_data, dict):
                        audio_b64 = audio_data.get("data")
                    else:
                        audio_b64 = getattr(audio_data, "data", None)
                    if not audio_b64:
                        continue

                    pcm_data = base64.b64decode(audio_b64)
                    if not header_sent:
                        yield _create_wav_header()
                        header_sent = True

                    buffer.extend(pcm_data)
                    while len(buffer) >= AUDIO_BUFFER_SIZE:
                        yield buffer[:AUDIO_BUFFER_SIZE]
                        buffer = buffer[AUDIO_BUFFER_SIZE:]

                if buffer:
                    yield bytes(buffer)

            except Exception as err:
                _LOGGER.exception("PCM16 streaming failed: %s", err)
                if not header_sent:
                    # 尚未发送任何流式数据，可安全降级到单次 WAV
                    _LOGGER.warning("Falling back to 1-shot TTS")
                    result = await self.async_get_tts_audio(message, request.language, request.options)
                    if result:
                        _, data = result
                        yield data
                    else:
                        _LOGGER.error("Fallback TTS failed")
                else:
                    # 已发送流式头部，无法降级；只能终止，避免混合音频
                    _LOGGER.error("Stream failed after sending WAV header; cannot fallback")
                    return

        # 返回扩展名为 "wav"，因为现在我们发送的是合法的 WAV 流
        return TTSAudioResponse("wav", stream_generator())
