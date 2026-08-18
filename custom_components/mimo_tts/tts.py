"""Support for Mimo text-to-speech service."""
from __future__ import annotations

import asyncio
import base64
import logging
import struct
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
    AUDIO_FORMAT,
)

_LOGGER = logging.getLogger(__name__)

# 分句相关常量
SENTENCE_DELIMITERS = "。！？!?；;\n\r…"
SUB_DELIMITERS = "，,、:："
MAX_CHUNK_CHARS = 500
MAX_CONCURRENT_REQUESTS = 5


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            cut = -1
            for delimiters in (SENTENCE_DELIMITERS, SUB_DELIMITERS):
                for i in range(end - 1, start, -1):
                    if text[i] in delimiters:
                        cut = i + 1
                        break
                if cut > start:
                    break
            end = cut if cut > start else end
        chunks.append(text[start:end])
        start = end
    return chunks


def concatenate_wav(fragments: list[bytes]) -> bytes:
    """Merge multiple WAV fragments into a single valid WAV file."""
    if not fragments:
        return b""
    if len(fragments) == 1:
        return fragments[0]

    first = fragments[0]
    data_idx = first.find(b"data")
    if data_idx == -1:
        # Fallback: simply concatenate
        return b"".join(fragments)

    # Prepare header from first fragment (up to "data" + 4 bytes length)
    header = bytearray(first[: data_idx + 8])
    total_data = 0
    data_parts = []

    for frag in fragments:
        idx = frag.find(b"data")
        if idx != -1:
            size = struct.unpack_from("<I", frag, idx + 4)[0]
            total_data += size
            data_parts.append(frag[idx + 8 : idx + 8 + size])
        else:
            # If no data chunk, treat entire content as PCM (should not happen)
            total_data += len(frag)
            data_parts.append(frag)

    # Update RIFF chunk size (total file size - 8)
    riff_size = 36 + total_data  # 44 - 8 = 36
    struct.pack_into("<I", header, 4, riff_size)
    # Update data chunk size
    struct.pack_into("<I", header, data_idx + 4, total_data)

    return bytes(header) + b"".join(data_parts)


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
        """Initialize Mimo TTS entity."""
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
        """异步获取客户端，在 executor 中创建以避免阻塞事件循环."""
        if self._client is None:
            _LOGGER.debug("Creating AsyncOpenAI client in executor")
            api_key = self._api_key

            def _create_client():
                return AsyncOpenAI(
                    api_key=api_key,
                    base_url=MIMO_API_BASE,
                )

            self._client = await self.hass.async_add_executor_job(_create_client)
        return self._client

    @property
    def default_language(self) -> str | None:
        """Return the default language."""
        return self._default_language

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_options(self) -> list[str] | None:
        """Return a list of supported options like voice, style."""
        return ["voice", "style"]

    @property
    def default_options(self) -> dict[str, Any] | None:
        """Return the default options."""
        return {"voice": self._default_voice}

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return a list of supported voices for a language."""
        if language not in SUPPORTED_LANGUAGES:
            return None
        return [
            Voice(voice_id=voice_id, name=name)
            for voice_id, name in SUPPORTED_VOICES.items()
        ]

    async def _synthesize_chunk(
        self,
        text: str,
        voice: str,
        style: str,
    ) -> bytes:
        """Synthesize a single text chunk and return the WAV bytes."""
        client = await self._async_get_client()
        messages = []
        if style:
            messages.append({"role": "user", "content": style})
        messages.append({"role": "assistant", "content": text})

        try:
            completion = await client.chat.completions.create(
                model=MIMO_TTS_MODEL,
                messages=messages,
                audio={
                    "format": AUDIO_FORMAT,
                    "voice": voice,
                },
            )
            if (
                completion.choices
                and completion.choices[0].message
                and hasattr(completion.choices[0].message, "audio")
                and completion.choices[0].message.audio
            ):
                audio_b64 = completion.choices[0].message.audio.data
                if audio_b64:
                    return base64.b64decode(audio_b64)
            _LOGGER.error("No audio data in response for chunk: %s", text[:30])
            raise RuntimeError("Empty audio response")
        except Exception as err:
            _LOGGER.exception("Chunk synthesis failed: %s", err)
            raise

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType | None:
        """Get TTS audio with concurrent chunking for long texts."""
        if language not in SUPPORTED_LANGUAGES:
            _LOGGER.error("Unsupported language: %s. Using default.", language)
            language = self.default_language

        voice = self._default_voice
        if options and "voice" in options:
            voice = options["voice"]
            if voice not in SUPPORTED_VOICES:
                _LOGGER.error("Unsupported voice: %s. Using default.", voice)
                voice = self._default_voice

        style = options.get("style", "") if options else ""

        # Split text
        chunks = split_text(message)
        if not chunks:
            _LOGGER.error("Empty text after split")
            return None

        _LOGGER.debug("Splitted into %d chunks, voice=%s", len(chunks), voice)

        # If only one chunk, call directly
        if len(chunks) == 1:
            try:
                audio = await self._synthesize_chunk(chunks[0], voice, style)
                return (AUDIO_FORMAT, audio)
            except Exception:
                return None

        # Concurrent requests with semaphore
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def _limited_request(chunk: str) -> bytes:
            async with semaphore:
                return await self._synthesize_chunk(chunk, voice, style)

        tasks = [_limited_request(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors
        successful = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                _LOGGER.error("Chunk %d failed: %s", idx, result)
                return None
            successful.append(result)

        if not successful:
            return None

        # Merge all WAV fragments
        merged = concatenate_wav(successful)
        _LOGGER.debug("Merged %d chunks into %d bytes", len(successful), len(merged))
        return (AUDIO_FORMAT, merged)

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse | None:
        """Stream synthesized audio (legacy compatibility)."""
        # 由于我们实现了并发合并，这里简单调用一次性方法并流式返回单个块
        result = await self.async_get_tts_audio(
            message="".join([chunk async for chunk in request.message_gen]),
            language=request.language,
            options=request.options,
        )
        if result is None:
            return None
        extension, data = result

        async def data_gen() -> AsyncGenerator[bytes]:
            yield data

        return TTSAudioResponse(extension, data_gen())
