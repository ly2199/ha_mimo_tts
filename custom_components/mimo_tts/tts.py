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
)

_LOGGER = logging.getLogger(__name__)

# 文本分割参数
MAX_CHUNK_CHARS = 500
SENTENCE_DELIMITERS = "。！？!?；;\n\r…"
SUB_DELIMITERS = "，,、：:"

# Mimo 流式返回 PCM16，采样率 24kHz
STREAM_SAMPLE_RATE = 24000
STREAM_CHANNELS = 1
STREAM_BITS_PER_SAMPLE = 16


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """按句子边界分割长文本，确保每段不超过 max_chars。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
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


def pcm_to_wav(pcm_data: bytes, sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    """将 PCM 数据转换为标准 WAV 格式（带头部）。"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)
    header = bytearray()
    # RIFF header
    header += b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    # fmt chunk
    header += b"fmt "
    header += struct.pack("<I", 16)  # fmt chunk size
    header += struct.pack("<H", 1)   # audio format (PCM)
    header += struct.pack("<H", channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)
    # data chunk
    header += b"data"
    header += struct.pack("<I", data_size)
    return bytes(header) + pcm_data


def concatenate_wav(wav_chunks: list[bytes]) -> bytes:
    """合并多个 WAV 文件为单个完整 WAV（提取 PCM 数据并重新打包头部）。"""
    if not wav_chunks:
        return b""
    if len(wav_chunks) == 1:
        return wav_chunks[0]

    # 提取每个 WAV 的 PCM 数据（跳过头部）
    pcm_parts = []
    total_pcm_len = 0
    for wav in wav_chunks:
        # 查找 "data" 块
        data_idx = wav.find(b"data")
        if data_idx == -1:
            # 如果没有找到，可能是纯 PCM，直接使用
            pcm_parts.append(wav)
            total_pcm_len += len(wav)
            continue
        # 读取 data 块大小
        size = struct.unpack("<I", wav[data_idx + 4 : data_idx + 8])[0]
        pcm = wav[data_idx + 8 : data_idx + 8 + size]
        pcm_parts.append(pcm)
        total_pcm_len += len(pcm)

    # 构建新 WAV 头（使用第一个文件的参数）
    first = wav_chunks[0]
    # 检查 RIFF 和 fmt 块
    riff_idx = first.find(b"RIFF")
    if riff_idx == -1:
        # 不是标准 WAV，直接拼接（保险）
        return b"".join(wav_chunks)

    # 复制 fmt 块（通常在偏移 12 开始）
    fmt_start = first.find(b"fmt ")
    if fmt_start == -1:
        return b"".join(wav_chunks)

    # fmt 块大小通常为 16
    fmt_size = struct.unpack("<I", first[fmt_start + 4 : fmt_start + 8])[0]
    fmt_data = first[fmt_start : fmt_start + 8 + fmt_size]

    # 构建新 WAV
    header = bytearray()
    # RIFF header
    header += b"RIFF"
    header += struct.pack("<I", 36 + total_pcm_len)
    header += b"WAVE"
    header += fmt_data
    # data chunk
    header += b"data"
    header += struct.pack("<I", total_pcm_len)

    # 拼接 PCM
    return bytes(header) + b"".join(pcm_parts)


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

    async def _synthesize_chunk(
        self, message: str, voice: str, style: str
    ) -> bytes:
        """合成单个文本块，使用流式 API 收集 PCM 数据，返回完整 WAV。"""
        client = await self._async_get_client()
        messages = []
        if style:
            messages.append({"role": "user", "content": style})
        messages.append({"role": "assistant", "content": message})

        try:
            # 使用流式调用，指定 pcm16 格式（无头）
            stream = await client.chat.completions.create(
                model=MIMO_TTS_MODEL,
                messages=messages,
                audio={"format": "pcm16", "voice": voice},
                stream=True,
            )

            pcm_data = bytearray()
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                audio_data = getattr(delta, "audio", None)
                if audio_data is not None:
                    # audio_data 可能是 dict 或对象
                    if isinstance(audio_data, dict):
                        pcm_b64 = audio_data.get("data")
                    else:
                        pcm_b64 = getattr(audio_data, "data", None)
                    if pcm_b64:
                        pcm_bytes = base64.b64decode(pcm_b64)
                        pcm_data.extend(pcm_bytes)

            if not pcm_data:
                _LOGGER.error("No PCM data received for chunk: %s", message[:30])
                return b""

            # 转换为 WAV
            wav = pcm_to_wav(
                bytes(pcm_data),
                sample_rate=STREAM_SAMPLE_RATE,
                channels=STREAM_CHANNELS,
                bits_per_sample=STREAM_BITS_PER_SAMPLE,
            )
            _LOGGER.debug("Chunk synthesized: %d PCM bytes -> %d WAV bytes",
                          len(pcm_data), len(wav))
            return wav

        except Exception as err:
            _LOGGER.exception("Chunk synthesis failed: %s", err)
            return b""

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

        # 分割文本
        chunks = split_text(message)
        if len(chunks) > 1:
            _LOGGER.debug("Split into %d chunks for concurrent synthesis", len(chunks))

        # 并发合成所有块
        tasks = [self._synthesize_chunk(chunk, voice, style) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        valid_results = [r for r in results if r]

        if not valid_results:
            _LOGGER.error("All chunks failed")
            return None

        # 合并为单个 WAV
        merged = concatenate_wav(valid_results)
        if not merged:
            return None

        _LOGGER.debug("Final audio size: %d bytes", len(merged))
        return ("wav", merged)

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse | None:
        """流式接口（内部使用并发合成，然后一次性返回完整 WAV）。"""
        # 收集完整消息
        message_parts = []
        async for chunk in request.message_gen:
            message_parts.append(chunk)
        message = "".join(message_parts)
        if not message:
            return None

        # 使用 async_get_tts_audio 获得完整 WAV
        result = await self.async_get_tts_audio(
            message=message,
            language=request.language,
            options=request.options,
        )
        if not result:
            return None
        extension, audio_data = result

        async def data_gen() -> AsyncGenerator[bytes]:
            yield audio_data

        return TTSAudioResponse(extension, data_gen())
