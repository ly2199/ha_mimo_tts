"""Support for Mimo text-to-speech service."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import soundfile as sf
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

# Mimo 流式返回 PCM16，采样率 24kHz
STREAM_SAMPLE_RATE = 24000
STREAM_AUDIO_FORMAT = "pcm16"


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
        """Return a list of supported options."""
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

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType | None:
        """Get TTS audio (1-shot fallback)."""
        _LOGGER.debug(
            "TTS 1-shot: language=%s, message=%s",
            language,
            message[:50],
        )

        if language not in SUPPORTED_LANGUAGES:
            _LOGGER.error("Unsupported language: %s", language)
            language = self.default_language

        voice = self._default_voice
        if options and "voice" in options:
            voice = options["voice"]
            if voice not in SUPPORTED_VOICES:
                _LOGGER.error("Unsupported voice: %s", voice)
                voice = self._default_voice

        user_content = options.get("style", "") if options else ""

        messages = []
        if user_content:
            messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": message})

        client = await self._async_get_client()

        try:
            # 使用流式模式获取音频，但一次性收集所有块
            completion = await client.chat.completions.create(
                model=MIMO_TTS_MODEL,
                messages=messages,
                audio={
                    "format": "wav",
                    "voice": voice,
                },
            )

            if (
                completion.choices
                and completion.choices[0].message
                and hasattr(completion.choices[0].message, "audio")
                and completion.choices[0].message.audio
            ):
                audio_data_b64 = completion.choices[0].message.audio.data
                if not audio_data_b64:
                    _LOGGER.error("No audio data in response")
                    return None

                audio_bytes = base64.b64decode(audio_data_b64)
                _LOGGER.debug("Generated audio of %d bytes", len(audio_bytes))
                return ("wav", audio_bytes)
            else:
                _LOGGER.error("No audio data in response")
                return None

        except Exception as err:
            _LOGGER.exception("TTS generation failed: %s", err)
            return None

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse | None:
        """Stream synthesized audio chunk by chunk using Mimo stream mode."""
        _LOGGER.debug(
            "TTS stream: language=%s, options=%s",
            request.language,
            request.options,
        )

        # 收集完整消息（HA 传入的是 AsyncGenerator）
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

        # 提取语音
        voice = self._default_voice
        if request.options and "voice" in request.options:
            voice = request.options["voice"]
            if voice not in SUPPORTED_VOICES:
                _LOGGER.error("Unsupported voice: %s", voice)
                voice = self._default_voice

        # 提取风格控制
        user_content = request.options.get("style", "") if request.options else ""

        # 构建消息
        messages = []
        if user_content:
            messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": message})

        client = await self._async_get_client()

        # 创建异步生成器，流式返回音频块
        async def audio_stream_generator() -> AsyncGenerator[bytes]:
            """生成流式音频块."""
            try:
                # 使用流式模式，返回 PCM16 格式
                stream = await client.chat.completions.create(
                    model=MIMO_TTS_MODEL,
                    messages=messages,
                    audio={
                        "format": STREAM_AUDIO_FORMAT,  # pcm16
                        "voice": voice,
                    },
                    stream=True,
                )

                # 收集 PCM 块并实时转换为 WAV
                pcm_chunks: list[bytes] = []
                total_pcm = b""

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    audio_data = getattr(delta, "audio", None)

                    if audio_data is not None:
                        # audio_data 是 dict，包含 "data" 字段
                        if isinstance(audio_data, dict):
                            pcm_b64 = audio_data.get("data")
                        else:
                            # 可能是对象，尝试获取 data 属性
                            pcm_b64 = getattr(audio_data, "data", None)

                        if pcm_b64:
                            pcm_bytes = base64.b64decode(pcm_b64)
                            total_pcm += pcm_bytes
                            pcm_chunks.append(pcm_bytes)
                            _LOGGER.debug(
                                "Received PCM chunk: %d bytes, total: %d",
                                len(pcm_bytes),
                                len(total_pcm),
                            )

                if not total_pcm:
                    _LOGGER.error("No PCM data received from stream")
                    return

                # 将 PCM16 转换为 WAV（HA 期望 WAV 格式）
                # 使用 numpy + soundfile 在 executor 中执行
                def _convert_pcm_to_wav(pcm_data: bytes) -> bytes:
                    """将 PCM16 数据转换为 WAV 格式."""
                    try:
                        # 转换为 numpy array (int16)
                        np_pcm = np.frombuffer(pcm_data, dtype=np.int16)
                        # 归一化到 float32 (-1.0 ~ 1.0)
                        np_float = np_pcm.astype(np.float32) / 32768.0

                        # 写入内存 WAV
                        wav_buffer = io.BytesIO()
                        sf.write(
                            wav_buffer,
                            np_float,
                            STREAM_SAMPLE_RATE,
                            format="WAV",
                            subtype="PCM_16",
                        )
                        wav_buffer.seek(0)
                        return wav_buffer.read()
                    except Exception as e:
                        _LOGGER.error("PCM to WAV conversion failed: %s", e)
                        return b""

                # 在 executor 中执行转换（避免阻塞事件循环）
                wav_data = await self.hass.async_add_executor_job(
                    _convert_pcm_to_wav, total_pcm
                )

                if not wav_data:
                    _LOGGER.error("WAV conversion returned empty data")
                    return

                _LOGGER.debug(
                    "Stream complete: %d PCM bytes -> %d WAV bytes",
                    len(total_pcm),
                    len(wav_data),
                )

                # 将完整 WAV 作为单个块返回（HA 会处理播放）
                yield wav_data

            except Exception as err:
                _LOGGER.exception("TTS stream failed: %s", err)
                return

        return TTSAudioResponse("wav", audio_stream_generator())
