"""Support for Mimo text-to-speech service."""
from __future__ import annotations

import base64
import logging
from typing import Any

from openai import AsyncOpenAI

from homeassistant.components.tts import (
    TextToSpeechEntity,
    TtsAudioType,
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
        self._attr_unique_id = f"{config_entry.entry_id}"

        # 默认语音和语言
        self._default_voice = DEFAULT_VOICE
        self._default_language = DEFAULT_LANGUAGE

        _LOGGER.debug("Mimo TTS entity initialized")

    async def _async_get_client(self) -> AsyncOpenAI:
        """异步获取客户端，在 executor 中创建以避免阻塞事件循环。"""
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
    def async_get_supported_voices(self, language: str) -> list[str] | None:
        """Return a list of supported voices for a language."""
        # 所有音色对所有语言都可用
        return list(SUPPORTED_VOICES.keys())

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType | None:
        """Get TTS audio for the specified text.

        :param message: Text to synthesize
        :param language: Language code (BCP47)
        :param options: Options dict, may contain 'voice' and 'style'
        """
        _LOGGER.debug(
            "TTS request: language=%s, message=%s, options=%s",
            language,
            message[:50],
            options,
        )

        # 语言检查
        if language not in SUPPORTED_LANGUAGES:
            _LOGGER.error("Unsupported language: %s. Using default.", language)
            language = self.default_language

        # 提取语音（从 options 或使用默认）
        voice = self._default_voice
        if options and "voice" in options:
            voice = options["voice"]
            if voice not in SUPPORTED_VOICES:
                _LOGGER.error("Unsupported voice: %s. Using default.", voice)
                voice = self._default_voice

        # 提取风格控制（可选）
        user_content = options.get("style", "") if options else ""

        # 构建消息
        messages = []
        if user_content:
            messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": message})

        # 获取客户端
        client = await self._async_get_client()

        try:
            completion = await client.chat.completions.create(
                model=MIMO_TTS_MODEL,
                messages=messages,
                audio={
                    "format": AUDIO_FORMAT,
                    "voice": voice,
                },
            )

            # 提取音频数据
            # 根据 Mimo 官方文档，audio 是一个对象，使用 .data 访问
            # https://mimo.mi.com/docs/zh-CN/api/audio/tts
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
                return (AUDIO_FORMAT, audio_bytes)
            else:
                _LOGGER.error("No audio data in response: %s", completion)
                return None

        except Exception as err:
            _LOGGER.exception("TTS generation failed: %s", err)
            return None
