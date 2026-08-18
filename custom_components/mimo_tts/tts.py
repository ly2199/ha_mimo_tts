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
from homeassistant.core import HomeAssistant
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

        # 默认语音
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
    def default_voice(self) -> str | None:
        """Return the default voice."""
        return self._default_voice

    @property
    def supported_voices(self) -> list[str] | None:
        """Return a list of supported voices."""
        return list(SUPPORTED_VOICES.keys())

    async def async_get_tts_audio(
        self,
        voice: str,
        language: str,
        message: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType | None:
        """Get TTS audio for the specified text."""
        _LOGGER.debug(
            "TTS request: voice=%s, language=%s, message=%s, options=%s",
            voice,
            language,
            message[:50],
            options,
        )

        # 语音有效性检查
        if voice not in SUPPORTED_VOICES:
            _LOGGER.error("Unsupported voice: %s. Using default.", voice)
            voice = self.default_voice

        if language not in SUPPORTED_LANGUAGES:
            _LOGGER.error("Unsupported language: %s. Using default.", language)
            language = self.default_language

        # 构建消息：user 消息用于风格控制（可留空或使用默认）
        # 这里我们允许用户通过 options 传入风格描述，或留空
        user_content = options.get("style", "") if options else ""

        # 构建 assistant 消息（即要朗读的文本）
        # 根据 Mimo 文档，目标文本放在 assistant 角色中
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
            if (
                completion.choices
                and completion.choices[0].message
                and hasattr(completion.choices[0].message, "audio")
            ):
                audio_data_b64 = completion.choices[0].message.audio.get("data")
                if not audio_data_b64:
                    _LOGGER.error("No audio data in response")
                    return None

                audio_bytes = base64.b64decode(audio_data_b64)
                _LOGGER.debug("Generated audio of %d bytes", len(audio_bytes))
                # 返回 (扩展名, 数据)
                return (AUDIO_FORMAT, audio_bytes)
            else:
                _LOGGER.error("No audio data in response: %s", completion)
                return None

        except Exception as err:
            _LOGGER.exception("TTS generation failed: %s", err)
            return None
