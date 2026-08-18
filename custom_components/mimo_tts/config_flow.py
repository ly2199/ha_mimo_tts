"""Config flow for Mimo TTS."""
from __future__ import annotations

from typing import Any
import voluptuous as vol
import logging

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.core import callback

from .const import DOMAIN, NAME, CONF_API_KEY

_LOGGER = logging.getLogger(__name__)


class ConfigFlowHandler(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Mimo TTS."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({
                    vol.Required(CONF_API_KEY): str,
                }),
                errors=errors,
                description_placeholders={
                    "docs_url": "https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/audio/Speech-Synthesis"
                }
            )

        # 简单测试 API Key 有效性（可选）
        # 使用 aiohttp 做最小请求测试，避免阻塞
        from homeassistant.helpers.aiohttp_client import async_get_clientsession
        import json

        session = async_get_clientsession(self.hass)
        try:
            headers = {
                "Authorization": f"Bearer {user_input[CONF_API_KEY]}",
                "Content-Type": "application/json",
            }
            # 最小测试请求
            test_payload = {
                "model": "mimo-v2.5-tts",
                "messages": [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "test"}
                ],
                "audio": {"format": "wav", "voice": "mimo_default"},
                "max_tokens": 1
            }
            async with session.post(
                "https://api.xiaomimimo.com/v1/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=10
            ) as resp:
                if resp.status == 401:
                    errors["base"] = "invalid_auth"
                    _LOGGER.error("Invalid API Key")
                elif resp.status != 200:
                    _LOGGER.warning("API test returned status %d", resp.status)
        except Exception as err:
            _LOGGER.warning("API test failed: %s", err)

        if errors:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({
                    vol.Required(CONF_API_KEY): str,
                }),
                errors=errors,
                description_placeholders={
                    "docs_url": "https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/audio/Speech-Synthesis"
                }
            )

        return self.async_create_entry(
            title=NAME,
            data=user_input
        )
