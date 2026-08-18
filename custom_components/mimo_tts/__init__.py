"""The Mimo TTS integration."""
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.TTS]


async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Set up Mimo TTS from a config entry."""
    _LOGGER.info("Setting up Mimo TTS integration for entry: %s", config_entry.entry_id)

    await hass.config_entries.async_forward_entry_setups(config_entry, PLATFORMS)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][config_entry.entry_id] = config_entry.data

    _LOGGER.info("Mimo TTS integration setup completed successfully.")
    return True


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.info("Unloading Mimo TTS integration")

    if DOMAIN in hass.data:
        hass.data[DOMAIN].pop(config_entry.entry_id, None)

    return await hass.config_entries.async_unload_platforms(config_entry, PLATFORMS)
