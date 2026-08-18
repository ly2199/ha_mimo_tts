"""Constants for Mimo TTS integration."""
DOMAIN = "mimo_tts"
NAME = "Mimo Text-to-Speech"

CONF_API_KEY = "api_key"

# Mimo API 配置
MIMO_API_BASE = "https://api.xiaomimimo.com/v1"
MIMO_TTS_MODEL = "mimo-v2.5-tts"

# 支持的语音列表（预置音色）
# 根据官方文档: https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/audio/Speech-Synthesis
SUPPORTED_VOICES = {
    "mimo_default": "MiMo-默认",
    "冰糖": "冰糖",
    "茉莉": "茉莉",
    "苏打": "苏打",
    "白桦": "白桦",
    "Mia": "Mia",
    "Chloe": "Chloe",
    "Milo": "Milo",
    "Dean": "Dean",
}

# 默认语音
DEFAULT_VOICE = "mimo_default"

# 支持的语言 (BCP47)
SUPPORTED_LANGUAGES = ["zh", "zh-CN", "en"]
DEFAULT_LANGUAGE = "zh"

# 音频格式 (Mimo 支持 wav, pcm16, mp3)
# 我们选择 wav 以兼容 HA 的 TTS 播放
AUDIO_FORMAT = "wav"
