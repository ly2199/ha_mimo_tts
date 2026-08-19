"""Support for Mimo text-to-speech service."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import struct
import wave
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

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
    CONF_API_KEY,
    MIMO_API_BASE,
    MIMO_TTS_MODEL,
    SUPPORTED_VOICES,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
)

_LOGGER = logging.getLogger(__name__)

# ========== 音频参数 ==========
# 第一句流式使用的小缓冲，降低首字延迟
FIRST_CHUNK_BUFFER_SIZE = 8192

# Mimo PCM16 音频参数（24kHz mono 16-bit，来自官方文档）
MIMO_SAMPLE_RATE = 24000
MIMO_CHANNELS = 1
MIMO_SAMPLE_WIDTH = 2  # 16-bit

# 分句最小长度，避免过短请求
MIN_SENTENCE_LEN = 40
# 后续块合并时的最大字符数（约 200 字符，平衡一致性与延迟）
MAX_CHUNK_LEN = 200
# 块间插入静音时长（毫秒）
SILENCE_DURATION_MS = 250
# 流尾静音时长（毫秒），防止播放器在流结束时截掉句尾
TRAILING_SILENCE_MS = 400
# 瞬态错误重试次数（含首次尝试）
RETRY_ATTEMPTS = 2
# 后续块并发合成的最大并发数（避免触发限流）
MAX_CONCURRENT_CHUNKS = 3


# ========== 辅助函数 ==========
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


def _create_silence(duration_ms: int = SILENCE_DURATION_MS) -> bytes:
    """生成指定时长的静音 PCM16 数据（单声道 16-bit）。"""
    num_samples = int(MIMO_SAMPLE_RATE * duration_ms / 1000)
    return b"\x00\x00" * num_samples


def _split_sentences(text: str) -> list[str]:
    """按句子结束标点分割，合并过短句，确保自然分割。"""
    if not text:
        return []

    # 按中英文句末标点分割，保留标点
    parts = re.split(r'(?<=[。！？!?；;])', text)
    sentences = [p.strip() for p in parts if p.strip()]

    # 合并过短句到前一个句子
    merged = []
    for sent in sentences:
        if merged and len(merged[-1]) < MIN_SENTENCE_LEN:
            merged[-1] += sent
        else:
            merged.append(sent)
    return merged


def _wav_to_pcm16(data: bytes) -> bytes:
    """从 WAV 数据中提取 PCM16 裸数据（去除文件头）。"""
    try:
        with wave.open(io.BytesIO(data), 'rb') as wf:
            # 仅支持 16-bit PCM 单声道
            if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                _LOGGER.warning("WAV format not PCM16 mono, using raw data")
                return data
            return wf.readframes(wf.getnframes())
    except Exception as err:
        _LOGGER.warning("Failed to parse WAV, assuming raw PCM16: %s", err)
        return data


def _is_transient_error(err: Exception) -> bool:
    """判断是否为可重试的瞬态网络错误。"""
    if isinstance(
        err,
        (
            httpx.TimeoutException,
            httpx.TransportError,
            APIConnectionError,
            APITimeoutError,
        ),
    ):
        return True
    # 服务端 5xx 或 429 限流同样可重试
    if isinstance(err, APIStatusError):
        return err.status_code == 429 or 500 <= err.status_code < 600
    return False


def _retry_delay(err: Exception, attempt: int) -> float:
    """计算重试退避延迟（秒）。429 限流使用更长退避。"""
    if isinstance(err, APIStatusError) and err.status_code == 429:
        return 1.0 * (attempt + 1)
    return 0.3 * (attempt + 1)


async def _run_with_retry(
    create_call: Callable[[], Awaitable[Any]],
    attempts: int = RETRY_ATTEMPTS,
) -> Any:
    """执行 API 调用，瞬态网络错误时自动重试。"""
    for attempt in range(attempts):
        try:
            return await create_call()
        except Exception as err:
            if not _is_transient_error(err) or attempt >= attempts - 1:
                raise
            delay = _retry_delay(err, attempt)
            _LOGGER.warning(
                "Transient API error, retrying (%d/%d) in %.1fs: %s",
                attempt + 1,
                attempts,
                delay,
                err,
            )
            await asyncio.sleep(delay)
    return None


# ========== 平台注册 ==========
async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Mimo TTS entity from a config entry."""
    _LOGGER.info("Setting up Mimo TTS entity")
    entity = MimoTTSEntity(hass, config_entry)
    async_add_entities([entity])
    # 预热客户端，避免首次播报时额外支付连接初始化成本
    hass.async_create_background_task(entity._async_get_client())


# ========== TTS 实体 ==========
class MimoTTSEntity(TextToSpeechEntity):
    """Mimo TTS entity."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        self.hass = hass
        self._config_entry = config_entry
        self._api_key: str = config_entry.data[CONF_API_KEY]
        self._client: AsyncOpenAI | None = None
        self._chunk_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)

        self._attr_name = "Mimo Text-to-Speech"
        self._attr_unique_id = f"{config_entry.entry_id}_tts"
        self._default_voice = DEFAULT_VOICE
        self._default_language = DEFAULT_LANGUAGE

        _LOGGER.debug("Mimo TTS entity initialized")

    async def _async_get_client(self) -> AsyncOpenAI:
        """获取带超时设置的 AsyncOpenAI 客户端（惰性创建）。"""
        if self._client is None:
            api_key = self._api_key

            def _create_client():
                return AsyncOpenAI(
                    api_key=api_key,
                    base_url=MIMO_API_BASE,
                    timeout=httpx.Timeout(
                        connect=5.0, read=20.0, write=10.0, pool=5.0
                    ),
                )

            self._client = await self.hass.async_add_executor_job(_create_client)
        return self._client

    # ===== 属性 =====
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

    # ===== 非流式单次合成（整段文本） =====
    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any] | None = None,
    ) -> TtsAudioType | None:
        """单次合成（非流式，返回 (扩展名, 音频数据)）。"""
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

    # ===== 辅助：非流式获取句子/块的 PCM16 裸数据 =====
    async def _async_get_pcm16_for_chunk(
        self,
        text: str,
        voice: str,
        style: str = "",
    ) -> bytes:
        """使用非流式 API 获取一段文本的 PCM16 裸数据（带瞬态重试）。"""
        client = await self._async_get_client()
        messages = []
        if style:
            messages.append({"role": "user", "content": style})
        messages.append({"role": "assistant", "content": text})

        async def _request() -> bytes:
            # 直接请求 pcm16，与首句流式数据同规格，免去 WAV 剥头
            completion = await client.chat.completions.create(
                model=MIMO_TTS_MODEL,
                messages=messages,
                audio={"format": "pcm16", "voice": voice},
            )
            if (
                completion.choices
                and hasattr(completion.choices[0].message, "audio")
                and completion.choices[0].message.audio
            ):
                audio_b64 = completion.choices[0].message.audio.data
                if audio_b64:
                    data = base64.b64decode(audio_b64)
                    if data[:4] == b"RIFF":
                        # 服务端意外返回 WAV 时剥头
                        _LOGGER.warning(
                            "API returned WAV instead of PCM16, stripping header"
                        )
                        return _wav_to_pcm16(data)
                    return data
            return b""

        try:
            async with self._chunk_semaphore:
                return await _run_with_retry(_request)
        except Exception as err:
            _LOGGER.exception("Non-stream TTS for chunk failed: %s", err)
            return b""

    # ===== 辅助：流式获取一段文本的 PCM16 裸数据 =====
    async def _async_stream_pcm16(
        self,
        text: str,
        voice: str,
        style: str = "",
    ) -> AsyncGenerator[bytes, None]:
        """流式获取一段文本的 PCM16 裸数据，失败时回退为非流式全量合成。

        仅在尚未产出任何字节前允许重试/回退；一旦开始产出部分音频，
        为避免重复/错乱音频，中断后直接结束（由调用方决定后续处理）。
        """
        messages = (
            [{"role": "user", "content": style}] if style else []
        ) + [{"role": "assistant", "content": text}]

        yielded_any = False
        for attempt in range(RETRY_ATTEMPTS):
            try:
                client = await self._async_get_client()
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
                    yielded_any = True
                    yield base64.b64decode(audio_b64)
                return
            except Exception as err:
                if yielded_any:
                    _LOGGER.warning("Stream TTS interrupted mid-stream: %s", err)
                    return
                if attempt < RETRY_ATTEMPTS - 1 and _is_transient_error(err):
                    delay = _retry_delay(err, attempt)
                    _LOGGER.warning(
                        "Stream TTS failed, retrying (%d/%d) in %.1fs: %s",
                        attempt + 1,
                        RETRY_ATTEMPTS,
                        delay,
                        err,
                    )
                    await asyncio.sleep(delay)
                    continue
                _LOGGER.warning(
                    "Stream TTS failed, falling back to 1-shot: %s", err
                )
                data = await self._async_get_pcm16_for_chunk(text, voice, style)
                if data:
                    yield data
                return

    # ===== 混合流式主入口 =====
    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse | None:
        """
        混合流式策略：
        1. 单句：流式合成（SSE 小缓冲低首字节延迟），失败自动回退非流式
        2. 多句：第一句流式输出，其余句子合并成块、限流并发合成
        3. 块间插入静音，流尾追加静音防止播放器截断句尾
        4. 整体输出为合法 WAV 流
        """
        _LOGGER.debug(
            "Stream TTS: language=%s, options=%s", request.language, request.options
        )

        # 收集完整消息（Mimo 不支持文本流式输入）
        message_parts = []
        async for chunk in request.message_gen:
            message_parts.append(chunk)
        message = "".join(message_parts)

        if not message:
            _LOGGER.error("Empty message received")
            return None

        language = request.language
        if language not in SUPPORTED_LANGUAGES:
            _LOGGER.warning(
                "Unsupported language %s, falling back to %s",
                language,
                self.default_language,
            )
            language = self.default_language

        voice = self._default_voice
        if request.options and "voice" in request.options:
            voice = request.options["voice"]
            if voice not in SUPPORTED_VOICES:
                _LOGGER.warning("Unsupported voice %s, falling back to default", voice)
                voice = self._default_voice

        style = request.options.get("style", "") if request.options else ""

        # 分句
        sentences = _split_sentences(message)
        if not sentences:
            sentences = [message]  # 保底
        _LOGGER.debug("Split into %d sentences: %s", len(sentences), sentences)

        first_sentence = sentences[0]
        rest_sentences = sentences[1:]

        # 将剩余句子合并成块（每块不超过 MAX_CHUNK_LEN 字符）
        rest_chunks = []
        current_chunk = ""
        for sent in rest_sentences:
            if len(current_chunk) + len(sent) <= MAX_CHUNK_LEN:
                current_chunk += sent
            else:
                if current_chunk:
                    rest_chunks.append(current_chunk)
                current_chunk = sent
        if current_chunk:
            rest_chunks.append(current_chunk)

        # ---- 单句路径：流式合成，低首字节延迟 ----
        if not rest_chunks:
            async def single_stream_generator() -> AsyncGenerator[bytes]:
                header_sent = False
                buffer = bytearray()
                async for pcm in self._async_stream_pcm16(message, voice, style):
                    if not header_sent:
                        yield _create_wav_header()
                        header_sent = True
                    buffer.extend(pcm)
                    while len(buffer) >= FIRST_CHUNK_BUFFER_SIZE:
                        yield bytes(buffer[:FIRST_CHUNK_BUFFER_SIZE])
                        buffer = buffer[FIRST_CHUNK_BUFFER_SIZE:]
                if not header_sent:
                    _LOGGER.error("TTS failed for single sentence")
                    return
                if buffer:
                    yield bytes(buffer)
                yield _create_silence(TRAILING_SILENCE_MS)

            return TTSAudioResponse("wav", single_stream_generator())

        # ---- 多句混合路径 ----
        async def stream_generator() -> AsyncGenerator[bytes]:
            """生成最终 WAV 流：第一句流式 PCM16 + 后续块 PCM16 拼接。"""
            header_sent = False
            first_buffer = bytearray()
            rest_tasks = [
                asyncio.create_task(
                    self._async_get_pcm16_for_chunk(chunk, voice, style)
                )
                for chunk in rest_chunks
            ]

            try:
                # ---- 第一句流式（小缓冲，低首字节延迟）----
                async for pcm in self._async_stream_pcm16(
                    first_sentence, voice, style
                ):
                    if not header_sent:
                        yield _create_wav_header()
                        header_sent = True
                    first_buffer.extend(pcm)
                    while len(first_buffer) >= FIRST_CHUNK_BUFFER_SIZE:
                        yield bytes(first_buffer[:FIRST_CHUNK_BUFFER_SIZE])
                        first_buffer = first_buffer[FIRST_CHUNK_BUFFER_SIZE:]

                if not header_sent:
                    # 第一句流式与非流式兜底均失败：整段一次性合成
                    _LOGGER.warning("Falling back to full 1-shot")
                    pcm_data = await self._async_get_pcm16_for_chunk(
                        message, voice, style
                    )
                    if pcm_data:
                        yield _create_wav_header()
                        yield pcm_data
                        yield _create_silence(TRAILING_SILENCE_MS)
                    else:
                        _LOGGER.error("Fallback TTS failed")
                    return

                if first_buffer:
                    yield bytes(first_buffer)

                # ---- 后续块输出（块间静音）----
                for task in rest_tasks:
                    try:
                        pcm_data = await task
                    except Exception as err:
                        _LOGGER.error("Non-stream chunk task failed: %s", err)
                        pcm_data = b""

                    if pcm_data:
                        yield _create_silence(SILENCE_DURATION_MS)
                        yield pcm_data

                # ---- 流尾静音，防止播放器截断句尾 ----
                yield _create_silence(TRAILING_SILENCE_MS)

            finally:
                for task in rest_tasks:
                    task.cancel()
                if rest_tasks:
                    await asyncio.gather(*rest_tasks, return_exceptions=True)

        return TTSAudioResponse("wav", stream_generator())
