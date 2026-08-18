# Mimo Text-to-Speech Integration for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![ha_version](https://img.shields.io/badge/Home%20Assistant-2024.1+-blue.svg)](https://www.home-assistant.io/)
[![GitHub release](https://img.shields.io/github/v/release/ly2199/ha_mimo_tts)](https://github.com/ly2199/ha_mimo_tts/releases)

使用 Mimo TTS 服务为 Home Assistant 提供高质量文本转语音（TTS）能力，支持多种预置音色。

## ✨ 特性

- 基于 [Mimo TTS](https://mimo.mi.com/docs/zh-CN/quick-start/usage-guide/audio/Speech-Synthesis) 云端引擎
- 支持多种精品预置音色（中文/英文）
- 支持自然语言风格控制（通过 `options` 传递）
- 低延迟、高自然度
- 仅需 API Key，配置简单

## 📋 前提条件

- Home Assistant 2024.1 或更高版本
- 有效的 [Mimo API Key](https://mimo.mi.com/)

## 📦 安装

### HACS（推荐）
1. 添加自定义仓库：`https://github.com/ly2199/ha_mimo_tts`（类别：Integration）
2. 下载并重启 HA

### 手动
将 `custom_components/mimo_tts/` 放入你的 `custom_components/` 目录，重启 HA。

## ⚙️ 配置

1. **设置 → 设备与服务 → 添加集成** → 搜索 “Mimo Text-to-Speech”
2. 输入你的 API Key。
3. 完成。

## 🗣️ 使用

在自动化或语音助手中调用 `tts.speak` 服务，或选择 Mimo 作为默认 TTS 引擎：

```yaml
service: tts.speak
data:
  entity_id: tts.mimo_text_to_speech
  message: "你好，欢迎使用 Mimo TTS"
  language: zh
  options:
    voice: "冰糖"
    style: "温柔亲切，语速稍慢"
```
支持的音色列表
| 音色名 | Voice ID | 语言 | 性别 |
| ---- | ---- | ---- | ----|
| MiMo-默认 | mimo_default |	中文/英文 |	女性/男性|
| 冰糖 |	冰糖	| 中文	| 女性 |
| 茉莉 |	茉莉 |	中文	|	女性 |
| 苏打 |	苏打 |	中文	|	男性 |
| 白桦 |	白桦 |	中文	|	男性 |
| Mia |	Mia |	英文	|	女性 |
| Chloe |	Chloe |	英文	|	女性 |
| Milo |	Milo |	英文	|	男性 |
| Dean |	Dean |	英文	|	男性 |
## 风格控制（可选）
在 options 中传入 style 字段，例如：

```yaml
options:
  style: "用轻快上扬的语调，语速稍快，带着激动和喜悦"
```
🔧 支持参数
| 项目 |	详情 |
| ---- | ---- |
| 音频格式 |	WAV (16kHz, 16-bit mono) |
| 语言 |	zh, zh-CN, en |
| 流式支持 |	否（本次实现非流式，适合 HA 播放）|
>如需流式或更多高级功能（音色设计、克隆），欢迎提交 Issue。

❓ 故障排除
1. TTS 无声音
 - 检查 API Key 是否有效，余额是否充足。
 - 查看日志：logger.default=debug 观察 custom_components.mimo_tts 的日志。

2. 音色不存在
 - 使用支持的音色列表中的 ID，不存在的将自动回退到默认。
3. 阻塞警告
 - 本插件已在客户端创建时使用 executor 避免阻塞，若出现警告请提交 Issue。

Enjoy your voice! 🎙️
