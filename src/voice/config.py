# -*- coding: utf-8 -*-
"""语音模块配置：默认值 + 环境变量覆盖。

所有开关都能在 ``.env`` 里改，不需要动代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class VoiceConfig:
    """语音功能配置。"""

    # ---------------- 语音识别 (ASR) ----------------
    asr_provider: str = "bailian"        # bailian | funasr | whisper | mock
    asr_model: str = "qwen3-asr-flash"
    asr_language: str = "zh"

    # ---------------- 语音合成 (TTS) ----------------
    tts_provider: str = "bailian"        # bailian | edge | pyttsx3 | mock
    tts_model: str = "qwen-audio-3.1-tts-flash"
    tts_voice: str = "longanhuan_v3.1"
    tts_format: str = "wav"              # wav | mp3（原生接口示例用 wav）
    tts_speed: float = 1.0               # 语速倍率，edge / pyttsx3 生效

    # ---------------- 音频采集参数 ----------------
    sample_rate: int = 16000
    channels: int = 1

    # ---------------- 端点检测 (VAD) ----------------
    # 采用能量法，不依赖额外依赖；噪声大的现场可把阈值调高
    vad_start_threshold: float = 0.045   # 超过该 RMS 认为开始说话
    vad_end_threshold: float = 0.028     # 低于该 RMS 累计静音时长达标即结束
    vad_silence_seconds: float = 0.75
    min_record_seconds: float = 0.6
    max_record_seconds: float = 15.0

    # ---------------- 朗读行为 ----------------
    auto_tts: bool = True                # 是否自动朗读回答
    speak_thinking: bool = False         # 是否朗读「思维链」（默认只显示不念）
    speak_reference_cases: bool = False  # 是否朗读「参考案例」
    max_sentence_chars: int = 120        # 流式断句阈值

    # ---------------- 大模型接口（复用主项目配置）----------------
    api_base: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""

    # 唤醒词（持续对话模式下用）
    wake_words: list[str] = field(default_factory=lambda: ["你好小狗", "小安", "在吗"])

    # 机器狗控制接口（HTTP 适配，可选）
    robot_http_base: str = ""             # 例如 http://192.168.12.1:8080
    robot_http_token: str = ""

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        cfg = cls()
        cfg.asr_provider = os.getenv("ASR_PROVIDER", cfg.asr_provider)
        cfg.asr_model = os.getenv("ASR_MODEL", cfg.asr_model)
        cfg.asr_language = os.getenv("ASR_LANGUAGE", cfg.asr_language)

        cfg.tts_provider = os.getenv("TTS_PROVIDER", cfg.tts_provider)
        cfg.tts_model = os.getenv("TTS_MODEL", cfg.tts_model)
        cfg.tts_voice = os.getenv("TTS_VOICE", cfg.tts_voice)
        cfg.tts_format = os.getenv("TTS_FORMAT", cfg.tts_format)
        cfg.tts_speed = _env_float("TTS_SPEED", cfg.tts_speed)

        cfg.sample_rate = _env_int("AUDIO_SAMPLE_RATE", cfg.sample_rate)
        cfg.channels = _env_int("AUDIO_CHANNELS", cfg.channels)

        cfg.vad_start_threshold = _env_float("VAD_START_THRESHOLD", cfg.vad_start_threshold)
        cfg.vad_end_threshold = _env_float("VAD_END_THRESHOLD", cfg.vad_end_threshold)
        cfg.vad_silence_seconds = _env_float("VAD_SILENCE_SECONDS", cfg.vad_silence_seconds)
        cfg.min_record_seconds = _env_float("MIN_RECORD_SECONDS", cfg.min_record_seconds)
        cfg.max_record_seconds = _env_float("MAX_RECORD_SECONDS", cfg.max_record_seconds)

        cfg.auto_tts = _env_flag("AUTO_TTS", cfg.auto_tts)
        cfg.speak_thinking = _env_flag("SPEAK_THINKING", cfg.speak_thinking)
        cfg.speak_reference_cases = _env_flag("SPEAK_REFERENCE_CASES", cfg.speak_reference_cases)
        cfg.max_sentence_chars = _env_int("MAX_SENTENCE_CHARS", cfg.max_sentence_chars)

        cfg.api_base = os.getenv("QWEN_API_BASE", cfg.api_base)
        cfg.api_key = os.getenv("QWEN_API_KEY", cfg.api_key)

        wake = os.getenv("WAKE_WORDS", "")
        if wake.strip():
            cfg.wake_words = [w.strip() for w in wake.replace("，", ",").split(",") if w.strip()]

        cfg.robot_http_base = os.getenv("ROBOT_HTTP_BASE", cfg.robot_http_base)
        cfg.robot_http_token = os.getenv("ROBOT_HTTP_TOKEN", cfg.robot_http_token)
        return cfg


def load_voice_config(dotenv_path: str | Path | None = None) -> VoiceConfig:
    """加载语音配置；可选的 .env 由调用方传入（主启动脚本已加载过，这里不再重复读）。"""
    return VoiceConfig.from_env()
