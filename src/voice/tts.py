# -*- coding: utf-8 -*-
"""语音合成（TTS）：把智能体的文本回答变成机器狗能播报的语音。

支持四种实现，用 ``TTS_PROVIDER`` 切换：

- ``bailian``：阿里云百炼原生 ``/audio/tts/SpeechSynthesizer`` 接口（默认，需联网）
- ``edge``   ：edge-tts，免费、无需 Key、音色自然（需联网）
- ``pyttsx3``：本地离线 TTS，零网络依赖，音色机械但可用
- ``mock``   ：不发声，返回空字节（用于代码自测）
"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from .config import VoiceConfig

__all__ = ["SpeechSynthesizer", "TTSError"]


class TTSError(RuntimeError):
    """语音合成失败。"""


class SpeechSynthesizer:
    """语音合成器。"""

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig.from_env()
        self._engine: Any = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
    ) -> bytes:
        """把文本合成成音频字节。"""
        text = (text or "").strip()
        if not text:
            return b""

        provider = (self.config.tts_provider or "bailian").lower()
        fmt = response_format or self.config.tts_format
        voice_name = voice or self.config.tts_voice

        if provider == "bailian":
            return self._bailian(text, voice=voice_name, response_format=fmt)
        if provider == "edge":
            return self._edge(text, voice=voice_name, response_format=fmt)
        if provider == "pyttsx3":
            return self._pyttsx3(text)
        if provider == "mock":
            return b""
        raise TTSError(f"未知的 TTS_PROVIDER: {provider}")

    # ------------------------------------------------------------------
    # 阿里云百炼（原生 HTTP 接口，返回音频 URL 再下载）
    # ------------------------------------------------------------------
    def _bailian(self, text: str, *, voice: str, response_format: str) -> bytes:
        api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise TTSError("缺少 API Key：请在 .env 中配置 QWEN_API_KEY")

        url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
        headers = {"Authorization": f"Bearer {api_key}"}
        body = {
            "model": self.config.tts_model,
            "input": {
                "text": text,
                "voice": voice,
                "format": response_format or "wav",
                "sample_rate": 24000,
            },
        }

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
        except requests.RequestException as exc:
            raise TTSError(f"百炼 TTS 连接失败：{exc}") from exc

        if resp.status_code != 200:
            raise TTSError(f"百炼 TTS 请求失败（{resp.status_code}）：{resp.text[:300]}")

        payload = resp.json()
        audio_url = (payload.get("output") or {}).get("audio", {}).get("url", "")
        if not audio_url:
            raise TTSError(f"百炼 TTS 未返回音频地址：{str(payload)[:300]}")

        audio = requests.get(audio_url, timeout=60)
        if audio.status_code != 200:
            raise TTSError(f"下载合成音频失败（{audio.status_code}）")
        return audio.content

    # ------------------------------------------------------------------
    # edge-tts（免费，无需 Key）
    # ------------------------------------------------------------------
    def _edge(self, text: str, *, voice: str, response_format: str) -> bytes:
        try:
            import edge_tts  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TTSError("未安装 edge-tts，请执行：pip install edge-tts") from exc

        fallback_voice = voice if voice else "zh-CN-XiaoxiaoNeural"
        if "-" not in fallback_voice:
            fallback_voice = "zh-CN-XiaoxiaoNeural"

        rate = self.config.tts_speed
        rate_str = "+0%"
        if abs(rate - 1.0) > 1e-6:
            percent = int(round((rate - 1.0) * 100))
            rate_str = f"{percent:+d}%"

        async def _run() -> bytes:
            data = b""
            communicate = edge_tts.Communicate(text, fallback_voice, rate=rate_str)
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    data += chunk.get("data", b"")
            return data

        # 在独立线程里跑 asyncio.run，避免与 Gradio / FastAPI 的事件循环冲突
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run_sync, _run).result()

    # ------------------------------------------------------------------
    # 本地离线（pyttsx3）
    # ------------------------------------------------------------------
    def _pyttsx3(self, text: str) -> bytes:
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise TTSError("未安装 pyttsx3，请执行：pip install pyttsx3") from exc

        if self._engine is None:
            self._engine = pyttsx3.init()
            rate = self._engine.getProperty("rate")
            self._engine.setProperty("rate", int(rate * self.config.tts_speed))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self._engine.save_to_file(text, tmp_path)
            self._engine.runAndWait()
            return Path(tmp_path).read_bytes()
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _run_sync(coro_fn) -> bytes:
    """在新建事件循环里执行一个返回 bytes 的协程函数。"""
    import asyncio

    return asyncio.run(coro_fn())
