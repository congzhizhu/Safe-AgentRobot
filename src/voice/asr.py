# -*- coding: utf-8 -*-
"""语音识别（ASR）：把机器狗麦克风采集到的音频转成文本。

支持三种实现，用 ``ASR_PROVIDER`` 切换：

- ``bailian``：阿里云百炼 qwen3-asr-flash（dashscope 原生多模态接口，默认，需联网）
- ``funasr``：本地 FunASR（Paraformer / SenseVoice），适合离线部署
- ``whisper``：本地 openai-whisper / faster-whisper，适合离线部署
- ``mock``：不做识别，返回空字符串（用于代码自测）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import VoiceConfig

__all__ = ["SpeechRecognizer", "transcribe_file", "ASRError"]


class ASRError(RuntimeError):
    """语音识别失败。"""


class SpeechRecognizer:
    """语音识别器。"""

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig.from_env()
        self._local_model: Any = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        sample_rate: int | None = None,
        language: str | None = None,
    ) -> str:
        """把音频转成文本。

        参数
        ----
        audio:
            音频字节，或音频文件路径。
        sample_rate:
            音频字节模式下的采样率（百炼接口需要 wav 头，传路径更稳）。
        language:
            语言代码，默认取配置里的 ``asr_language``。
        """
        lang = language or self.config.asr_language
        provider = (self.config.asr_provider or "bailian").lower()

        if provider == "bailian":
            return self._bailian(audio, sample_rate=sample_rate, language=lang)
        if provider == "funasr":
            return self._funasr(audio, language=lang)
        if provider == "whisper":
            return self._whisper(audio, language=lang)
        if provider == "mock":
            return ""
        raise ASRError(f"未知的 ASR_PROVIDER: {provider}")

    # ------------------------------------------------------------------
    # 阿里云百炼（qwen3-asr-flash，dashscope 原生多模态接口）
    # ------------------------------------------------------------------
    def _bailian(
        self,
        audio: bytes | str | Path,
        *,
        sample_rate: int | None,
        language: str,
    ) -> str:
        try:
            from dashscope import MultiModalConversation  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ASRError("未安装 dashscope，请执行：pip install dashscope") from exc

        api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ASRError("缺少 API Key：请在 .env 中配置 QWEN_API_KEY")

        # qwen3-asr-flash 走多模态接口，需要传 file:// 绝对路径
        if isinstance(audio, (str, Path)):
            path = Path(audio).resolve()
        else:
            path = self._ensure_file(audio)

        audio_uri = "file://" + str(path).replace("\\", "/")
        messages = [{"role": "user", "content": [{"audio": audio_uri}]}]

        kwargs: dict[str, Any] = {"result_format": "message"}
        if language and language not in ("auto", "*"):
            kwargs["asr_options"] = {"language": language}

        try:
            resp = MultiModalConversation.call(
                model=self.config.asr_model,
                messages=messages,
                api_key=api_key,
                **kwargs,
            )
        except Exception as exc:
            raise ASRError(f"百炼 ASR 请求异常：{exc}") from exc

        if resp.status_code != 200:
            raise ASRError(f"百炼 ASR 请求失败（{resp.status_code}）：{resp.message}")

        try:
            content = resp.output.choices[0].message.content
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    return item["text"].strip()
        except (AttributeError, IndexError, TypeError):
            pass
        return ""

    # ------------------------------------------------------------------
    # 本地离线方案
    # ------------------------------------------------------------------
    def _funasr(self, audio: bytes | str | Path, *, language: str) -> str:
        path = self._ensure_file(audio)
        try:
            from funasr import AutoModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ASRError(
                "未安装 funasr，请执行：pip install funasr modelscope torchaudio"
            ) from exc

        if self._local_model is None:
            self._local_model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
        result = self._local_model.generate(input=str(path), language=language)
        return _flatten_funasr(result)

    def _whisper(self, audio: bytes | str | Path, *, language: str) -> str:
        path = self._ensure_file(audio)
        try:
            import whisper  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ASRError("未安装 openai-whisper，请执行：pip install -U openai-whisper") from exc

        if self._local_model is None:
            self._local_model = whisper.load_model("small")
        result = self._local_model.transcribe(str(path), language=language)
        return str(result.get("text", "")).strip()

    @staticmethod
    def _ensure_file(audio: bytes | str | Path) -> Path:
        if isinstance(audio, (str, Path)):
            return Path(audio)
        tmp_dir = Path(os.getenv("SAFETY_AGENT_TMP", "")) if os.getenv("SAFETY_AGENT_TMP") else Path(os.path.expanduser("~")) / ".safety_agent_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / "asr_input.wav"
        tmp.write_bytes(_as_wav(audio))
        return tmp


def _flatten_funasr(result: Any) -> str:
    """把 FunASR 的嵌套返回拍平成纯文本。"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("text", "transcription"):
            value = result.get(key)
            if value:
                return _flatten_funasr(value)
        return ""
    if isinstance(result, list):
        return " ".join(part for part in (_flatten_funasr(item) for item in result) if part).strip()
    return str(result).strip()


def _as_wav(raw: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """给裸 PCM 加一个 wav 头，方便接口识别。"""
    import wave
    import io

    if raw[:4] == b"RIFF":
        return raw

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)
    return buffer.getvalue()


def transcribe_file(
    audio_path: str | Path, *, config: VoiceConfig | None = None, language: str | None = None
) -> str:
    """便捷函数：直接识别一个音频文件。"""
    return SpeechRecognizer(config).transcribe(audio_path, language=language)
