# -*- coding: utf-8 -*-
"""音频输入输出：麦克风录音（含静音端点检测）与扬声器播放。

依赖都是可选的，按需导入：

- 录音：``sounddevice``（推荐）或 ``pyaudio``
- 播放：``soundfile`` + ``sounddevice``（推荐）、``winsound``（Windows wav）、
  ``ffplay``（FFmpeg）作为兜底
- 端点检测：默认用能量法（无额外依赖）；装了 ``webrtcvad`` 时优先用它，
  在车间噪声环境下更稳
"""
from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = [
    "record_seconds",
    "record_until_silence",
    "record_stream",
    "to_wav_bytes",
    "play_audio",
    "save_audio",
    "SentenceSplitter",
    "AudioIOError",
]


def save_audio(audio: bytes, fmt: str = "mp3", *, directory: str | Path | None = None) -> Path:
    """把音频字节存成临时文件，返回路径（便于 Gradio / HTTP 播放）。"""
    if not audio:
        raise AudioIOError("音频内容为空，无法保存")

    base = Path(directory) if directory else Path(tempfile.gettempdir()) / "safety_agent_voice"
    base.mkdir(parents=True, exist_ok=True)

    suffix = fmt if fmt.startswith(".") else f".{fmt}"
    fd, path_str = tempfile.mkstemp(suffix=suffix, dir=str(base))
    with os.fdopen(fd, "wb") as handle:
        handle.write(audio)
    return Path(path_str)


class AudioIOError(RuntimeError):
    """音频设备操作失败。"""


# ----------------------------------------------------------------------
# 录音
# ----------------------------------------------------------------------
def _open_input_stream(sample_rate: int, channels: int, blocksize: int = 1024):
    """优先用 sounddevice，其次 pyaudio。返回 (read_fn, close_fn)。"""
    try:
        import sounddevice as sd  # type: ignore

        stream = sd.RawInputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
        )
        stream.start()

        def read(n: int):
            data, _ = stream.read(n)
            return data

        return read, stream.stop
    except Exception:
        pass

    try:
        import pyaudio  # type: ignore

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=blocksize,
        )

        def read(n: int):
            return stream.read(n, exception_on_overflow=False)

        def close():
            stream.stop_stream()
            stream.close()
            pa.terminate()

        return read, close
    except Exception as exc:
        raise AudioIOError(
            "未找到可用的录音后端，请执行：pip install sounddevice（或 pip install pyaudio）"
        ) from exc


def _rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    data = block.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(data * data)))


def record_seconds(seconds: float = 5.0, sample_rate: int = 16000, channels: int = 1) -> np.ndarray:
    """固定时长录音，返回 int16 一维数组。"""
    read, close = _open_input_stream(sample_rate, channels)
    frames: list[np.ndarray] = []
    try:
        blocksize = 1024
        total_blocks = max(1, int(seconds * sample_rate / blocksize))
        for _ in range(total_blocks):
            raw = read(blocksize)
            frames.append(np.frombuffer(raw, dtype=np.int16))
    finally:
        close()
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)


def record_stream(sample_rate: int = 16000, channels: int = 1) -> Iterator[np.ndarray]:
    """持续产出录音数据块（用于自定义端点检测或唤醒逻辑）。"""
    read, close = _open_input_stream(sample_rate, channels)
    try:
        while True:
            raw = read(1024)
            yield np.frombuffer(raw, dtype=np.int16)
    finally:
        close()


def record_until_silence(
    sample_rate: int = 16000,
    channels: int = 1,
    start_threshold: float = 0.045,
    end_threshold: float = 0.028,
    silence_seconds: float = 0.75,
    min_seconds: float = 0.6,
    max_seconds: float = 15.0,
    idle_seconds: float = 8.0,
) -> np.ndarray:
    """说话自动开始、静音自动结束的录音。

    返回 int16 一维数组；若始终没人说话，返回空数组。
    """
    read, close = _open_input_stream(sample_rate, channels)
    frames: list[np.ndarray] = []
    blocksize = 1024
    block_duration = blocksize / float(sample_rate)

    started = False
    silence_acc = 0.0
    elapsed = 0.0
    idle_acc = 0.0

    try:
        while elapsed < max_seconds:
            raw = read(blocksize)
            block = np.frombuffer(raw, dtype=np.int16)
            level = _rms(block)
            elapsed += block_duration

            if not started:
                if level >= start_threshold:
                    started = True
                    frames.append(block)
                    silence_acc = 0.0
                    idle_acc = 0.0
                else:
                    idle_acc += block_duration
                    if idle_acc >= idle_seconds:
                        break
                continue

            frames.append(block)
            if level < end_threshold:
                silence_acc += block_duration
                if silence_acc >= silence_seconds and elapsed >= min_seconds:
                    break
            else:
                silence_acc = 0.0
    finally:
        close()

    if not frames:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(frames)


# ----------------------------------------------------------------------
# 格式转换
# ----------------------------------------------------------------------
def to_wav_bytes(frames: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """把 int16 数组编码成 wav 字节。"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames.astype(np.int16).tobytes())
    return buffer.getvalue()


# ----------------------------------------------------------------------
# 播放
# ----------------------------------------------------------------------
def play_audio(audio: bytes, fmt: str = "mp3") -> None:
    """播放音频字节（阻塞直到播完）。

    优先 soundfile + sounddevice；失败则写临时文件调用系统播放器。
    """
    if not audio:
        return

    fmt = (fmt or "mp3").lower()

    # 方案一：soundfile + sounddevice（支持 mp3/wav/ogg 等）
    try:
        import sounddevice as sd  # type: ignore
        import soundfile as sf  # type: ignore

        data, samplerate = sf.read(io.BytesIO(audio), dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
        return
    except Exception:
        pass

    # 方案二：临时文件 + 系统播放器
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        tmp.write(audio)
        tmp_path = Path(tmp.name)

    try:
        system = platform.system().lower()
        if system == "windows":
            if _which("ffplay"):
                subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(tmp_path)],
                               check=False)
            else:
                import winsound

                if fmt == "wav":
                    winsound.PlaySound(str(tmp_path), winsound.SND_FILENAME)
                else:
                    os.startfile(str(tmp_path))  # noqa: S606 - 兜底：调用默认播放器
        elif system == "darwin":
            subprocess.run(["afplay", str(tmp_path)], check=False)
        else:
            if _which("ffplay"):
                subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(tmp_path)],
                               check=False)
            elif _which("aplay"):
                subprocess.run(["aplay", str(tmp_path)], check=False)
            else:
                raise AudioIOError("未找到可用的播放器（ffplay / aplay / afplay）")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


# ----------------------------------------------------------------------
# 流式断句
# ----------------------------------------------------------------------
class SentenceSplitter:
    """把模型吐出的增量文本攒成一句一句，用于边生成边朗读。

    只在句末标点、换行，或者实在太长了才切一刀，
    避免把「第 3.2 条」这种中间带数字点的内容误切成两句。
    """

    # 句末标点
    _END_PUNCT = "。！？；!?\n"
    # 用于兜底长句切分的次级标点
    _SOFT_PUNCT = "，,、:："

    def __init__(self, max_chars: int = 120) -> None:
        self.max_chars = max_chars
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """喂入增量文本，返回本次新切出的完整句子列表。"""
        self._buffer += text or ""
        return self._split(force=False)

    def flush(self) -> list[str]:
        """输出结束时调用，把残留内容一并吐出。"""
        return self._split(force=True)

    def _split(self, *, force: bool) -> list[str]:
        sentences: list[str] = []
        buf = self._buffer

        while buf:
            # 找最近的句末标点
            cut = -1
            for i, ch in enumerate(buf):
                if ch in self._END_PUNCT:
                    cut = i
                    break

            # 没找到句末标点，但已经太长了：退而求其次在逗号处切
            if cut < 0 and len(buf) >= self.max_chars:
                for i, ch in enumerate(buf):
                    if ch in self._SOFT_PUNCT:
                        cut = i
                        break

            if cut >= 0:
                piece = buf[: cut + 1].strip()
                buf = buf[cut + 1 :].lstrip()
                if piece:
                    sentences.append(piece)
                continue

            # 实在没有可切的位置
            break

        if force and buf.strip():
            sentences.append(buf.strip())
            buf = ""

        self._buffer = buf
        return sentences
