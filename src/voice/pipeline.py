# -*- coding: utf-8 -*-
"""语音对话流水线：把 ASR -> 智能体流式推理 -> TTS 串起来。

两条主线：

1. ``ask_stream(question)``：产出与 ``agent.analyze_stream`` 同构的事件流，
   额外加了 ``spoken`` 事件（某一段文本已送去合成朗读），前端可直接消费；
2. ``run_turn()``：一次完整交互（听 -> 识别 -> 思考 -> 朗读），命令行常用。

延迟优化点
----------
- 模型还在生成时就已经开始按句合成与播放（生产者/消费者解耦）；
- 思维链默认只显示不朗读（``SPEAK_THINKING=1`` 可打开）；
- 句子切分用 ``SentenceSplitter``，遇到句末标点立刻出句，
  避免等整段结束才开口。
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .asr import SpeechRecognizer
from .tts import SpeechSynthesizer
from .audio_io import (
    AudioIOError,
    SentenceSplitter,
    play_audio,
    record_seconds,
    record_until_silence,
    to_wav_bytes,
)
from .config import VoiceConfig
from .robot_dog import (
    STATE_ERROR,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_SPEAKING,
    STATE_THINKING,
    ConsoleRobotDog,
    RobotDog,
)

__all__ = ["VoicePipeline", "TurnResult", "Speaker"]


# 默认不朗读的字段（只读结论，不念推理过程）
_DEFAULT_SKIP_FIELDS = ("思维链", "参考案例")


def _resolve_skip_fields(config: VoiceConfig) -> tuple[str, ...]:
    """根据配置决定哪些字段只显示、不朗读。"""
    skip = list(_DEFAULT_SKIP_FIELDS)
    if config.speak_thinking and "思维链" in skip:
        skip.remove("思维链")
    if config.speak_reference_cases and "参考案例" in skip:
        skip.remove("参考案例")
    return tuple(skip)


class Speaker:
    """后台朗读器：送入文本，按句合成并播放，不阻塞生成线程。"""

    def __init__(self, tts: Any, *, fmt: str = "mp3", max_chars: int = 120) -> None:
        self._tts = tts
        self._fmt = fmt
        self._splitter = SentenceSplitter(max_chars)
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="speaker", daemon=True)
        self._thread.start()

    def feed(self, text: str) -> None:
        """喂入增量文本；凑够一句就异步合成播放。"""
        for sentence in self._splitter.feed(text):
            self._queue.put(sentence)

    def flush(self) -> None:
        """生成结束，把残留文本一并送出。"""
        for sentence in self._splitter.flush():
            self._queue.put(sentence)

    def wait(self, timeout: float | None = None) -> None:
        """等待队列播完。"""
        self._queue.join()

    def stop(self) -> None:
        self._queue.put(None)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                audio = self._tts.synthesize(item)
                play_audio(audio, fmt=self._fmt)
            except Exception as exc:
                print(f"[speaker] 朗读失败：{exc}", flush=True)
            finally:
                self._queue.task_done()


@dataclass
class TurnResult:
    """一轮语音交互的结果。"""

    question: str = ""
    raw_answer: str = ""
    structured_output: dict[str, Any] = field(default_factory=dict)
    intent: str = ""


class VoicePipeline:
    """语音交互流水线。"""

    def __init__(
        self,
        agent: Any | None = None,
        config: VoiceConfig | None = None,
        robot: RobotDog | None = None,
        *,
        auto_tts: bool | None = None,
    ) -> None:
        self.config = config or VoiceConfig.from_env()
        if agent is None:
            from agent.workflow import build_agent_from_env

            agent = build_agent_from_env()
        self.agent = agent
        self.robot = robot or ConsoleRobotDog()
        self.asr = SpeechRecognizer(self.config)
        self.tts = SpeechSynthesizer(self.config)
        self._auto_tts = self.config.auto_tts if auto_tts is None else auto_tts

    # ------------------------------------------------------------------
    # 单步能力
    # ------------------------------------------------------------------
    def listen(self, *, seconds: float | None = None) -> bytes:
        """录音，返回 wav 字节。"""
        self._notify(STATE_LISTENING)
        try:
            if seconds:
                frames = record_seconds(seconds, self.config.sample_rate, self.config.channels)
            else:
                frames = record_until_silence(
                    sample_rate=self.config.sample_rate,
                    channels=self.config.channels,
                    start_threshold=self.config.vad_start_threshold,
                    end_threshold=self.config.vad_end_threshold,
                    silence_seconds=self.config.vad_silence_seconds,
                    min_seconds=self.config.min_record_seconds,
                    max_seconds=self.config.max_record_seconds,
                )
        except AudioIOError as exc:
            self._notify(STATE_ERROR, reason=str(exc))
            raise
        if frames.size == 0:
            return b""
        return to_wav_bytes(frames, self.config.sample_rate, self.config.channels)

    def understand(self, audio: bytes | str | Path) -> str:
        """语音识别。"""
        return self.asr.transcribe(audio, sample_rate=self.config.sample_rate)

    # ------------------------------------------------------------------
    # 核心：边思考边生成
    # ------------------------------------------------------------------
    def ask_stream(
        self,
        question: str,
        image_path: str | Path | None = None,
        *,
        auto_tts: bool | None = None,
    ) -> Iterator[dict[str, Any]]:
        """向智能体提问，边生成边产出事件（含朗读事件）。

        事件类型（在 ``agent.analyze_stream`` 基础上增加）：

        - ``{"type": "spoken", "text": "..."}`` 该段文本已送去朗读
        - ``{"type": "error", "message": "..."}`` 出错
        """
        use_tts = self._auto_tts if auto_tts is None else auto_tts
        skip_fields = _resolve_skip_fields(self.config)

        speaker: Speaker | None = None
        if use_tts:
            try:
                speaker = Speaker(self.tts, fmt=self.config.tts_format,
                                  max_chars=self.config.max_sentence_chars)
            except Exception as exc:
                yield {"type": "warn", "message": f"语音合成不可用，仅文本输出：{exc}"}

        self._notify(STATE_THINKING, text=question)

        text_parts: list[str] = []
        pending_plain = ""
        in_json = False
        current_key: str | None = None
        spoken_len = 0

        try:
            for event in self.agent.analyze_stream(image_path=image_path, question=question):
                etype = event.get("type")

                if etype == "json_start":
                    in_json = True
                    pending_plain = ""
                    current_key = None
                    spoken_len = 0

                elif etype == "raw_delta":
                    text_parts.append(event.get("text", ""))
                    if not in_json:
                        # 还没确认是 JSON：先按纯文本处理（greeting / legal_only）
                        pending_plain += event.get("text", "")
                        if len(pending_plain) >= 16 and speaker is not None:
                            speaker.feed(pending_plain)
                            pending_plain = ""

                elif etype == "field_start":
                    in_json = True
                    current_key = event.get("key")
                    spoken_len = 0

                elif etype == "field_text":
                    in_json = True
                    key = event.get("key")
                    if key != current_key:
                        current_key = key
                        spoken_len = 0
                    text = event.get("text") or ""
                    if speaker is not None and key not in skip_fields:
                        if len(text) < spoken_len:
                            spoken_len = 0
                        if len(text) > spoken_len:
                            delta = text[spoken_len:]
                            spoken_len = len(text)
                            speaker.feed(delta)
                            yield {"type": "spoken", "key": key, "text": delta}

                elif etype == "field_done":
                    in_json = True
                    current_key = None
                    spoken_len = 0

                yield event
        except Exception as exc:
            yield {"type": "error", "message": f"推理过程异常：{exc}"}

        if speaker is not None:
            if pending_plain.strip():
                speaker.feed(pending_plain)
            speaker.flush()
            speaker.wait()
            speaker.stop()

        self._notify(STATE_IDLE)

    def ask(self, question: str, image_path: str | Path | None = None) -> TurnResult:
        """阻塞式提问，返回结构化结果（命令行场景好用）。"""
        result = TurnResult(question=question)
        for event in self.ask_stream(question, image_path=image_path):
            if event.get("type") == "raw_delta":
                print(event.get("text", ""), end="", flush=True)
            elif event.get("type") == "done":
                payload = event.get("result", {})
                result.raw_answer = payload.get("raw_answer", "")
                result.structured_output = payload.get("structured_output", {}) or {}
                result.intent = payload.get("intent", "")
        print()
        return result

    # ------------------------------------------------------------------
    # 一轮完整交互 & 持续对话
    # ------------------------------------------------------------------
    def run_turn(
        self,
        *,
        audio: bytes | None = None,
        text: str | None = None,
        image_path: str | Path | None = None,
        listen_seconds: float | None = None,
    ) -> TurnResult:
        """听一次 -> 回答一次。audio 或 text 二选一；都不传则自动录音。"""
        if text:
            question = text
        else:
            wav = audio if audio is not None else self.listen(seconds=listen_seconds)
            if not wav:
                return TurnResult()
            question = self.understand(wav)
            if not question:
                print("（没听清，请再说一次）")
                return TurnResult()

        print(f"\n你：{question}")
        print("智能体：", end="", flush=True)
        return self.ask(question, image_path=image_path)

    def loop(self, *, wake_word: bool = False) -> None:
        """持续对话循环，Ctrl+C 退出。

        wake_word=True 时，需要先说出唤醒词才进入回答流程。
        """
        print("进入语音循环，按 Ctrl+C 退出。")
        while True:
            try:
                if wake_word:
                    self._notify(STATE_IDLE)
                    wav = self.listen()
                    if not wav:
                        continue
                    heard = self.understand(wav)
                    if not any(word in heard for word in self.config.wake_words):
                        continue
                    print("（已唤醒，请说话）")
                    wav = self.listen()
                    if not wav:
                        continue
                    question = self.understand(wav)
                else:
                    self._notify(STATE_IDLE)
                    wav = self.listen()
                    if not wav:
                        continue
                    question = self.understand(wav)

                if not question:
                    continue
                print(f"\n你：{question}")
                print("智能体：", end="", flush=True)
                self.ask(question)
            except KeyboardInterrupt:
                print("\n已退出语音循环。")
                break
            except AudioIOError as exc:
                print(f"[audio] {exc}")
                continue

    # ------------------------------------------------------------------
    def _notify(self, state: str, **payload: Any) -> None:
        try:
            self.robot.on_state(state, **payload)
        except Exception:
            pass
