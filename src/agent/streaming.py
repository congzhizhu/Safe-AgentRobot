# -*- coding: utf-8 -*-
"""流式推理支撑模块：让智能体「边思考边生成」。

背景
----
原流程是阻塞式的：

    模型生成完整 JSON -> 一次性 json.loads -> 一次性渲染

这在有图片的隐患分析场景里体验很差，一次回答往往几千 token，
用户要盯着空白界面等十几秒甚至更久。

本模块把链路拆成两段增量能力：

1. ``iter_sse_deltas``  —— 把 OpenAI 兼容接口返回的 SSE 字节流转成文本增量；
2. ``StreamingJSONParser`` —— 边喂文本增量，边产出
      - ``field_start``：某个顶层字段开始输出（如「思维链」）
      - ``field_text`` ：该字段**当前已生成的可读文本**（可用于实时上屏）
      - ``field_done`` ：该字段写完了，给出 ``json.loads`` 后的结构化值

这样「思维链」可以在模型还在推理时就逐字上屏，
后面的「隐患定性 / 法律依据 / 整改措施」等字段则一个接一个地补全。

设计取舍
--------
- 只解析**顶层**字段：足够支撑展示，且不需要完整实现 JSON 解析器；
- 容错优先：中途 JSON 残缺、模型输出多余废话、带代码围栏、中文全角标点，
  都不会抛异常，能解析多少算多少；
- 不依赖任何第三方库。
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Iterator

__all__ = [
    "strip_code_fence",
    "extract_text_fragment",
    "format_partial_chain",
    "iter_sse_deltas",
    "StreamingJSONParser",
]

# JSON 字符串里的转义还原表
_UNESCAPE: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    '"': '"',
    "\\": "\\",
    "/": "/",
}

# 去掉 ```json ... ``` 这类代码围栏
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$")


def strip_code_fence(text: str) -> str:
    """去掉包裹在内容外层的 markdown 代码围栏。"""
    return _FENCE_RE.sub("", text or "")


def extract_text_fragment(fragment: str) -> str:
    """从（可能残缺的）JSON 片段中抽取给人看的文字。

    模型正在输出 ``[{"步骤": "视觉分析", "内容": "脚手架未..."}]`` 时，
    我们只想把「视觉分析 / 脚手架未...」这类内容上屏，
    而不是把 ``"``、``{``、``,`` 这些结构符号一起显示出来。
    """
    out: list[str] = []
    in_str = False
    esc = False

    for ch in fragment or "":
        if in_str:
            if esc:
                out.append(_UNESCAPE.get(ch, ch))
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            else:
                out.append(ch)
            continue

        if ch == '"':
            in_str = True
        elif ch in "{[":
            out.append("\n")
        elif ch in "}]":
            # 结构符号直接丢弃
            continue
        elif ch == ",":
            out.append("\n")
        elif ch == ":":
            out.append("：")
        elif ch in " \t\r\n":
            out.append(ch)
        else:
            out.append(ch)

    text = "".join(out)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"：[ \t]+", "：", text)  # 「条款号」: "xxx" 冒号后的空格收掉
    return text.strip()


_STEP_LINE_RE = re.compile(r"^\s*步骤\s*[:：]\s*(.+?)\s*$")
_CONTENT_LINE_RE = re.compile(r"^\s*内容\s*[:：]\s*(.*)$")


def format_partial_chain(text: str) -> str:
    """把 ``步骤：xxx\\n内容：yyy`` 整理成 ``**xxx**：yyy`` 这样更好看的流式片段。

    模型正在生成「思维链」数组时，抽取出来的是一行行 ``步骤/内容`` 键值对，
    直接上屏比较难读，这里做一次轻量排版。
    """
    lines = [ln for ln in (text or "").split("\n") if ln.strip()]
    out: list[str] = []
    pending_step: str | None = None

    for line in lines:
        m_content = _CONTENT_LINE_RE.match(line)
        if m_content:
            content = m_content.group(1).strip()
            if pending_step:
                out.append(f"**{pending_step}**：{content}")
                pending_step = None
            elif content:
                out.append(content)
            continue

        m_step = _STEP_LINE_RE.match(line)
        if m_step:
            pending_step = m_step.group(1).strip()
            continue

        stripped = line.strip()
        if stripped:
            if pending_step:
                out.append(f"**{pending_step}**：{stripped}")
                pending_step = None
            else:
                out.append(stripped)

    if pending_step:
        out.append(f"**{pending_step}**：")

    return "\n\n".join(out)


def iter_sse_deltas(lines: Iterable[bytes], encoding: str = "utf-8") -> Iterator[str]:
    """把 SSE 行序列解析成文本增量。

    期望格式（OpenAI / 百炼兼容模式）：

        data: {"choices":[{"delta":{"content":"你"}}]}
        data: {"choices":[{"delta":{"content":"好"}}]}
        data: [DONE]

    非 data 行、心跳行、``[DONE]`` 都会被安全忽略。
    """
    for raw in lines:
        if raw is None:
            continue
        if isinstance(raw, bytes):
            line = raw.decode(encoding, errors="replace")
        else:
            line = str(raw)
        line = line.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue

        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            chunk = json.loads(payload)
        except Exception:
            continue

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            yield str(content)
            continue

        # 少数实现会把整段放在 message 里
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content:
            yield str(content)


class StreamingJSONParser:
    """增量 JSON 解析器：只跟踪顶层字段，边喂边吐事件。

    用法::

        parser = StreamingJSONParser()
        for chunk in iter_sse_deltas(resp.iter_lines()):
            for event in parser.feed(chunk):
                handle(event)

    事件类型：

    - ``{"type": "field_start", "key": "思维链"}``
    - ``{"type": "field_text", "key": "思维链", "text": "目前已生成的可读文本"}``
    - ``{"type": "field_done", "key": "思维链", "value": <解析后的值>}``
    """

    def __init__(self) -> None:
        self._buf = ""
        self._pos = 0
        self._started = False

        # 扫描状态
        self._depth = 0
        self._in_string = False
        self._escape = False

        # 顶层字段状态
        self._key_start = -1
        self._key: str | None = None
        self._value_start = -1
        self._expect_key = True
        self._emitted_len = 0

        self.fields: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 主要入口
    # ------------------------------------------------------------------
    @property
    def started(self) -> bool:
        """是否已经在输出流中定位到 JSON 起点。"""
        return self._started

    @property
    def current_key(self) -> str | None:
        """当前正在生成的顶层字段名（未开始解析时为 None）。"""
        return self._key

    def feed(self, chunk: str | None) -> list[dict[str, Any]]:
        """喂入一段文本增量，返回本次产生的事件列表。"""
        events: list[dict[str, Any]] = []
        if not chunk:
            return events

        self._buf += chunk

        # 还没找到 JSON 起点：跳过模型可能吐出的前缀废话 / 代码围栏
        if not self._started:
            head = strip_code_fence(self._buf.lstrip())
            idx = head.find("{")
            if idx == -1:
                # 连 '{' 都还没出现，继续攒
                if len(self._buf) > 4096:  # 防止无限堆积
                    self._buf = self._buf[-512:]
                return events
            self._buf = head[idx:]
            self._pos = 0
            self._started = True
            events.append({"type": "json_start"})
            self._depth = 0
            self._in_string = False
            self._escape = False
            self._expect_key = True
            self._key = None
            self._value_start = -1
            self._emitted_len = 0

        self._scan(events)
        return events

    def finish(self) -> list[dict[str, Any]]:
        """模型输出结束，冲刷残留的未闭合字段。"""
        events: list[dict[str, Any]] = []
        if self._started and self._key is not None and self._value_start >= 0:
            self._complete_field(self._key, self._buf[self._value_start :].strip(), events)
        return events

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _scan(self, events: list[dict[str, Any]]) -> None:
        buf = self._buf
        n = len(buf)

        while self._pos < n:
            ch = buf[self._pos]

            if self._in_string:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                    # 顶层键名结束
                    if self._depth == 1 and self._expect_key:
                        self._key = _unescape(buf[self._key_start : self._pos])
                        self._expect_key = False
                        self._value_start = -1
                        self._emitted_len = 0
                        events.append({"type": "field_start", "key": self._key})
            else:
                if ch == '"':
                    self._in_string = True
                    self._key_start = self._pos + 1
                elif ch in "{[":
                    self._depth += 1
                    # 顶层字段的值是一个对象/数组：从这里开始
                    if self._depth == 2 and self._key is not None and self._value_start < 0:
                        self._value_start = self._pos
                elif ch in "}]":
                    self._depth -= 1
                    if self._key is not None and not self._expect_key:
                        if self._depth == 1:
                            # 嵌套对象/数组结束 -> 顶层字段值完整
                            self._complete_field(self._key, buf[self._value_start : self._pos + 1], events)
                        elif self._depth <= 0:
                            # 顶层对象已闭合，兜底冲刷
                            self._complete_field(self._key, buf[self._value_start : self._pos].strip(), events)
                elif ch == "," and self._depth == 1 and not self._expect_key:
                    self._complete_field(self._key, buf[self._value_start : self._pos].strip(), events)
                    self._expect_key = True
                elif ch == ":" and self._depth == 1 and not self._expect_key:
                    self._value_start = self._pos + 1
                    self._emitted_len = 0

            self._pos += 1

        # 正在写的字段：上报其可读文本的增量
        if self._started and self._key is not None and self._value_start >= 0:
            raw = buf[self._value_start :]
            readable = extract_text_fragment(raw)
            if len(readable) < self._emitted_len:
                self._emitted_len = len(readable)
            if len(readable) > self._emitted_len:
                events.append(
                    {"type": "field_text", "key": self._key, "text": readable}
                )
                self._emitted_len = len(readable)

    def _complete_field(self, key: str, raw: str, events: list[dict[str, Any]]) -> None:
        value: Any = None
        text = (raw or "").strip().rstrip(",")
        for candidate in (text, _repair_json_text(text)):
            if not candidate:
                continue
            try:
                value = json.loads(candidate)
                break
            except Exception:
                continue
        else:
            value = extract_text_fragment(text)

        self.fields[key] = value
        events.append({"type": "field_done", "key": key, "value": value})

        # 复位，准备下一个顶层字段
        self._key = None
        self._value_start = -1
        self._expect_key = True
        self._emitted_len = 0


def _unescape(s: str) -> str:
    """还原 JSON 字符串中的转义（键名一般不需要，但稳妥起见）。"""
    if "\\" not in s:
        return s
    out: list[str] = []
    esc = False
    for ch in s:
        if esc:
            out.append(_UNESCAPE.get(ch, ch))
            esc = False
        elif ch == "\\":
            esc = True
        else:
            out.append(ch)
    return "".join(out)


def _repair_json_text(s: str) -> str:
    """修复大模型写 JSON 时最常见的语法错误（含全角标点）。"""
    s = re.sub(r"([}\]])\s*[;；，]\s*", r"\1,", s)
    s = re.sub(r'("\s*)\s*：\s*', r"\1: ", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s
