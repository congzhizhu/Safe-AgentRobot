# -*- coding: utf-8 -*-
"""机器狗适配层：把对话状态映射成机器狗能表达的动作。

这里刻意不绑定任何具体厂商 SDK，只定义一个极简接口 ``RobotDog``。
接入真实设备时实现该接口即可（Unitree Go2 / 宇树 / 自研底盘 / ROS 都适用）。

推荐的状态 -> 动作映射（可在实现里自由发挥）：

    listening  ：竖耳 / 头部微微抬起 / 蓝色呼吸灯
    thinking   ：头部轻微左右摆动 / 黄色流水灯
    speaking   ：前俯点头 / 绿色常亮灯（可配合音量做律动）
    idle       ：站立姿态 / 白色常亮灯
    error      ：红色闪烁 / 后退半步
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["RobotDog", "ConsoleRobotDog", "HttpRobotDog", "STATE_IDLE", "STATE_LISTENING", "STATE_THINKING", "STATE_SPEAKING", "STATE_ERROR"]

STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"
STATE_ERROR = "error"


class RobotDog(ABC):
    """机器狗动作接口。"""

    @abstractmethod
    def on_state(self, state: str, **payload: Any) -> None:
        """状态变化时回调。

        常见 payload：
            - ``text``：该状态下的文本（用户问句 / 回答片段）
            - ``reason``：error 状态下的原因
        """

    def close(self) -> None:  # 可选实现
        return None


class ConsoleRobotDog(RobotDog):
    """默认实现：只在控制台打印状态，便于调试和 CI 自测。"""

    _ICONS = {
        STATE_IDLE: "·",
        STATE_LISTENING: "🎧",
        STATE_THINKING: "🧠",
        STATE_SPEAKING: "🔊",
        STATE_ERROR: "⚠️",
    }

    def on_state(self, state: str, **payload: Any) -> None:
        icon = self._ICONS.get(state, "·")
        text = str(payload.get("text", "")).strip()
        suffix = f" {text}" if text else ""
        print(f"{icon} [{state}]{suffix}", flush=True)


class HttpRobotDog(RobotDog):
    """HTTP 适配：把状态 POST 给机器狗上的控制服务。

    对接示例（宇树/自研网关暴露一个 ``/action`` 即可）::

        robot = HttpRobotDog("http://192.168.12.1:8080", token="xxx")
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        endpoint: str = "/action",
        timeout: float = 3.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.token = token
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.timeout = timeout

    def on_state(self, state: str, **payload: Any) -> None:
        if not self.base_url:
            return
        import requests

        url = f"{self.base_url}{self.endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        # 机器狗断连不应该打断对话，这里静默吞掉异常
        try:
            requests.post(url, headers=headers, json={"state": state, **payload}, timeout=self.timeout)
        except Exception:
            pass
