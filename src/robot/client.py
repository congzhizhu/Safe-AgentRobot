# -*- coding: utf-8 -*-
"""
机器狗能力封装（Windows 侧）

职责
----
1. 与 WSL 侧的 robot_service.py 通信（HTTP 127.0.0.1:8010）
2. 服务不可达时自动进入 **mock 模式**：用 data/data_processed/images 里的
   真实隐患样本图冒充"狗拍到的照片"，让「指令→巡检→诊断→报告」全链路
   在狗没插线时也能演示和调试。

设计原则
--------
- 上层（patrol.py / WebUI）只调用本类的方法，不关心底下是 HTTP 还是 mock，
  也不关心机器人是 Go2 还是别的型号。
- 所有物理动作都经过 robot_service 的白名单与限幅，本层不做"聪明"的兜底。
"""

from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_SERVICE = "http://127.0.0.1:8010"


class RobotUnavailable(RuntimeError):
    """控狗服务不可达或狗未连接。"""


@dataclass
class RobotSnapshot:
    """一次状态查询的结果，供 UI 展示。"""

    ok: bool = False
    connected: bool = False
    mock: bool = False
    nic: str = ""
    camera_ok: bool | None = None
    last_error: str = ""
    note: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class RobotClient:
    """控狗的统一入口。"""

    def __init__(
        self,
        base_url: str = DEFAULT_SERVICE,
        *,
        project_root: str | Path | None = None,
        allow_mock: bool = True,
        timeout: float = 90.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.allow_mock = allow_mock
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.mock = False
        self.note = ""
        self._mock_pool: list[Path] = []
        self._mock_idx = 0
        self._state_checked = False

    # ------------------------------------------------------------------
    # 可用性
    # ------------------------------------------------------------------
    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200 and bool(resp.json().get("ok"))
        except Exception:
            return False

    def _lazy_ensure_state(self) -> None:
        """首次执行动作前若尚未检测过连接状态，则惰性检测一次。

        这样即使调用方忘记先调 snapshot()，也能正确进入 mock 或报错，
        避免 mock 状态依赖调用顺序。真机模式（allow_mock=False）下服务不可达
        会如实报错，不会静默降级。
        """
        if self._state_checked:
            return
        self._state_checked = True
        self.snapshot()

    def snapshot(self) -> RobotSnapshot:
        """查询状态；服务不可达时按 allow_mock 决定进 mock 还是报错。"""
        self._state_checked = True
        if not self.health():
            if self.allow_mock:
                self.mock = True
                self.note = (
                    "未检测到控狗服务（127.0.0.1:8010），当前为**演示模式**："
                    "照片取自本地隐患样本库，狗不会真的移动。"
                )
                return RobotSnapshot(ok=True, connected=False, mock=True, note=self.note)
            self.mock = False
            return RobotSnapshot(
                ok=False,
                connected=False,
                note="控狗服务未启动或狗未连接。请在 WSL 里运行 robot_service.py。",
            )

        try:
            data = requests.get(f"{self.base_url}/status", timeout=20).json()
        except Exception as exc:
            return RobotSnapshot(ok=False, connected=False, note=f"状态查询失败: {exc}")

        self.mock = False
        self.note = ""
        return RobotSnapshot(
            ok=bool(data.get("ok", True)),
            connected=bool(data.get("connected")),
            mock=False,
            nic=str(data.get("nic", "")),
            camera_ok=data.get("camera_ok"),
            last_error=str(data.get("last_error") or ""),
            note=str(data.get("camera_note") or ""),
            raw=data,
        )

    def list_actions(self) -> dict[str, Any]:
        try:
            return requests.get(f"{self.base_url}/actions", timeout=5).json()
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------
    def act(self, action: str, params: list[float] | None = None, duration: float = 0.0) -> tuple[bool, str]:
        """执行原子动作。mock 模式下只做时间模拟。"""
        params = params or []
        self._lazy_ensure_state()
        if self.mock:
            time.sleep(min(duration, 2.0) if duration else 0.35)
            return True, f"[演示] {action} {params or ''}".strip()

        try:
            resp = requests.post(
                f"{self.base_url}/action",
                json={"action": action, "params": params, "duration": duration},
                timeout=self.timeout,
            )
            data = resp.json()
        except Exception as exc:
            return False, f"调用失败: {exc}"
        return bool(data.get("ok")), str(data.get("msg") or data.get("error") or "")

    def move(self, vx: float, vy: float, vyaw: float, duration: float) -> tuple[bool, str]:
        self._lazy_ensure_state()
        if self.mock:
            time.sleep(min(duration, 2.0))
            return True, f"[演示] 移动 vx={vx} vy={vy} vyaw={vyaw} {duration}s"
        try:
            resp = requests.post(
                f"{self.base_url}/move",
                json={"vx": vx, "vy": vy, "vyaw": vyaw, "duration": duration},
                timeout=self.timeout,
            )
            data = resp.json()
        except Exception as exc:
            return False, f"移动失败: {exc}"
        return bool(data.get("ok")), str(data.get("msg") or data.get("error") or "")

    def avoid_move(self, vx: float, vy: float, vyaw: float, duration: float) -> tuple[bool, str]:
        """避障模式下的移动（走 ObstaclesAvoidClient，遇到障碍自动避让）。

        mock 模式下退化为时间模拟。用于导航场景，避免转弯/前进时撞墙。
        """
        self._lazy_ensure_state()
        if self.mock:
            time.sleep(min(duration, 2.0))
            return True, f"[演示] 避障移动 vx={vx} vy={vy} vyaw={vyaw} {duration}s"
        try:
            resp = requests.post(
                f"{self.base_url}/avoid_move",
                json={"vx": vx, "vy": vy, "vyaw": vyaw, "duration": duration},
                timeout=self.timeout,
            )
            data = resp.json()
        except Exception as exc:
            return False, f"避障移动失败: {exc}"
        return bool(data.get("ok")), str(data.get("msg") or data.get("error") or "")

    def estop(self) -> tuple[bool, str]:
        if self.mock:
            return True, "[演示] 急停"
        try:
            resp = requests.post(f"{self.base_url}/estop", json={}, timeout=15)
            data = resp.json()
        except Exception as exc:
            return False, f"急停调用失败: {exc}"
        return bool(data.get("ok", True)), str(data.get("msg") or "")

    # ------------------------------------------------------------------
    # 拍照
    # ------------------------------------------------------------------
    def capture(self, label: str = "shot") -> dict[str, Any]:
        """抓一帧，返回 {ok, path_win, file, bytes} 或 {ok:False, error}。"""
        self._lazy_ensure_state()
        if self.mock:
            return self._mock_capture(label)
        try:
            resp = requests.post(
                f"{self.base_url}/capture", json={"label": label}, timeout=self.timeout
            )
            return resp.json()
        except Exception as exc:
            return {"ok": False, "error": f"抓拍失败: {exc}"}

    # ------------------------------------------------------------------
    # 位置状态
    # ------------------------------------------------------------------
    def get_pose(self) -> dict[str, Any]:
        """查询当前位置/姿态。返回 {ok, pose:{x,y,z,yaw,...}} 或 {ok:False, error}。

        mock 模式下返回一个静态零位姿，保证导航代码在无狗时也能跑通流程。
        """
        self._lazy_ensure_state()
        if self.mock:
            return {
                "ok": True,
                "pose": {
                    "x": 0.0, "y": 0.0, "z": 0.0,
                    "vx": 0.0, "vy": 0.0, "vz": 0.0,
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                },
                "mock": True,
            }
        try:
            resp = requests.get(f"{self.base_url}/pose", timeout=10)
            return resp.json()
        except Exception as exc:
            return {"ok": False, "error": f"位置查询失败: {exc}"}

    # ------------------------------------------------------------------
    # mock 实现
    # ------------------------------------------------------------------
    def _load_mock_pool(self) -> list[Path]:
        if self._mock_pool:
            return self._mock_pool
        base = self.project_root / "data" / "data_processed" / "images"
        exts = {".jpg", ".jpeg", ".png"}
        files = [p for p in base.glob("*") if p.suffix.lower() in exts] if base.exists() else []
        random.shuffle(files)
        self._mock_pool = files[:40]
        return self._mock_pool

    def _mock_capture(self, label: str) -> dict[str, Any]:
        pool = self._load_mock_pool()
        if not pool:
            return {"ok": False, "error": "演示模式下未找到样本图（data/data_processed/images 为空）"}
        src = pool[self._mock_idx % len(pool)]
        self._mock_idx += 1

        shot_dir = self.project_root / "data" / "robot_shots"
        shot_dir.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{self._mock_idx:03d}_{label or 'shot'}{src.suffix}"
        dst = shot_dir / name
        try:
            shutil.copyfile(src, dst)
        except Exception as exc:
            return {"ok": False, "error": f"演示图复制失败: {exc}"}

        time.sleep(0.4)
        return {
            "ok": True,
            "file": name,
            "path_win": str(dst),
            "path_linux": "",
            "bytes": dst.stat().st_size,
            "mock": True,
            "mock_source": src.name,
        }
