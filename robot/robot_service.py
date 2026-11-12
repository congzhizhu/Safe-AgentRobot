#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Go2 机器狗执行服务（WSL 侧 · 「手脚」层）
================================================================================

定位
----
本服务只做两件事：**执行一个原子动作**、**抓一张照片**。
所有编排逻辑（去哪个区域、怎么扫描、怎么诊断）都在 Windows 侧的
Safety_Agent 里（src/robot/patrol.py）。这样换掉底层实现（比如以后上 Nav2
自主导航）不影响上层智能体。

为什么不是 MCP
--------------
MCP 走 stdio，只适合"人在对话里临时点一下"。巡检需要「智能体连续下发几十条
指令 + 拿回图片路径」，用本机 HTTP 更直接、可被 WebUI 调用、也便于压测和日志。

通道
----
和已验证可用的控狗链路完全一致：unitree_sdk2py + CycloneDDS。
- 动作：SportClient        → /api/sport/request
- 拍照：VideoClient         → /api/videohub/request（GetImageSample，直接返回 JPEG）
两者共用一次 ChannelFactoryInitialize，同一块网卡，不需要 WebRTC / security token。

启动
----
    cd /mnt/d/Safety_Agent/robot
    python3 robot_service.py --nic auto --port 8010

API
---
    GET  /health                 服务存活 + 依赖状态（不碰硬件）
    GET  /status                 连接状态 / 当前网卡 / 最近一次错误
    GET  /actions                动作白名单及说明
    POST /action    {action, params?, duration?}   执行原子动作
    POST /move      {vx, vy, vyaw, duration}       连续速度控制（内部按 10Hz 重发）
    POST /capture   {label?}                       抓一帧并落盘，返回路径
    POST /estop                                    急停（Damp）
    GET  /log?n=50                                 最近日志

安全约束（写死在本层，智能体无法绕过）
--------------------------------------
1. 动作白名单：只允许 ACTIONS 中的动作
2. 参数限幅：vx/vy/vyaw 有硬上限，单次时长 ≤ 6s
3. 急停：/estop 立即 Damp；任何动作异常自动 Damp
4. 全量日志：每次调用写 JSONL，可回溯
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 动作白名单（与已验证的 go2_ctrl.py / SportController 保持一致）
#   动作名 -> (SDK 方法名, 参数个数, 说明)
# ---------------------------------------------------------------------------
ACTIONS: dict[str, tuple[str, int, str]] = {
    "damp":       ("Damp",          0, "阻尼/急停：电机关节泄力"),
    "stand":      ("StandUp",       0, "正常站高"),
    "standdown":  ("StandDown",     0, "趴下"),
    "rise":       ("RiseSit",       0, "从坐姿恢复站立"),
    "sit":        ("Sit",           0, "坐下"),
    "balance":    ("BalanceStand",  0, "平衡站立"),
    "stop":       ("StopMove",      0, "停止移动（保持站姿）"),
    "recover":    ("RecoveryStand", 0, "从翻倒/趴下恢复"),
    "stretch":    ("Stretch",       0, "伸懒腰"),
    "hello":      ("Hello",         0, "打招呼（前倾）"),
    "heart":      ("Heart",         0, "比心"),
    "content":    ("Content",       0, "开心"),
    "scrape":     ("Scrape",        0, "作揖拜年"),
    "dance1":     ("Dance1",        0, "舞蹈 1"),
    "dance2":     ("Dance2",        0, "舞蹈 2"),
    "staticwalk": ("StaticWalk",    0, "常规行走模式"),
    "speedlevel": ("SpeedLevel",    1, "速度档位 -1/0/1"),
    "move":       ("Move",          3, "速度控制（vx vy vyaw），需配合 duration"),
}

# 硬限幅：智能体算错了也撞不出去
LIMIT_VX = 0.6      # m/s   前进/后退
LIMIT_VY = 0.4      # m/s   左右平移
LIMIT_VYAW = 1.2    # rad/s 原地转向
MAX_DURATION = 6.0  # s     单次连续运动上限
MOVE_HZ = 10        # 速度指令重发频率（DDS 侧有超时，必须持续刷新）

# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ts_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_win_path(p: str) -> str:
    """把 WSL 的 /mnt/d/xxx 转成 Windows 的 D:\\xxx，方便 Windows 侧直接用。"""
    p = str(p)
    if p.startswith("/mnt/") and len(p) > 6 and p[6:7] == "/":
        drive = p[5].upper()
        return f"{drive}:\\" + p[7:].replace("/", "\\")
    return p


def detect_nic(prefer: str = "auto") -> tuple[str, str]:
    """
    选网卡。返回 (nic, 说明)。
    优先级：显式指定且存在 > 持有 192.168.123.x 的网卡 > 报错提示。
    """
    ifaces: dict[str, str] = {}
    try:
        out = subprocess.check_output(
            ["ip", "-o", "-4", "addr", "show"], text=True, timeout=5
        )
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                name = parts[1]
                addr = parts[3].split("/")[0]
                ifaces[name] = addr
    except Exception as exc:
        return prefer if prefer != "auto" else "eth0", f"无法枚举网卡（{exc}），按默认值处理"

    for name, addr in ifaces.items():
        if addr.startswith("192.168.123."):
            if prefer not in ("auto", name):
                return name, f"指定网卡 {prefer} 不存在，自动改用持有 192.168.123.x 的 {name}"
            return name, f"使用持有 192.168.123.x（{addr}）的网卡 {name}"

    if prefer != "auto" and prefer in ifaces:
        return prefer, f"使用指定网卡 {prefer}（{ifaces[prefer]}），未发现 192.168.123.x 网段"
    return prefer if prefer != "auto" else "eth0", (
        f"未发现 192.168.123.x 网段（当前网卡：{ifaces or '无'}）。"
        "网线是否插好？电脑静态 IP 是否设为 192.168.123.222？"
    )


# ---------------------------------------------------------------------------
# 硬件驱动
# ---------------------------------------------------------------------------
class Go2Driver:
    """Go2 的 DDS 通道封装：动作 + 拍照。线程安全，懒初始化。"""

    def __init__(self, nic: str, shot_dir: Path, log_dir: Path) -> None:
        self.nic = nic
        self.shot_dir = shot_dir
        self.log_dir = log_dir
        self._lock = threading.RLock()
        self._channel_ready = False
        self._sport = None
        self._avoid = None
        self._video = None
        self._sport_state_sub = None
        self._sport_state_ready = False
        self._pose_lock = threading.Lock()
        self._pose: dict[str, Any] = {}
        self._pose_error = ""
        # 朝向数据源（相对订阅首帧，避免 IMU yaw 绝对基准漂移）：
        #   imu_yaw_rel  = 当前 IMU yaw - 首帧 IMU yaw（相对转角，转向/直行都可靠）
        #   pos_yaw      = 位置增量反推的绝对朝向（position 有效时才可靠）
        # 导航器会优先用 imu_yaw_rel（相对转角），因为它对原地转向同样有效。
        self._imu_yaw0 = None           # 首帧 IMU yaw（基准）
        self._imu_yaw_rel = 0.0         # 相对首帧的转角
        self._pos_yaw = 0.0
        self._pos_yaw_valid = False
        self._last_pos = None  # (x, y, t)
        self._shot_idx = 0
        self.last_error = ""
        self.last_action = ""
        self.camera_ok: bool | None = None
        self.camera_note = ""
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 初始化 ----------------
    def _ensure_channel(self) -> None:
        """ChannelFactoryInitialize 全进程只能调一次，用标志位保护。"""
        if self._channel_ready:
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(0, self.nic)
        time.sleep(0.5)
        self._channel_ready = True

    def _get_sport(self):
        with self._lock:
            if self._sport is not None:
                return self._sport
            self._ensure_channel()
            from unitree_sdk2py.go2.sport.sport_client import SportClient

            sport = SportClient()
            sport.SetTimeout(10.0)
            sport.Init()
            self._sport = sport
            return sport

    def _get_avoid(self):
        """避障运动客户端（ObstaclesAvoidClient）。

        与 SportClient 的区别：SportClient 的 Move 是「盲目速度指令」，不会避障；
        避障必须走 ObstaclesAvoidClient，且需先 SwitchSet(True) + UseRemoteCommandFromApi(True)。
        """
        with self._lock:
            if self._avoid is not None:
                return self._avoid
            self._ensure_channel()
            from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
                ObstaclesAvoidClient,
            )

            avoid = ObstaclesAvoidClient()
            avoid.SetTimeout(10.0)
            avoid.Init()
            # 开启避障 + 接管遥控器命令源
            avoid.SwitchSet(True)
            time.sleep(0.3)
            avoid.UseRemoteCommandFromApi(True)
            self._avoid = avoid
            return avoid

    def _get_video(self):
        with self._lock:
            if self._video is not None:
                return self._video
            self._ensure_channel()
            from unitree_sdk2py.go2.video.video_client import VideoClient

            vc = VideoClient()
            try:
                vc.SetTimeout(5.0)
            except Exception:
                pass
            vc.Init()
            self._video = vc
            return vc

    # ---------------- 位置状态订阅 ----------------
    def _get_sport_state(self):
        """订阅 rt/sportmodestate，持续更新位置/姿态。

        只允许成功订阅一次（DDS 通道已初始化时复用）。返回 subscriber 或 None。
        注意：Go2 固件是否在 position 字段发布有效的里程计融合位置，取决于
        固件版本与运动模式，须真机实测确认。本方法同时缓存 position / velocity /
        imu rpy，由上层按需判断可用性。
        """
        with self._lock:
            if self._sport_state_sub is not None:
                return self._sport_state_sub
            self._ensure_channel()
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            def _on_state(msg) -> None:
                try:
                    pos = list(getattr(msg, "position", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
                    vel = list(getattr(msg, "velocity", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
                    rpy = [0.0, 0.0, 0.0]
                    imu = getattr(msg, "imu_state", None)
                    if imu is not None:
                        rpy = list(getattr(imu, "rpy", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
                    with self._pose_lock:
                        self._pose = {
                            "x": float(pos[0]) if len(pos) > 0 else 0.0,
                            "y": float(pos[1]) if len(pos) > 1 else 0.0,
                            "z": float(pos[2]) if len(pos) > 2 else 0.0,
                            "vx": float(vel[0]) if len(vel) > 0 else 0.0,
                            "vy": float(vel[1]) if len(vel) > 1 else 0.0,
                            "vz": float(vel[2]) if len(vel) > 2 else 0.0,
                            "roll": float(rpy[0]) if len(rpy) > 0 else 0.0,
                            "pitch": float(rpy[1]) if len(rpy) > 1 else 0.0,
                            "yaw": float(rpy[2]) if len(rpy) > 2 else 0.0,
                        }
                        # 相对 yaw：以订阅首帧为基准的转角（原地转向也可靠）
                        cur_yaw = float(rpy[2]) if len(rpy) > 2 else 0.0
                        if self._imu_yaw0 is None:
                            self._imu_yaw0 = cur_yaw
                        raw_rel = cur_yaw - self._imu_yaw0
                        # 归一化到 (-pi, pi]
                        self._imu_yaw_rel = math.fmod(raw_rel + math.pi, 2 * math.pi)
                        if self._imu_yaw_rel <= 0:
                            self._imu_yaw_rel += 2 * math.pi
                        self._imu_yaw_rel -= math.pi
                        # 位置增量反推朝向（与 x/y 同源）
                        px = float(pos[0]) if len(pos) > 0 else 0.0
                        py = float(pos[1]) if len(pos) > 1 else 0.0
                        now_t = time.monotonic()
                        if self._last_pos is not None:
                            lx, ly, lt = self._last_pos
                            dxp = px - lx
                            dyp = py - ly
                            dt = now_t - lt
                            if dt > 1e-6 and math.hypot(dxp, dyp) > 0.01:
                                self._pos_yaw = math.atan2(dyp, dxp)
                                self._pos_yaw_valid = True
                        self._last_pos = (px, py, now_t)
                        self._sport_state_ready = True
                except Exception as exc:
                    self._pose_error = f"状态回调异常: {exc}"

            sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            sub.Init(_on_state, 10)
            self._sport_state_sub = sub
            return sub

    def get_pose(self) -> dict[str, Any]:
        """返回最近一帧位置状态。若从未收到数据，尝试惰性订阅一次。"""
        if not self._sport_state_ready:
            try:
                self._get_sport_state()
            except Exception as exc:
                self._pose_error = f"订阅 sportmodestate 失败: {exc}"
                self.log("pose_error", error=self._pose_error)
        with self._pose_lock:
            pose = dict(self._pose)
            pose["imu_yaw_rel"] = self._imu_yaw_rel
            pose["pos_yaw"] = self._pos_yaw
            pose["pos_yaw_valid"] = self._pos_yaw_valid
        return {
            "ok": self._sport_state_ready,
            "pose": pose,
            "error": self._pose_error if not self._sport_state_ready else "",
        }

    # ---------------- 避障移动 ----------------
    def avoid_move(self, vx: float, vy: float, vyaw: float, duration: float) -> tuple[int, str]:
        """避障模式下的连续速度控制：走 ObstaclesAvoidClient，遇到障碍自动避让。

        与 move_for（SportClient）的区别：move_for 是盲目速度指令（撞墙），
        本方法走避障服务，狗的 LiDAR 会自动减速/绕行。
        """
        vx = max(-LIMIT_VX, min(LIMIT_VX, float(vx)))
        vy = max(-LIMIT_VY, min(LIMIT_VY, float(vy)))
        vyaw = max(-LIMIT_VYAW, min(LIMIT_VYAW, float(vyaw)))
        duration = max(0.0, min(MAX_DURATION, float(duration)))
        if duration <= 0:
            return -1, "duration 必须大于 0"

        try:
            avoid = self._get_avoid()
        except Exception as exc:
            self.last_error = f"ObstaclesAvoidClient 初始化失败: {exc}"
            self.log("avoid_error", error=self.last_error)
            return -1, self.last_error

        interval = 1.0 / MOVE_HZ
        deadline = time.monotonic() + duration
        code_last = 0
        try:
            while time.monotonic() < deadline:
                code_last = int(avoid.Move(vx, vy, vyaw))
                time.sleep(interval)
            avoid.Move(0.0, 0.0, 0.0)
        except Exception as exc:
            self.last_error = f"避障移动异常: {exc}"
            self.log("avoid_error", error=self.last_error, tb=traceback.format_exc())
            self._safe_damp()
            return -1, self.last_error

        self.last_action = "avoid_move"
        self.log("avoid_move", vx=vx, vy=vy, vyaw=vyaw, duration=duration, code=code_last)
        return 0, f"已避障移动：vx={vx} vy={vy} vyaw={vyaw}，持续 {duration:.1f}s"

    # ---------------- 日志 ----------------
    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": now_str(), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False)
        try:
            path = self.log_dir / f"robot_service_{datetime.now():%Y%m%d}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        print(f"[{record['ts']}] {event} {json.dumps(fields, ensure_ascii=False)}", flush=True)

    # ---------------- 动作 ----------------
    def act(self, action: str, params: list[float] | None = None) -> tuple[int, str]:
        """执行一个原子动作。返回 (code, message)，code==0 成功。"""
        params = params or []
        if action not in ACTIONS:
            return -1, f"动作不在白名单: {action}（可用：{', '.join(sorted(ACTIONS))}）"

        method, nargs, desc = ACTIONS[action]
        if nargs and len(params) != nargs:
            return -1, f"动作 {action}（{desc}）需要 {nargs} 个参数，收到 {len(params)}"

        # 参数限幅
        if action == "move":
            vx = max(-LIMIT_VX, min(LIMIT_VX, float(params[0])))
            vy = max(-LIMIT_VY, min(LIMIT_VY, float(params[1])))
            vyaw = max(-LIMIT_VYAW, min(LIMIT_VYAW, float(params[2])))
            params = [vx, vy, vyaw]

        try:
            sport = self._get_sport()
        except Exception as exc:
            self.last_error = f"SportClient 初始化失败: {exc}"
            self.log("act_error", action=action, error=self.last_error)
            return -1, self.last_error

        try:
            fn = getattr(sport, method, None)
            if fn is None:
                return -1, f"SDK 无此方法: {method}（固件/SDK 版本不匹配？）"
            code = fn(*params) if params else fn()
        except Exception as exc:
            self.last_error = f"执行 {action} 异常: {exc}"
            self.log("act_error", action=action, error=self.last_error, tb=traceback.format_exc())
            self._safe_damp()
            return -1, self.last_error

        self.last_action = action
        ok = int(code) == 0
        if not ok:
            self.last_error = f"{action} 返回错误码 {code}"
        self.log("act", action=action, params=params, code=int(code), ok=ok)
        return int(code), f"{action} {'成功' if ok else f'返回错误码 {code}'}"

    def move_for(self, vx: float, vy: float, vyaw: float, duration: float) -> tuple[int, str]:
        """连续速度控制：按 10Hz 重发，到时自动 StopMove。"""
        vx = max(-LIMIT_VX, min(LIMIT_VX, float(vx)))
        vy = max(-LIMIT_VY, min(LIMIT_VY, float(vy)))
        vyaw = max(-LIMIT_VYAW, min(LIMIT_VYAW, float(vyaw)))
        duration = max(0.0, min(MAX_DURATION, float(duration)))
        if duration <= 0:
            return -1, "duration 必须大于 0"

        try:
            sport = self._get_sport()
        except Exception as exc:
            self.last_error = f"SportClient 初始化失败: {exc}"
            self.log("move_error", error=self.last_error)
            return -1, self.last_error

        interval = 1.0 / MOVE_HZ
        deadline = time.monotonic() + duration
        code_last = 0
        try:
            while time.monotonic() < deadline:
                code_last = int(sport.Move(vx, vy, vyaw))
                time.sleep(interval)
            sport.StopMove()
        except Exception as exc:
            self.last_error = f"move 异常: {exc}"
            self.log("move_error", error=self.last_error, tb=traceback.format_exc())
            self._safe_damp()
            return -1, self.last_error

        self.last_action = "move"
        self.log("move", vx=vx, vy=vy, vyaw=vyaw, duration=duration, code=code_last)
        return 0, f"已移动：vx={vx} vy={vy} vyaw={vyaw}，持续 {duration:.1f}s"

    def _safe_damp(self) -> None:
        try:
            if self._sport is not None:
                self._sport.Damp()
                self.log("auto_damp", reason="动作异常触发保护")
        except Exception:
            pass

    def estop(self) -> tuple[int, str]:
        return self.act("damp")

    # ---------------- 拍照 ----------------
    def capture(self, label: str = "") -> dict[str, Any]:
        """
        抓一帧并存成 JPEG。返回 {ok, path_win, path_linux, bytes, size, error}
        校验点：返回值码为 0、数据非空、JPEG 魔数正确（fFD8）、体积 > 1KB。
        """
        try:
            vc = self._get_video()
        except Exception as exc:
            self.camera_ok = False
            self.camera_note = f"VideoClient 初始化失败: {exc}"
            self.log("capture_error", error=self.camera_note)
            return {"ok": False, "error": self.camera_note}

        data: Any = None
        err = ""
        for attempt in range(1, 4):
            try:
                api = getattr(vc, "GetImageSample", None) or getattr(vc, "GetImage", None)
                if api is None:
                    err = "SDK 中既无 GetImageSample 也无 GetImage"
                    break
                ret = api()
                code, payload = (ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, None))
                if int(code) != 0:
                    err = f"取图返回错误码 {code}（1=服务无响应，检查狗是否在运行/摄像头是否被占用）"
                elif not payload:
                    err = "取图返回空数据"
                else:
                    if isinstance(payload, list):
                        payload = bytes(payload)
                    if len(payload) < 1024:
                        err = f"图像数据过小（{len(payload)} 字节），可能不是有效画面"
                    elif bytes(payload[:2]) != b"\xff\xd8":
                        err = f"返回数据不是 JPEG（头部 {bytes(payload[:4])!r}）"
                    else:
                        data = bytes(payload)
                        break
            except Exception as exc:
                err = f"取图异常: {exc}"
            if attempt < 3:
                time.sleep(0.4)

        if data is None:
            self.camera_ok = False
            self.camera_note = err
            self.last_error = err
            self.log("capture_error", error=err)
            return {"ok": False, "error": err}

        self._shot_idx += 1
        safe_label = "".join(c for c in (label or "shot") if c.isalnum() or c in "-_")[:24] or "shot"
        fname = f"{ts_str()}_{self._shot_idx:03d}_{safe_label}.jpg"
        path = self.shot_dir / fname
        try:
            path.write_bytes(data)
        except Exception as exc:
            self.log("capture_save_error", error=str(exc))
            return {"ok": False, "error": f"保存图片失败: {exc}"}

        self.camera_ok = True
        self.camera_note = "正常"
        size = path.stat().st_size
        self.log("capture", file=fname, bytes=size, label=label)
        return {
            "ok": True,
            "file": fname,
            "path_linux": str(path),
            "path_win": to_win_path(str(path)),
            "bytes": size,
        }

    # ---------------- 状态 ----------------
    def status(self, try_connect: bool = True) -> dict[str, Any]:
        info: dict[str, Any] = {
            "nic": self.nic,
            "channel_ready": self._channel_ready,
            "connected": self._sport is not None,
            "avoid_ready": self._avoid is not None,
            "camera_ok": self.camera_ok,
            "camera_note": self.camera_note,
            "pose_ready": self._sport_state_ready,
            "last_action": self.last_action,
            "last_error": self.last_error,
            "shot_dir_win": to_win_path(str(self.shot_dir)),
        }
        if try_connect and self._sport is None:
            try:
                self._get_sport()
                info["connected"] = True
                info["last_error"] = ""
            except Exception as exc:
                info["connected"] = False
                info["last_error"] = f"连接失败: {exc}"
        return info


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------
class ServiceState:
    driver: Go2Driver | None = None
    started_at = time.time()
    recent: list[dict] = []


STATE = ServiceState()
_RECENT_LOCK = threading.Lock()


def remember(rec: dict) -> None:
    with _RECENT_LOCK:
        STATE.recent.append(rec)
        if len(STATE.recent) > 200:
            STATE.recent = STATE.recent[-200:]


class Handler(BaseHTTPRequestHandler):
    server_version = "Go2RobotService/1.0"

    # ---------- 工具 ----------
    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def log_message(self, fmt: str, *args: Any) -> None:  # 静音默认访问日志
        return

    # ---------- 路由 ----------
    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/health":
            return self._send(200, {
                "ok": True,
                "service": "go2-robot-service",
                "uptime_s": round(time.time() - STATE.started_at, 1),
                "nic": STATE.driver.nic if STATE.driver else None,
            })
        if path == "/status":
            if not STATE.driver:
                return self._send(503, {"ok": False, "error": "驱动未初始化"})
            return self._send(200, {"ok": True, **STATE.driver.status()})
        if path == "/pose":
            if not STATE.driver:
                return self._send(503, {"ok": False, "error": "驱动未初始化"})
            result = STATE.driver.get_pose()
            return self._send(200 if result.get("ok") else 503, result)
        if path == "/actions":
            return self._send(200, {
                "ok": True,
                "actions": {k: {"desc": v[2], "nargs": v[1]} for k, v in ACTIONS.items()},
                "limits": {"vx": LIMIT_VX, "vy": LIMIT_VY, "vyaw": LIMIT_VYAW, "max_duration": MAX_DURATION},
            })
        if path == "/log":
            n = 50
            if "n=" in self.path:
                try:
                    n = int(self.path.split("n=")[1].split("&")[0])
                except Exception:
                    pass
            with _RECENT_LOCK:
                return self._send(200, {"ok": True, "items": STATE.recent[-n:]})
        return self._send(404, {"ok": False, "error": f"未知路径 {path}"})

    def do_POST(self) -> None:
        if not STATE.driver:
            return self._send(503, {"ok": False, "error": "驱动未初始化"})
        path = self.path.split("?")[0]
        body = self._body()
        drv = STATE.driver

        try:
            if path == "/action":
                action = str(body.get("action", "")).strip()
                params = body.get("params") or []
                duration = float(body.get("duration") or 0)
                if action == "move" and duration > 0:
                    code, msg = drv.move_for(*(list(params) + [0, 0, 0])[:3], duration)
                else:
                    code, msg = drv.act(action, [float(x) for x in params])
                rec = {"ts": now_str(), "api": path, "action": action, "code": code, "msg": msg}
                remember(rec)
                return self._send(200 if code == 0 else 500, {"ok": code == 0, "code": code, "msg": msg})

            if path == "/move":
                code, msg = drv.move_for(
                    body.get("vx", 0.0), body.get("vy", 0.0),
                    body.get("vyaw", 0.0), body.get("duration", 0.0),
                )
                remember({"ts": now_str(), "api": path, "code": code, "msg": msg})
                return self._send(200 if code == 0 else 500, {"ok": code == 0, "code": code, "msg": msg})

            if path == "/avoid_move":
                code, msg = drv.avoid_move(
                    body.get("vx", 0.0), body.get("vy", 0.0),
                    body.get("vyaw", 0.0), body.get("duration", 0.0),
                )
                remember({"ts": now_str(), "api": path, "code": code, "msg": msg})
                return self._send(200 if code == 0 else 500, {"ok": code == 0, "code": code, "msg": msg})

            if path == "/capture":
                result = drv.capture(label=str(body.get("label", "shot")))
                remember({"ts": now_str(), "api": path, **result})
                return self._send(200 if result.get("ok") else 500, result)

            if path == "/estop":
                code, msg = drv.estop()
                remember({"ts": now_str(), "api": path, "code": code, "msg": msg})
                return self._send(200, {"ok": True, "code": code, "msg": msg})

            return self._send(404, {"ok": False, "error": f"未知路径 {path}"})
        except Exception as exc:
            drv.log("handler_error", path=path, error=str(exc), tb=traceback.format_exc())
            return self._send(500, {"ok": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Go2 机器狗执行服务（WSL 侧）")
    ap.add_argument("--nic", default="auto", help="网卡名，auto=自动找持有 192.168.123.x 的网卡")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（WSL 镜像模式下 Windows 可直连 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--shot-dir", default="/mnt/d/Safety_Agent/data/robot_shots")
    ap.add_argument("--log-dir", default="/mnt/d/Safety_Agent/data/robot_logs")
    args = ap.parse_args()

    nic, note = detect_nic(args.nic)
    print(f"[网卡] {note}")

    driver = Go2Driver(nic=nic, shot_dir=Path(args.shot_dir), log_dir=Path(args.log_dir))
    STATE.driver = driver

    # 主动探一次连接，让启动日志直接暴露问题
    print("[连接] 正在初始化 DDS 通道与 SportClient ...")
    try:
        driver._get_sport()
        print("[连接] ✅ SportClient 就绪，狗应该可以被控制")
    except Exception as exc:
        print(f"[连接] ⚠️ 初始化失败：{exc}")
        print("       服务仍会启动，后续每次调用都会重试。请检查：")
        print("       1) 网线是否插好、狗是否开机")
        print("       2) ip -br addr 是否有 192.168.123.222")
        print(f"       3) 网卡名是否正确（当前用 {nic}）")

    # 主动订阅一次位置状态，暴露 sportmodestate 是否可用
    print("[定位] 正在订阅 rt/sportmodestate ...")
    try:
        driver._get_sport_state()
        print("[定位] ✅ 已发起位置状态订阅（/pose 端点可用，数据有效性需真机确认）")
    except Exception as exc:
        print(f"[定位] ⚠️ 位置订阅失败：{exc}")
        print("        /pose 端点仍会存在，但会返回 ok:false。导航功能依赖此数据。")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[服务] 监听 http://{args.host}:{args.port}  （Ctrl+C 停止）")
    print(f"[图片] {args.shot_dir}  →  Windows: {to_win_path(args.shot_dir)}")
    print(f"[日志] {args.log_dir}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出] 收到中断")
    finally:
        try:
            driver.estop()
        except Exception:
            pass
        srv.server_close()


if __name__ == "__main__":
    main()
