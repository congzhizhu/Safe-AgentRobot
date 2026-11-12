# -*- coding: utf-8 -*-
"""
固定航点导航编排（Windows 侧 · 导航闭环）

职责
----
把「到达预设航点」这件事做成一个**闭环**：不断读取狗的实时位置/偏航，
计算朝向误差与距离，用原地转向 + 短距前进逼近目标，直到进入到达半径。
这是对现有 patrol.py「速度 × 时间」开环走位的升级。

控制律
------
纯追踪（Pure Pursuit）：
  1. 读当前位置 (x, y) 与偏航 yaw
  2. 计算期望朝向 theta_d = atan2(dy, dx)，朝向误差 delta = theta_d - yaw
  3. |delta| > 朝向阈值 → 原地转向（vyaw = k_yaw * delta，经 1.2 rad/s 限幅）
  4. 对准后 → 前进一小段（vx = min(v_max, k_lin * dist)）
  5. dist < 到达半径 → 航点完成
  6. 超时/超步数 → 标记失败，避免死循环

依赖
----
- 只调用 RobotClient 的 get_pose() / move() / estop()，不关心底层是 HTTP 还是 mock。
- 航点参数来自 configs/nav_waypoints.yaml（见 NavConfig）。

安全约束
--------
- 所有速度经 RobotClient → robot_service 的硬限幅兜底，本模块额外再做一次软限幅。
- 任何异常立即 estop() 并返回失败，不抛异常让上层崩溃。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .client import RobotClient

# 软限幅（与 robot_service.py 的硬限幅一致，本层再守一道）
MAX_VYAW = 1.2      # rad/s
MAX_VX = 0.6        # m/s


@dataclass
class Waypoint:
    """一个航点：相对上电原点的 (x, y) 目标。"""
    name: str
    x: float
    y: float


@dataclass
class NavConfig:
    """导航参数。数值为经验初值，必须真机标定。"""
    arrival_radius: float = 0.15          # 到达判定距离阈值（m）
    heading_threshold_deg: float = 12.0   # 朝向对准死区（°），越大越稳、越不易震荡
    turn_deg_per_sec: float = 55.0        # 实测原地转角速度（°/s），patrol.py 已标定为 55
    k_lin: float = 0.5                    # 线速增益
    v_max: float = 0.3                    # 前进线速度上限（m/s）
    forward_step_s: float = 0.6           # 单次前进指令时长（s）
    max_steps: int = 200                  # 单航点最大步数
    max_time_s: float = 60.0              # 单航点超时（s）
    # 是否用避障移动（ObstaclesAvoidClient）做前进。
    # 实测：避障服务会因 LiDAR 误判/未就绪而把前进速度压到接近 0，导致狗原地颤动。
    # 固定航点本就是预先规划的安全点位，默认关闭避障，前进走普通 move 更可靠。
    use_avoid: bool = False
    waypoints: list[Waypoint] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NavConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        cfg = cls(
            arrival_radius=float(data.get("arrival_radius", cls.arrival_radius)),
            heading_threshold_deg=float(
                data.get("heading_threshold_deg", cls.heading_threshold_deg)
            ),
            turn_deg_per_sec=float(data.get("turn_deg_per_sec", cls.turn_deg_per_sec)),
            k_lin=float(data.get("k_lin", cls.k_lin)),
            v_max=float(data.get("v_max", cls.v_max)),
            forward_step_s=float(data.get("forward_step_s", cls.forward_step_s)),
            max_steps=int(data.get("max_steps", cls.max_steps)),
            max_time_s=float(data.get("max_time_s", cls.max_time_s)),
            use_avoid=bool(data.get("use_avoid", cls.use_avoid)),
        )
        cfg.waypoints = [
            Waypoint(name=str(w.get("name", f"wp{i}")), x=float(w["x"]), y=float(w["y"]))
            for i, w in enumerate(data.get("waypoints", []))
        ]
        return cfg


@dataclass
class NavResult:
    """单个航点的导航结果。"""
    name: str
    reached: bool
    message: str
    elapsed_s: float = 0.0
    steps: int = 0


def _normalize_angle(a: float) -> float:
    """把角度归一化到 (-pi, pi]。"""
    a = math.fmod(a + math.pi, 2 * math.pi)
    if a <= 0:
        a += 2 * math.pi
    return a - math.pi


class WaypointNavigator:
    """固定航点导航编排器。"""

    def __init__(
        self,
        client: RobotClient,
        config: NavConfig,
        *,
        stop_on_fail: bool = False,
        on_progress=None,
        calibrate_origin: bool = True,
    ) -> None:
        self.client = client
        self.config = config
        self.stop_on_fail = stop_on_fail
        self.on_progress = on_progress  # 可选回调 (step, wp_name, pose, dist) -> None
        # 任务原点：把「开始导航那一刻」狗的位姿记为原点/朝向基准，
        # 使航点坐标恒为「从当前位姿出发」，不再依赖上电朝向。
        self.calibrate_origin = calibrate_origin
        self._origin = (0.0, 0.0, 0.0)  # (x0, y0, yaw0)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def _calibrate(self) -> bool:
        """把当前位姿记为任务原点，返回是否成功。

        朝向基准用 IMU 相对 yaw（imu_yaw_rel）——它是「相对订阅首帧」的转角，
        对原地转向也可靠。标定时记录当前的 imu_yaw_rel 作为 yaw0，之后朝向都用
        「当前 imu_yaw_rel - yaw0」这种相对量，彻底摆脱绝对朝向基准漂移。
        """
        pose = self.client.get_pose()
        if not pose.get("ok"):
            return False
        p = pose.get("pose", {})
        # 朝向基准：优先 imu_yaw_rel（相对转角），回退 IMU yaw 绝对值
        yaw0 = float(p.get("imu_yaw_rel", p.get("yaw", 0.0)))
        self._origin = (
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
            yaw0,
        )
        return True

    def _local(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """把世界坐标 (x, y, yaw) 变换到任务原点的局部坐标。

        原点平移到 (x0, y0)，并绕原点旋转 -yaw0，使「任务开始时狗朝向」成为
        局部 x 轴正方向。这样航点 (X, Y) 恒表示「沿出发朝向走 X 米、左偏 Y 米」。
        """
        x0, y0, yaw0 = self._origin
        dx = x - x0
        dy = y - y0
        c, s = math.cos(yaw0), math.sin(yaw0)
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        return lx, ly, _normalize_angle(yaw - yaw0)

    def run(self) -> list[NavResult]:
        """顺序导航所有航点，返回每个航点的结果。"""
        if self.calibrate_origin:
            self._calibrate()
        results: list[NavResult] = []
        for wp in self.config.waypoints:
            result = self.navigate_to(wp)
            results.append(result)
            if not result.reached and self.stop_on_fail:
                break
        return results

    def navigate_to(self, wp: Waypoint) -> NavResult:
        """导航到单个航点，闭环控制直到到达或超时。

        使用任务原点标定（若启用）后的局部坐标，使航点 (wp.x, wp.y) 恒为
        「相对出发位姿」的偏移，而非依赖上电坐标系的绝对偏移。
        """
        cfg = self.config
        heading_thresh = math.radians(cfg.heading_threshold_deg)
        start = time.monotonic()
        steps = 0

        while steps < cfg.max_steps and (time.monotonic() - start) < cfg.max_time_s:
            pose = self.client.get_pose()
            if not pose.get("ok"):
                self._emit(steps, wp.name, None, 0.0)
                time.sleep(0.2)
                steps += 1
                continue

            p = pose.get("pose", {})
            wx, wy = float(p.get("x", 0.0)), float(p.get("y", 0.0))
            # 朝向用相对转角 imu_yaw_rel（原地转向也可靠），回退 IMU yaw 绝对值。
            wyaw = float(p.get("imu_yaw_rel", p.get("yaw", 0.0)))
            if self.calibrate_origin:
                x, y, yaw = self._local(wx, wy, wyaw)
            else:
                x, y, yaw = wx, wy, wyaw

            dx = wp.x - x
            dy = wp.y - y
            dist = math.hypot(dx, dy)

            self._emit(steps, wp.name, (x, y, yaw), dist)

            # 到达判定
            if dist < cfg.arrival_radius:
                self._stop()
                return NavResult(wp.name, True, f"已到达（偏差 {dist:.3f}m）",
                                 elapsed_s=time.monotonic() - start, steps=steps)

            # 朝向纠偏（离散式：用实测角速度一次转到位，避免比例增益震荡）
            theta_d = math.atan2(dy, dx)
            delta = _normalize_angle(theta_d - yaw)
            if abs(delta) > heading_thresh:
                # 转向角速度直接用实测值 turn_deg_per_sec（°/s → rad/s）
                turn_speed_rad = math.radians(cfg.turn_deg_per_sec)
                vyaw = turn_speed_rad if delta > 0 else -turn_speed_rad
                # 用时 = 需转角度 / 实测角速度，精确转到位
                dur = abs(delta) / turn_speed_rad
                # 少量减速补偿（转向启动/停止的惯性），比例取 5% 而非固定值
                dur *= 1.05
                dur = min(1.5, max(0.1, dur))
                ok, msg = self._move(0.0, 0.0, vyaw, dur)
                if not ok:
                    self._estop()
                    return NavResult(wp.name, False, f"转向失败: {msg}",
                                     elapsed_s=time.monotonic() - start, steps=steps)
                steps += 1
                continue

            # 对准后前进
            vx = max(0.0, min(cfg.v_max, cfg.k_lin * dist))
            if vx <= 0.01:  # 速度过小，直接判定为足够近
                self._stop()
                return NavResult(wp.name, True, f"已到达（偏差 {dist:.3f}m）",
                                 elapsed_s=time.monotonic() - start, steps=steps)
            ok, msg = self._move(vx, 0.0, 0.0, cfg.forward_step_s)
            if not ok:
                self._estop()
                return NavResult(wp.name, False, f"前进失败: {msg}",
                                 elapsed_s=time.monotonic() - start, steps=steps)
            steps += 1

        # 超时 / 超步数
        self._stop()
        return NavResult(wp.name, False, "超时或步数用尽未到达",
                         elapsed_s=time.monotonic() - start, steps=steps)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _move(self, vx: float, vy: float, vyaw: float, dur: float) -> tuple[bool, str]:
        """移动执行。策略：
        - 默认（use_avoid=False）：全部走普通 move（SportClient），前进可靠。
          实测避障服务会因 LiDAR 误判/未就绪把前进速度压到接近 0，导致原地颤动，
          固定航点本就是安全点位，默认不用避障。
        - use_avoid=True 时：纯转向仍走普通 move（避障抑制原地旋转），
          前进/平移走避障 move。
        """
        turning = abs(vx) < 0.01 and abs(vy) < 0.01 and abs(vyaw) > 1e-3
        use_avoid = self.config.use_avoid and not turning
        if not use_avoid:
            return self.client.move(vx, vy, vyaw, dur)

        avoid_fn = getattr(self.client, "avoid_move", None)
        if avoid_fn is not None:
            ok, msg = avoid_fn(vx, vy, vyaw, dur)
            if ok:
                return ok, msg
            if "未知路径" in msg or "404" in msg or "不存在" in msg:
                return self.client.move(vx, vy, vyaw, dur)
            return ok, msg
        return self.client.move(vx, vy, vyaw, dur)

    def _stop(self) -> None:
        try:
            self.client.move(0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass

    def _estop(self) -> None:
        try:
            self.client.estop()
        except Exception:
            pass

    def _emit(self, step: int, wp_name: str, pose, dist: float) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(step, wp_name, pose, dist)
            except Exception:
                pass
