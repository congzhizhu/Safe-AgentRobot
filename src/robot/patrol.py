# -*- coding: utf-8 -*-
"""
机器狗巡检编排器（Windows 侧 · 「大脑」层）

这里回答三个问题：
1. **去哪儿、怎么走** —— 区域路线（Zone）与指令计划（plan_command）
2. **拍什么、拍几张** —— scan 扫描序列（边走位边四向拍照）
3. **拍到的东西算不算隐患** —— 逐张丢给 Safety_Agent.analyze()，再汇总成报告

为什么要「开环巡检」而不是 Nav2
-----------------------------
当前狗没有地图/定位，所以走位靠「速度 × 时间」估算。这是刻意选择：
先把「下指令 → 狗动 → 拍照 → 诊断 → 出报告」的闭环跑通、可演示，
二期再把手脚层换成 Nav2 自主导航，本文件的路线定义与诊断逻辑不用改。
所有路线都是保守值（低速、短时），现场标定 `turn_deg_per_sec` 即可提高精度。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from robot.client import RobotClient
from robot.navigator import NavConfig, WaypointNavigator

# ---------------------------------------------------------------------------
# 现场标定参数
# ---------------------------------------------------------------------------
TURN_VYAW = 0.9          # 原地转角的角速度（rad/s），限幅内
TURN_DEG_PER_SEC = 55.0  # 实测每秒转多少度；90° ≈ 1.64s。首次数格子后改这里
WALK_VX = 0.35           # 巡检前进速度（m/s），保守值
DEFAULT_SHOTS = 4        # 每个巡视位拍几张
MAX_SHOTS_TOTAL = 12     # 单次巡检最多拍几张（防止无限跑）
MAX_PLAN_SECONDS = 30.0  # 单个指令计划总时长上限

# ---- 韧性参数（完成检查 / 失败恢复）----
ACTION_RETRY = 1         # 站姿类关键动作失败后的重试次数
RETRY_BACKOFF = 1.2      # 重试前的等待秒数，给狗留出执行时间
MIN_SHOT_BYTES = 1024    # 低于此体积视为无效照片（多为空白帧或传输截断）
MAX_CONSECUTIVE_FAIL = 3 # 连续失败达到此值即判定无法继续，转降级出报告

# 允许 LLM 使用的动作（与 robot_service.py 白名单对齐，双层防护）
LLM_ACTIONS = {
    "stand", "standdown", "sit", "rise", "balance", "stop", "recover",
    "stretch", "hello", "heart", "content", "damp",
}


# ---------------------------------------------------------------------------
# 区域路线
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    key: str
    name: str
    desc: str
    route: list[dict]
    tip: str = ""


ZONES: dict[str, Zone] = {
    "scan_only": Zone(
        key="scan_only",
        name="原地扫描（最安全，演示首选）",
        desc="狗站在原位，每转 90° 拍一张，共 4 张",
        route=[
            {"op": "action", "action": "stand"},
            {"op": "scan", "shots": 4, "settle": 1.3},
        ],
        tip="不需要场地余量，适合第一次联调与路演现场",
    ),
    "work_area": Zone(
        key="work_area",
        name="作业区（前行 + 四向扫描）",
        desc="站起 → 低速前行约 1 米 → 停稳 → 四向各拍一张",
        route=[
            {"op": "action", "action": "stand"},
            {"op": "move", "vx": WALK_VX, "vy": 0.0, "vyaw": 0.0, "duration": 3.0},
            {"op": "action", "action": "stop"},
            {"op": "scan", "shots": 4, "settle": 1.3},
        ],
        tip="前方 1 米内不要有台阶/线缆/人员",
    ),
    "storage_area": Zone(
        key="storage_area",
        name="仓库区（前行 + 右转 + 再前行 + 扫描）",
        desc="站起 → 前行约 1.4 米 → 右转 90° → 再前行约 1 米 → 四向扫描",
        route=[
            {"op": "action", "action": "stand"},
            {"op": "move", "vx": WALK_VX, "vy": 0.0, "vyaw": 0.0, "duration": 4.0},
            {"op": "action", "action": "stop"},
            {"op": "turn", "deg": 90},
            {"op": "move", "vx": WALK_VX, "vy": 0.0, "vyaw": 0.0, "duration": 3.0},
            {"op": "action", "action": "stop"},
            {"op": "scan", "shots": 4, "settle": 1.3},
        ],
        tip="两段行进后位置会累积误差，需现场数格子标定时长",
    ),
    "full_patrol": Zone(
        key="full_patrol",
        name="全区域巡视（三段推进 + 三次扫描）",
        desc="前行 → 扫描 → 前行 → 扫描 → 右转 → 前行 → 扫描，共约 9 张",
        route=[
            {"op": "action", "action": "stand"},
            {"op": "move", "vx": WALK_VX, "vy": 0.0, "vyaw": 0.0, "duration": 2.5},
            {"op": "action", "action": "stop"},
            {"op": "scan", "shots": 3, "settle": 1.2},
            {"op": "move", "vx": WALK_VX, "vy": 0.0, "vyaw": 0.0, "duration": 2.5},
            {"op": "action", "action": "stop"},
            {"op": "scan", "shots": 3, "settle": 1.2},
            {"op": "turn", "deg": 90},
            {"op": "move", "vx": WALK_VX, "vy": 0.0, "vyaw": 0.0, "duration": 2.5},
            {"op": "action", "action": "stop"},
            {"op": "scan", "shots": 3, "settle": 1.2},
        ],
        tip="耗时约 2~3 分钟（不含诊断），演示前先确认地面平整",
    ),
}


def zone_choices() -> list[tuple[str, str]]:
    return [(z.name, k) for k, z in ZONES.items()]


# ---------------------------------------------------------------------------
# 指令计划：自然语言 → 动作序列
# ---------------------------------------------------------------------------
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}


def _cn_number(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if text == "十几":
        return 10.0
    total = 0.0
    if text in _CN_NUM:
        return float(_CN_NUM[text])
    m = re.fullmatch(r"([一二两三四五六七八九])?十([一二三四五六七八九])?", text)
    if m:
        tens = _CN_NUM.get(m.group(1) or "一", 1)
        ones = _CN_NUM.get(m.group(2) or "", 0)
        return float(tens * 10 + ones)
    return None


def plan_by_rules(text: str) -> list[dict]:
    """正则兜底解析。够用、可解释，且永不产生白名单外的动作。"""
    t = (text or "").strip()
    ops: list[dict] = []
    if not t:
        return ops

    # 巡检类
    if re.search(r"巡检|巡查|巡视|扫描|转一圈|看一圈|看看周围|扫一圈", t):
        shots = 4
        m = re.search(r"([0-9一二两三四五六七八九十]+)\s*张", t)
        if m:
            v = _cn_number(m.group(1))
            if v:
                shots = int(max(1, min(8, v)))
        ops.append({"op": "action", "action": "stand"})
        ops.append({"op": "scan", "shots": shots, "settle": 1.3})
        return ops

    # 前进 / 后退
    for kw, vx in ((r"前进|向前|往前走|向前走|直行", WALK_VX), (r"后退|向后|退后", -0.25)):
        if re.search(kw, t):
            m = re.search(r"([0-9]+(?:\.[0-9]+)?|[一二两三四五六七八九十半]+)\s*(?:秒|s|S)", t)
            dur = _cn_number(m.group(1)) if m else 2.0
            dur = float(max(0.5, min(4.0, dur or 2.0)))
            dist = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:米|m|M)", t)
            if dist:
                dur = float(max(0.5, min(4.0, float(dist.group(1)) / abs(vx))))
            ops.append({"op": "action", "action": "stand"})
            ops.append({"op": "move", "vx": vx, "vy": 0.0, "vyaw": 0.0, "duration": round(dur, 1)})
            ops.append({"op": "action", "action": "stop"})
            break

    # 转向
    for kw, sign in ((r"左转|向左|往左", 1), (r"右转|向右|往右", -1)):
        if re.search(kw, t):
            m = re.search(r"([0-9]+(?:\.[0-9]+)?|[一二两三四五六七八九十半]+)\s*度", t)
            deg = _cn_number(m.group(1)) if m else 90.0
            deg = float(max(15.0, min(180.0, deg or 90.0)))
            ops.append({"op": "turn", "deg": sign * deg})
            break

    # 拍照
    if re.search(r"拍照|拍一张|拍张照|抓拍|拍个照", t):
        ops.append({"op": "capture", "label": "manual"})

    # 简单姿态/动作
    for kw, act in (
        (r"趴下|卧倒", "standdown"), (r"坐下", "sit"), (r"站起来|站起|起立", "stand"),
        (r"恢复|爬起来", "recover"), (r"伸懒腰", "stretch"), (r"打招呼|问好", "hello"),
        (r"比心", "heart"), (r"开心", "content"), (r"停止|停下|别动", "stop"),
        (r"急停|紧急停止|断电", "damp"),
    ):
        if re.search(kw, t):
            ops.append({"op": "action", "action": act})
            break

    return ops


def plan_with_llm(agent, text: str) -> tuple[list[dict], str]:
    """
    让智能体把自然语言翻译成动作计划（这才是"智能体下指令"）。
    失败/解析不出时退回正则，保证永远有可用计划。
    """
    system = (
        "你是机器狗巡检指令解析器，只输出 JSON，不要任何解释文字。\n"
        "可用动作（op）：\n"
        "1. {\"op\":\"action\",\"action\":\"<动作>\"} 动作可选：stand/standdown/sit/rise/balance/stop/recover/stretch/hello/heart/content/damp\n"
        "2. {\"op\":\"move\",\"vx\":<-0.5~0.5>,\"vy\":<-0.3~0.3>,\"vyaw\":<-1.0~1.0>,\"duration\":<0.5~4.0秒>}\n"
        "3. {\"op\":\"turn\",\"deg\":<角度，正数左转，负数右转>}\n"
        "4. {\"op\":\"capture\",\"label\":\"<短标签>\"}\n"
        "5. {\"op\":\"scan\",\"shots\":<1~6>,\"settle\":<0.8~2.0>} 原地每转约90°拍一张\n"
        "输出格式：{\"plan\":[...],\"say\":\"给操作员的一句话说明\"}\n"
        "安全约束：涉及移动时速度不超过 0.4 m/s，单段 duration 不超过 4 秒；"
        "总移动时长不超过 15 秒；不确定时宁可只 stand 不移动。"
    )
    user = f"操作员说：{text}\n请输出 JSON 计划。"
    raw = ""
    try:
        raw = agent._chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=600,
            timeout=40,
            model=getattr(agent.config, "text_model_name", None),
        )
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("LLM 未返回 JSON")
        data = json.loads(raw[start:end + 1])
        plan = validate_plan(data.get("plan") or [])
        say = str(data.get("say") or "")
        if plan:
            return plan, say
        raise ValueError("LLM 计划为空或未通过校验")
    except Exception as exc:
        rule_plan = plan_by_rules(text)
        if rule_plan:
            return rule_plan, f"（规则解析）{exc}"
        return [], f"无法解析为可执行计划：{exc}"


def validate_plan(ops: list[Any]) -> list[dict]:
    """硬校验：白名单动作 + 参数限幅 + 总时长上限。任何不合规的步骤直接丢弃。"""
    clean: list[dict] = []
    total = 0.0
    shots = 0
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = op.get("op")
        if kind == "action":
            act = str(op.get("action", ""))
            if act in LLM_ACTIONS:
                clean.append({"op": "action", "action": act})
        elif kind == "move":
            dur = min(4.0, max(0.5, float(op.get("duration", 2.0))))
            if total + dur > MAX_PLAN_SECONDS:
                continue
            total += dur
            clean.append({
                "op": "move",
                "vx": max(-0.5, min(0.5, float(op.get("vx", 0.0)))),
                "vy": max(-0.3, min(0.3, float(op.get("vy", 0.0)))),
                "vyaw": max(-1.0, min(1.0, float(op.get("vyaw", 0.0)))),
                "duration": round(dur, 1),
            })
        elif kind == "turn":
            deg = max(-180.0, min(180.0, float(op.get("deg", 90))))
            clean.append({"op": "turn", "deg": round(deg, 1)})
        elif kind == "capture":
            clean.append({"op": "capture", "label": str(op.get("label", "shot"))[:20]})
        elif kind == "scan":
            n = int(max(1, min(6, int(op.get("shots", 4)))))
            if shots + n > MAX_SHOTS_TOTAL:
                n = max(1, MAX_SHOTS_TOTAL - shots)
            shots += n
            clean.append({
                "op": "scan",
                "shots": n,
                "settle": round(min(2.0, max(0.8, float(op.get("settle", 1.2)))), 1),
            })
    return clean


def describe_plan(ops: list[dict]) -> str:
    """把计划渲染成人话，便于操作员执行前确认。"""
    if not ops:
        return "_（空计划）_"
    lines = []
    for i, op in enumerate(ops, 1):
        kind = op["op"]
        if kind == "action":
            lines.append(f"{i}. 执行动作 `{op['action']}`")
        elif kind == "move":
            lines.append(
                f"{i}. 移动 vx={op['vx']} vy={op['vy']} vyaw={op['vyaw']}，持续 {op['duration']}s"
            )
        elif kind == "turn":
            direction = "左转" if op["deg"] > 0 else "右转"
            lines.append(f"{i}. {direction} {abs(op['deg']):.0f}°")
        elif kind == "capture":
            lines.append(f"{i}. 拍照（{op.get('label', 'shot')}）")
        elif kind == "scan":
            lines.append(f"{i}. 原地扫描：{op['shots']} 个方向各拍 1 张（每张间隔转 90°）")
    total_move = sum(op.get("duration", 0) for op in ops if op["op"] == "move")
    n_shots = sum(op.get("shots", 0) for op in ops if op["op"] == "scan") + sum(
        1 for op in ops if op["op"] == "capture"
    )
    lines.append(f"\n> 预计移动 {total_move:.1f}s，拍照 {n_shots} 张")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 执行计划（生成器，供 UI 逐步刷新）
# ---------------------------------------------------------------------------
def _turn_seconds(deg: float) -> float:
    return max(0.2, min(3.0, abs(deg) / TURN_DEG_PER_SEC))


# ---------------------------------------------------------------------------
# 完成检查与失败恢复（参考 AdaHVLA 的 Completion / Recovery 机制）
# ---------------------------------------------------------------------------
def shot_is_valid(shot: dict) -> tuple[bool, str]:
    """完成检查：确认抓拍真的落盘，且体积足以构成一张可用画面。

    服务层已校验 JPEG 魔数与 >1KB，但网络重试、磁盘写入失败、路径映射错误
    仍可能让「接口返回成功」而文件不可用，所以在编排层再查一次文件系统。
    """
    if not isinstance(shot, dict) or not shot.get("ok"):
        return False, str(shot.get("error") if isinstance(shot, dict) else "抓拍返回异常")
    path = shot.get("path_win") or shot.get("path_linux")
    if not path:
        return False, "接口未返回图片路径"
    target = Path(str(path))
    if not target.exists():
        return False, f"图片未落盘：{path}"
    try:
        size = target.stat().st_size
    except OSError as exc:
        return False, f"图片不可读：{exc}"
    if size < MIN_SHOT_BYTES:
        return False, f"图片过小（{size} 字节），疑似空白帧"
    return True, f"{size} 字节"


def act_with_retry(client: RobotClient, action: str, retries: int = 0) -> tuple[bool, str, int]:
    """执行动作并在失败时重试。返回 (ok, message, 实际尝试次数)。"""
    ok, msg = client.act(action)
    attempts = 1
    for _ in range(max(0, retries)):
        if ok:
            break
        time.sleep(RETRY_BACKOFF)
        ok, msg = client.act(action)
        attempts += 1
    return ok, msg, attempts


def recover_posture(client: RobotClient, log: list[str]) -> bool:
    """姿态恢复：先强制停止，再尝试从异常姿态站起。

    只在「站立/恢复」这类关键动作失败后调用，避免把狗留在趴地状态下继续
    执行移动指令（这比直接放弃更危险）。
    """
    ok_stop, msg_stop = client.act("stop")
    log.append(f"   恢复① stop → {'✅' if ok_stop else '❌'} {msg_stop}")
    ok_recover, msg_recover = client.act("recover")
    log.append(f"   恢复② recover → {'✅' if ok_recover else '❌'} {msg_recover}")
    if ok_recover:
        time.sleep(RETRY_BACKOFF)
    return ok_recover


def _render_quality_md(stats: dict) -> str:
    """把执行质量统计渲染成报告尾部的固定区块。"""
    if not isinstance(stats, dict) or not stats:
        return ""
    lines = [
        "",
        "---",
        "### 🧪 执行质量（完成检查）",
        "",
        f"- 计划步骤：**{stats.get('步骤总数', 0)}** 个，成功 **{stats.get('成功', 0)}**，"
        f"失败 **{stats.get('失败', 0)}**",
        f"- 动作重试：**{stats.get('重试次数', 0)}** 次",
        f"- 照片有效性：**{stats.get('有效照片', 0)}** 张可用 / "
        f"**{stats.get('无效照片', 0)}** 张无效",
    ]
    if stats.get("提前结束"):
        lines.append(f"- 提前结束原因：**{stats['提前结束']}**")
    return "\n".join(lines)


_FIELD_HEADERS = {
    "思维链": "🧠 分析推理过程",
    "隐患定性": "🚨 隐患定性",
    "法律依据": "📖 法律依据",
    "整改措施": "🛠️ 整改措施",
    "参考案例": "🗂️ 参考案例",
    "引用校验": "🧾 法条引用核验",
}


def _render_streaming_text(fields: dict[str, str]) -> str:
    """把已生成的各字段可读文本拼成一份不断增长的流式 markdown。

    fields 是「字段名 -> 当前可读文本」的累积快照；渲染时按固定顺序输出，
    每来一个新字段/新增量，整份 markdown 就多一块，前端逐字上屏。

    思维链用折叠块（与最终 answer 一致），其余字段用加粗标题 + 正文。
    """
    parts: list[str] = []
    thought = (fields.get("思维链") or "").strip()
    if thought:
        parts.append(
            "<details open class='sa-details'><summary>🧠 <b>分析推理过程</b>（点击折叠）</summary>\n\n"
            f"{thought}\n\n</details>"
        )
    for key in ("隐患定性", "法律依据", "整改措施", "参考案例", "引用校验"):
        if key not in fields:
            continue
        text = (fields[key] or "").strip()
        if not text:
            continue
        header = _FIELD_HEADERS.get(key, key)
        parts.append(f"**{header}**\n\n{text}")
    return "\n\n---\n\n".join(parts)


def _stream_diagnose_one(agent, path: str, question: str) -> Generator[dict, None, dict]:
    """对单张照片做流式诊断，产出「可读流式文本」增量事件，最终返回完整 result。

    事件：{"type": "stream", "text": str} ｜ {"type": "result", "result": dict}
    正确消费 agent.analyze_stream 的真实事件协议（field_text / field_done / done），
    把逐字段可读文本累积成 markdown，供 UI 打字机上屏。
    """
    stream_fn = getattr(agent, "analyze_stream", None)
    if stream_fn is None:
        result = agent.analyze(image_path=path, question=question)
        yield {"type": "result", "result": result}
        return result

    result = None
    fields: dict[str, str] = {}  # 字段名 -> 当前可读文本（累积）
    for ev in stream_fn(image_path=path, question=question, force_hazard=True):
        etype = ev.get("type")
        if etype == "field_text":
            key = ev.get("key", "")
            fields[key] = ev.get("text", "")
            yield {"type": "stream", "text": _render_streaming_text(fields)}
        elif etype == "field_done":
            # 字段完整：优先用解析出的结构化值渲染成可读文本
            key = ev.get("key", "")
            value = ev.get("value")
            if isinstance(value, str):
                fields[key] = value
            elif isinstance(value, list) and key == "思维链":
                # 思维链是 [{步骤,内容},...]：格式化成可读的多步文本
                fields[key] = _render_thought_steps(value)
            elif isinstance(value, list):
                # 法律依据/整改措施等数组：拼接成可读文本
                fields[key] = _render_list_readable(value)
            yield {"type": "stream", "text": _render_streaming_text(fields)}
        elif etype == "done":
            result = ev.get("result")
            yield {"type": "result", "result": result}
    if result is None:
        raise RuntimeError("流式诊断未返回最终结果")
    return result


def _render_thought_steps(steps: list) -> str:
    """把 [{步骤,内容},...] 思维链数组渲染成可读的多步文本。"""
    lines = []
    for step in steps:
        if isinstance(step, dict):
            name = step.get("步骤", step.get("step", ""))
            content = step.get("内容", step.get("content", ""))
            lines.append(f"**{name}**：{content}" if name else str(content))
        else:
            lines.append(str(step))
    return "\n\n".join(lines)


def _render_list_readable(items: list) -> str:
    """把数组字段（法律依据/整改措施/参考案例）渲染成可读文本。"""
    lines = []
    for it in items:
        if isinstance(it, dict):
            # 法律依据：条款号 + 条款内容；整改措施：措施类型 + 措施内容
            if "条款号" in it:
                lines.append(f"- {it.get('条款号', '')}：{it.get('条款内容', '')}")
            elif "措施类型" in it:
                lines.append(f"- [{it.get('措施类型', '')}] {it.get('措施内容', '')}")
            else:
                lines.append("- " + "：".join(f"{k}:{v}" for k, v in it.items() if v))
        else:
            lines.append(f"- {it}")
    return "\n".join(lines)


def execute_plan_iter(
    client: RobotClient,
    ops: list[dict],
    *,
    question: str,
    agent=None,
    diagnose: bool = True,
    zone_name: str = "自定义指令",
) -> Generator[dict, None, None]:
    """
    逐步执行计划，并「拍一张 → 立即上传 → 立即流式回答这张 → 再拍下一张」。
    每完成一个阶段 yield 一次快照：
        {"status": str, "gallery": [(path, caption)], "report": str,
         "photo": (path, caption) 或 None, "answer": str 或 None, "done": bool}

    关键变化（相对「先拍完再统一诊断」的旧版）：
    1. 拍照后立即 yield photo=... 让界面第一时间回传显示（不再等到最后）；
    2. 诊断不再集中在尾部，而是在每次 capture/scan 拍完一张后立即跟一次流式诊断；
    3. answer 字段承载「当前这张」的流式回答文本，供 UI 打字机上屏。

    韧性设计（相对早期「失败即 return」的实现）：
    1. 完成检查：动作失败可知、照片必须落盘且体积达标，避免把空帧送去诊断；
    2. 失败恢复：站姿类动作失败先重试，再失败则 stop → recover 尝试自救；
    3. 降级而非终止：真正无法继续时跳出循环，已拍照片仍出诊断并保存报告。
    """
    shots: list[dict] = []
    log: list[str] = []
    gallery: list[tuple[str, str]] = []
    stats: dict[str, Any] = {
        "步骤总数": len(ops),
        "成功": 0,
        "失败": 0,
        "重试次数": 0,
        "有效照片": 0,
        "无效照片": 0,
        "提前结束": "",
    }
    aborted = ""
    consecutive_fail = 0
    # 记录是否处于「某一步骤中尚未完成移动后的拍照」——用于判断 scan 是否应即时诊断
    diagnose_inline = bool(diagnose and agent is not None)

    def snap(
        status: str,
        done: bool = False,
        streaming: str = "",
        streaming_label: str = "",
        done_title: str = "",
        photo: tuple | None = None,
        answer: str | None = None,
    ) -> dict:
        return {
            "status": status,
            "gallery": list(gallery),
            "photo": photo,
            "answer": answer,
            "streaming_text": streaming,
            "report": _render_running_report(
                zone_name, shots, log, client.mock, done, streaming, streaming_label, done_title
            ),
            "done": done,
        }

    def add_shot(shot: dict, label: str, caption_prefix: str) -> tuple | None:
        """完成检查通过才纳入，返回 (path, caption) 供立即回传；无效返回 None。"""
        ok, detail = shot_is_valid(shot)
        if not ok:
            stats["无效照片"] += 1
            log.append(f"❌ 抓拍无效（{label}）：{detail}")
            return None
        path = shot.get("path_win") or shot.get("path_linux")
        stats["有效照片"] += 1
        caption = f"{caption_prefix} · {label}"
        shots.append({"path": path, "file": shot.get("file"), "label": label})
        gallery.append((path, caption))
        log.append(f"✅ 抓拍有效（{label}）：`{shot.get('file')}`（{detail}）")
        return (path, caption)

    def diagnose_shot(item: dict, index: int) -> Generator[dict, None, None]:
        """对已纳入 shots 的某张照片立即做流式诊断，把结果写回 item 与 gallery。

        顺序：照片先上屏（streaming 起始即照片），随后回答逐字流式累积到照片下方。
        """
        photo_md = _embed_photo_md(item["path"], index, item["label"])
        yield snap(
            f"🧠 正在诊断第 {index} 张（{item['file']}）…",
            streaming=photo_md,
            streaming_label=f"📷 第 {index} 张 · {item['label']}",
            photo=(item["path"], f"第 {index} 张 · {item['label']}"),
        )
        try:
            for ev in _stream_diagnose_one(agent, item["path"], question):
                if ev.get("type") == "stream":
                    # 照片在上，回答文字逐字累积到下方
                    yield snap(
                        f"🧠 第 {index} 张诊断生成中…",
                        streaming=photo_md + "\n\n" + ev["text"],
                        streaming_label=f"🧠 第 {index} 张 · 诊断生成中（逐字流式）",
                    )
                elif ev.get("type") == "result":
                    result = ev["result"]
                    structured = result.get("structured_output") or {}
                    item["verdict"] = structured.get("隐患定性", "")
                    item["laws"] = structured.get("法律依据", [])
                    item["measures"] = structured.get("整改措施", [])
                    item["references"] = structured.get("参考案例", [])
                    item["thought"] = structured.get("思维链", "")
                    item["retrieved_laws"] = result.get("retrieved_laws", [])
                    item["citation_check"] = structured.get("引用校验")
                    item["severity"] = classify_severity(item["verdict"])
                    item["raw_answer"] = result.get("raw_answer", "")
                    item["ok_diag"] = bool(structured)
            if not item.get("ok_diag"):
                raise RuntimeError("诊断未产出结构化结论")
        except Exception as exc:
            item["error"] = str(exc)
            item["severity"] = "诊断失败"
            item["ok_diag"] = False

        log.append(
            f"诊断第 {index} 张 → "
            f"{'✅ ' + item.get('severity', '') if item.get('ok_diag') else '❌ ' + item.get('error', '')}"
        )
        gallery[index - 1] = (gallery[index - 1][0], f"{gallery[index - 1][1]} · {item.get('severity', '诊断失败')}")
        # 这张诊断完：给出富格式回答（answer 供 UI 上屏），照片内嵌在文字上方便于对应
        body_md = _render_one_verdict(item, index) if item.get("verdict") or item.get("error") else ""
        answer_md = photo_md + "\n\n" + body_md
        yield snap(
            f"🧠 第 {index} 张诊断完成（{item.get('severity', '—')}）",
            answer=answer_md,
            photo=None,
        )

    yield snap(f"准备执行 {len(ops)} 个步骤…")

    for idx, op in enumerate(ops, 1):
        kind = op["op"]

        if kind == "action":
            action = op["action"]
            critical = action in ("stand", "recover")
            ok, msg, attempts = act_with_retry(
                client, action, retries=ACTION_RETRY if critical else 0
            )
            stats["重试次数"] += max(0, attempts - 1)
            log.append(f"`{action}` → {'✅' if ok else '❌'} {msg}（尝试 {attempts} 次）")
            yield snap(f"[{idx}/{len(ops)}] 动作 {action} …")

            if ok:
                stats["成功"] += 1
                consecutive_fail = 0
            else:
                stats["失败"] += 1
                consecutive_fail += 1
                if critical:
                    log.append("⚠️ 关键动作失败，尝试姿态恢复")
                    if not recover_posture(client, log):
                        aborted = f"{action} 失败且姿态恢复未成功（避免趴地状态移动）"
                        break
                    log.append("♻️ 已恢复站姿，继续执行后续步骤")
                    consecutive_fail = 0

        elif kind == "move":
            ok, msg = client.move(op["vx"], op["vy"], op["vyaw"], op["duration"])
            log.append(f"移动 {op['duration']}s → {'✅' if ok else '❌'} {msg}")
            yield snap(f"[{idx}/{len(ops)}] 行进中（{op['duration']}s）…")
            if ok:
                stats["成功"] += 1
                consecutive_fail = 0
            else:
                stats["失败"] += 1
                consecutive_fail += 1
                client.act("stop")
                log.append("   恢复：已下发 stop 强制停止")
                if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                    aborted = f"连续 {consecutive_fail} 个步骤失败，停止走位"
                    break
                log.append("   本段走位跳过，继续后续步骤（位置可能偏移，报告中已标注）")

        elif kind == "turn":
            vyaw = TURN_VYAW if op["deg"] > 0 else -TURN_VYAW
            dur = _turn_seconds(op["deg"])
            ok, msg = client.move(0.0, 0.0, vyaw, round(dur, 2))
            log.append(f"转向 {op['deg']:.0f}°（{dur:.1f}s）→ {'✅' if ok else '❌'} {msg}")
            yield snap(f"[{idx}/{len(ops)}] 转向 {op['deg']:.0f}° …")
            if ok:
                stats["成功"] += 1
                consecutive_fail = 0
            else:
                stats["失败"] += 1
                consecutive_fail += 1
                log.append("   ⚠️ 转向未确认，后续拍摄视角可能偏移")

        elif kind == "capture":
            yield snap(f"[{idx}/{len(ops)}] 正在抓拍…")
            shot = client.capture(op.get("label", "shot"))
            added = add_shot(shot, op.get("label", "shot"), f"第 {len(shots) + 1} 张")
            if added:
                stats["成功"] += 1
                consecutive_fail = 0
                # 照片立即回传显示
                yield snap(f"[{idx}/{len(ops)}] 抓拍完成，照片已回传", photo=added)
                # 立即流式诊断这张（边拍边答）
                if diagnose_inline:
                    yield from diagnose_shot(shots[-1], len(shots))
            else:
                stats["失败"] += 1
                consecutive_fail += 1
                yield snap(f"[{idx}/{len(ops)}] 抓拍无效")

        elif kind == "scan":
            n = op["shots"]
            settle = op.get("settle", 1.2)
            before = stats["有效照片"]
            for k in range(n):
                yield snap(f"[{idx}/{len(ops)}] 扫描 {k + 1}/{n}：抓拍中…")
                shot = client.capture(f"scan{k + 1}")
                added = add_shot(shot, f"scan{k + 1}", f"第 {len(shots) + 1} 张")
                if added:
                    # 照片立即回传显示
                    yield snap(f"[{idx}/{len(ops)}] 扫描 {k + 1}/{n}：照片已回传", photo=added)
                    # 立即流式诊断这张（边拍边答）
                    if diagnose_inline:
                        yield from diagnose_shot(shots[-1], len(shots))
                else:
                    yield snap(f"[{idx}/{len(ops)}] 扫描 {k + 1}/{n}：抓拍无效")
                if k < n - 1:
                    turn_ok, turn_msg = client.move(
                        0.0, 0.0, TURN_VYAW, round(_turn_seconds(90), 2)
                    )
                    if not turn_ok:
                        log.append(f"   ⚠️ 扫描位转向失败：{turn_msg}，后续视角可能重叠")
                    time.sleep(settle)
            if stats["有效照片"] > before:
                stats["成功"] += 1
                consecutive_fail = 0
            else:
                stats["失败"] += 1
                consecutive_fail += 1

    # ---------------- 提前结束：降级但不停摆 ----------------
    if aborted:
        stats["提前结束"] = aborted
        log.append(f"⛔ 巡检提前结束：{aborted}")

    # ---------------- 兜底：若存在尚未诊断的照片（diagnose 关闭等），补一次诊断 ----------------
    if diagnose and agent is not None:
        for i, item in enumerate(shots, 1):
            if not item.get("verdict") and not item.get("error"):
                yield from diagnose_shot(item, i)

    report_path = save_report(zone_name, shots, log, client.mock, quality=stats)
    done_title = "⚠️ 巡检提前结束（已出部分报告）" if aborted else "✅ 巡检完成"
    final = snap(done_title, done=True, done_title=done_title)
    final["report"] += _render_quality_md(stats)
    final["report"] += f"\n\n---\n📄 报告已保存：`{report_path}`"
    final["report_path"] = str(report_path)
    final["quality"] = stats
    yield final


# ---------------------------------------------------------------------------
# 诊断结果整理
# ---------------------------------------------------------------------------
SEVERITY_RULES = [
    ("重大", r"重大隐患|重大事故隐患|重大风险|高风险|立即停工|停止作业"),
    ("较大", r"较大隐患|较大风险|中风险|严重|较大"),
    ("一般", r"一般隐患|一般风险|低风险|轻微|较小"),
]


def classify_severity(verdict: str) -> str:
    text = str(verdict or "")
    for level, pattern in SEVERITY_RULES:
        if re.search(pattern, text):
            return level
    return "待研判" if text else "未定性"


# ---------------------------------------------------------------------------
# 富格式渲染（与主对话「智能对话与检测」模板对齐）
# ---------------------------------------------------------------------------
def _render_law_item(law) -> str:
    """把一条法律依据渲染成卡片式 Markdown（含原文引用块）。"""
    if isinstance(law, dict):
        no = law.get("条款号", "")
        content = law.get("条款内容", "")
        apply_note = law.get("适用说明", "")
        block = f"**📜 {no}**\n\n"
        if content:
            block += f"> {content}\n\n"
        if apply_note:
            block += f"*📌 {apply_note}*\n"
        return block
    return f"- {law}\n"


def _render_measure(m) -> str:
    """把一条整改措施渲染成卡片式 Markdown。"""
    if isinstance(m, dict):
        mtype = m.get("措施类型", "措施")
        content = m.get("措施内容", "")
        owner = m.get("责任主体", "")
        deadline = m.get("时限要求", "")
        block = f"**🔧 [{mtype}]** {content}\n\n"
        meta = []
        if owner:
            meta.append(f"👤 责任主体：**{owner}**")
        if deadline:
            meta.append(f"⏱️ 时限：**{deadline}**")
        if meta:
            block += " ｜ ".join(meta) + "\n"
        return block
    return f"- {m}\n"


def _render_citation_check(check) -> str:
    """渲染法条引用核验结果（引用内容与检索原文的字符重合度）。"""
    if not isinstance(check, dict) or not check.get("条目"):
        return ""
    overall = str(check.get("总体结论", ""))
    head_icon = {"通过": "✅", "部分通过": "⚠️", "未通过": "❌"}.get(overall, "•")
    row_icon = {"已核实": "✅", "部分支持": "⚠️", "未获支持": "❌",
                "无检索依据": "❌", "无法核验": "○"}
    lines = [f"**🧾 法条引用核验 · {head_icon} {overall}**\n"]
    for row in check.get("条目", []):
        if not isinstance(row, dict):
            continue
        verdict = str(row.get("判定", ""))
        coverage = row.get("覆盖率")
        cov_text = "—" if coverage is None else f"{float(coverage):.0%}"
        lines.append(
            f"- {row_icon.get(verdict, '•')} {verdict} ｜ {row.get('条款号', '')} ｜ "
            f"覆盖率 {cov_text} ｜ 来源 `{row.get('最可能来源', '—')}`"
        )
    lines.append("")
    return "\n".join(lines)


def _render_thought(thought) -> str:
    """思维链折叠块。thought 可能是字符串，也可能是 [{步骤,内容},...] 数组。"""
    if not thought:
        return ""
    if isinstance(thought, list):
        lines = []
        for step in thought:
            if isinstance(step, dict):
                s_name = step.get("步骤", step.get("step", ""))
                s_content = step.get("内容", step.get("content", ""))
                lines.append(f"**{s_name}**：{s_content}")
            else:
                lines.append(str(step))
        thought_text = "\n\n".join(lines)
    else:
        thought_text = str(thought)
    return (
        "<details open class='sa-details'><summary>🧠 <b>分析推理过程</b>（点击折叠）</summary>\n\n"
        f"{thought_text}\n\n</details>\n\n---\n\n"
    )


def _render_one_verdict(shot: dict, idx: int) -> str:
    """把单张照片的诊断结论渲染成主对话同款富格式卡片。"""
    name = Path(str(shot.get("file") or "")).name
    verdict = str(shot.get("verdict") or shot.get("error") or "—")
    severity = shot.get("severity", "待研判")

    # 等级徽章配色（用于情绪化标记）
    sev_icon = {"重大": "🔴", "较大": "🟠", "一般": "🟡", "待研判": "⚪", "未定性": "⚪", "未知": "⚪", "诊断失败": "⚫"}.get(severity, "⚪")

    lines = [f"### 📷 第 {idx} 张 · {name}　{sev_icon} **{severity}**\n"]

    # 思维链
    thought = shot.get("thought") or (shot.get("structured") or {}).get("思维链")
    if thought:
        lines.append(_render_thought(thought))

    # 隐患定性
    if verdict:
        lines.append(f"#### 🚨 隐患定性\n\n{verdict}\n")

    # 法律依据（逐条卡片 + 原文引用）——来自知识库检索，保证不虚构
    laws = shot.get("laws") or []
    if laws:
        lines.append("#### 📖 法律依据（逐条原文引用）\n")
        for i, law in enumerate(laws, 1):
            lines.append(f"**{i}.** " + _render_law_item(law))
        lines.append("")

    # 引用核验：证明上面引用的条文确实来自检索库，而非模型改写
    check_md = _render_citation_check(shot.get("citation_check"))
    if check_md:
        lines.append(check_md)

    # 整改措施
    measures = shot.get("measures") or []
    if measures:
        lines.append("#### 🛠️ 整改措施（可执行 · 可复核）\n")
        for m in measures:
            lines.append("- " + _render_measure(m))
        lines.append("")

    # 参考案例
    refs = shot.get("references") or []
    if isinstance(refs, str):
        # LLM 偶发把参考案例输出成字符串而非数组，拆行兜底，避免逐字符渲染
        refs = [p.strip() for p in refs.split("\n") if p.strip()]
    if refs:
        lines.append("#### 🗂️ 参考案例\n")
        for ref in refs:
            lines.append(f"- {ref}\n")
        lines.append("")

    # 检索增强来源（RAG 溯源，可选）
    retrieved = shot.get("retrieved_laws") or []
    if retrieved:
        lines.append("<details class='sa-details sa-rag'><summary>🔍 <b>检索增强来源 (RAG Sources)</b>（点击展开）</summary>\n\n")
        lines.append("**📜 命中法条：**\n")
        for i, law in enumerate(retrieved, 1):
            if isinstance(law, dict):
                src = law.get("source_file", "未知源")
                content = law.get("legal_basis", "")
                score = law.get("score", 0)
                lines.append(f"{i}. `{src}`（距离 {score:.2f}）\n   > {content[:160]}{'…' if len(content) > 160 else ''}\n\n")
            else:
                lines.append(f"{i}. {law}\n")
        lines.append("</details>\n")

    lines.append("\n---\n")
    return "\n".join(lines)


def _embed_photo_md(path: str, index: int, label: str) -> str:
    """把一张照片内嵌进聊天消息（markdown 图片），放在回答文字上方，可点击放大。

    使用 Gradio 的 ``/gradio_api/file=<绝对路径>`` 端点，确保本地照片能被
    前端正确加载（相对路径/``file://`` 在 Chatbot markdown 中支持不稳定）。
    """
    from urllib.parse import quote

    abs_path = str(Path(path).resolve())
    src = f"/gradio_api/file={quote(abs_path, safe='')}"
    caption = f"📷 第 {index} 张 · {label}"
    return f"![{caption}]({src})"


def _render_running_report(
    zone_name: str,
    shots: list[dict],
    log: list[str],
    is_mock: bool,
    done: bool,
    streaming: str = "",
    streaming_label: str = "",
    done_title: str = "",
) -> str:
    title = done_title or "✅ 巡检完成"
    head = f"### 🤖 巡检进行中 · {zone_name}\n\n" if not done else f"### {title} · {zone_name}\n\n"
    if is_mock:
        head += "> ⚠️ **演示模式**：照片来自本地样本库，狗并未真实移动。\n\n"
    head += f"已采集 **{len(shots)}** 张照片\n\n"
    if log:
        head += "<details><summary>执行日志</summary>\n\n" + "\n".join(f"- {x}" for x in log[-25:]) + "\n\n</details>\n\n"
    # 正在流式生成中的当前照片诊断（进行中逐字上屏）
    if streaming:
        label = streaming_label or "🧠 正在逐字生成诊断…"
        head += f"\n#### {label}\n\n> {streaming}\n\n"
    # 仅在最终报告（done=True）时渲染全部诊断卡片；进行中的逐张结论由 answer 字段即时推送
    if done and shots and any(s.get("verdict") for s in shots):
        head += _render_findings(shots)
    return head


def _render_findings(shots: list[dict]) -> str:
    """把诊断结果渲染成主对话同款富格式（卡片 + 法条原文 + 整改措施）。

    保留一张简表作为「隐患速览」，主体是逐张的完整诊断卡片。
    """
    order = {"重大": 0, "较大": 1, "一般": 2, "待研判": 3, "未定性": 4, "未知": 5, "诊断失败": 6}
    ranked = sorted(shots, key=lambda s: order.get(s.get("severity", "未知"), 9))

    # 速览表
    lines = ["#### 🚨 隐患排查速览\n", "| # | 照片 | 定性 | 等级 |", "|---|---|---|---|"]
    for i, s in enumerate(ranked, 1):
        name = Path(str(s.get("file") or "")).name
        verdict = str(s.get("verdict") or s.get("error") or "—").replace("\n", " ")
        verdict = (verdict[:60] + "…") if len(verdict) > 60 else verdict
        lines.append(f"| {i} | `{name}` | {verdict} | **{s.get('severity', '—')}** |")
    lines.append("")

    # 逐张完整诊断卡片
    lines.append("#### 🔍 逐张诊断详情\n")
    for i, s in enumerate(ranked, 1):
        if not s.get("verdict") and not s.get("error"):
            continue
        lines.append(_render_one_verdict(s, i))
    return "\n".join(lines)


def _image_to_jpeg_stream(path: str) -> "BytesIO | None":
    """把本地图片转成标准 JPEG 字节流，供 python-docx 插入。

    Go2 摄像头输出的 JPEG 带非标准 APP0 标记（`AVI1` 而非 `JFIF`），python-docx 的
    图片头识别器（_ImageHeaderFactory）不认这种变体，会抛 UnrecognizedImageError。
    这里用 Pillow 统一重编码为标准 JPEG，转成内存流，保证 add_picture 一定能插入。
    """
    from io import BytesIO

    try:
        from PIL import Image
    except Exception:
        return None
    try:
        img = Image.open(str(path))
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return buf
    except Exception:
        return None


def _build_docx_report(zone_name: str, shots: list[dict], log: list[str], is_mock: bool, counts: dict[str, int]) -> Path:
    """用 python-docx 生成「隐患排查建议报告.docx」。"""
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn

    order = {"重大": 0, "较大": 1, "一般": 2, "待研判": 3, "未定性": 4, "未知": 5, "诊断失败": 6}
    ranked = sorted(shots, key=lambda s: order.get(s.get("severity", "未知"), 9))

    doc = Document()
    # 全局默认字体（中文宋体 + 西文 Times New Roman）
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # 标题样式统一黑色（Word 内置 Heading/Title 样式默认带蓝色主题色，这里强制黑）
    for _sn in ("Title", "Heading 1", "Heading 2", "Heading 3", "Heading 4"):
        try:
            _st = doc.styles[_sn]
            _st.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
        except Exception:
            pass

    def set_cn_font(run, name="宋体"):
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), name)

    # 标题
    title = doc.add_heading("隐患排查建议报告", level=0)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
        set_cn_font(run, "微软雅黑")

    # 概要
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run(f"巡检区域：{zone_name}　｜　报告时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)

    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("一、巡检概要")
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    set_cn_font(r, "微软雅黑")

    info = doc.add_paragraph()
    info.add_run(f"采集照片：{len(shots)} 张　　")
    info.add_run(f"运行模式：{'演示模式（样本图）' if is_mock else '真机模式'}　　")
    info.add_run(f"等级分布：{'、'.join(f'{k} {v}' for k, v in counts.items()) or '无'}")

    # 速览表
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("二、隐患排查速览")
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    set_cn_font(r, "微软雅黑")

    if ranked:
        table = doc.add_table(rows=1, cols=4)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        for j, htxt in enumerate(["序号", "照片", "隐患定性", "等级"]):
            hdr[j].text = htxt
            for para in hdr[j].paragraphs:
                for run in para.runs:
                    run.bold = True
        for i, s in enumerate(ranked, 1):
            name = Path(str(s.get("file") or "")).name
            verdict = str(s.get("verdict") or s.get("error") or "—").replace("\n", " ")
            verdict = verdict[:80] + ("…" if len(verdict) > 80 else "")
            cells = table.add_row().cells
            cells[0].text = str(i)
            # 照片列：插入真实缩略图（若路径有效），否则回退为文件名文字
            photo_path = s.get("path") or s.get("path_win") or s.get("path_linux")
            pic_inserted = False
            if photo_path and Path(str(photo_path)).exists():
                img_stream = _image_to_jpeg_stream(str(photo_path))
                if img_stream is not None:
                    try:
                        cell_para = cells[1].paragraphs[0]
                        cell_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        cell_run = cell_para.add_run()
                        cell_run.add_picture(img_stream, width=Inches(1.2))
                        pic_inserted = True
                    except Exception:
                        pic_inserted = False
            if not pic_inserted:
                cells[1].text = name
            cells[2].text = verdict
            cells[3].text = str(s.get("severity", "—"))

    # 逐张诊断详情
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("三、逐张诊断详情")
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    set_cn_font(r, "微软雅黑")

    for i, s in enumerate(ranked, 1):
        name = Path(str(s.get("file") or "")).name
        verdict = str(s.get("verdict") or s.get("error") or "—")
        severity = s.get("severity", "待研判")

        # 小标题
        h = doc.add_heading(level=2)
        hr = h.add_run(f"{i}. {name}（{severity}）")
        hr.font.size = Pt(12)
        hr.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
        set_cn_font(hr, "微软雅黑")

        # 现场照片（正文内嵌，居中显示）
        photo_path = s.get("path") or s.get("path_win") or s.get("path_linux")
        if photo_path and Path(str(photo_path)).exists():
            img_stream = _image_to_jpeg_stream(str(photo_path))
            if img_stream is not None:
                try:
                    pic_para = doc.add_paragraph()
                    pic_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = pic_para.add_run()
                    run.add_picture(img_stream, width=Inches(4.5))
                    cap = doc.add_paragraph()
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cr = cap.add_run(f"▲ 现场照片 · {name}")
                    cr.font.size = Pt(9)
                    cr.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                    set_cn_font(cr)
                except Exception:
                    # 图片插入失败（路径无效/损坏）时跳过，不中断报告生成
                    pass

        # 隐患定性
        p = doc.add_paragraph()
        pr = p.add_run("【隐患定性】")
        pr.bold = True
        p.add_run(verdict)

        # 法律依据（逐条原文）
        laws = s.get("laws") or []
        if isinstance(laws, str):
            laws = [laws]
        if laws:
            p = doc.add_paragraph()
            pr = p.add_run("【法律依据】")
            pr.bold = True
            for law in laws:
                if isinstance(law, dict):
                    no = law.get("条款号", "")
                    content = law.get("条款内容", "")
                    note = law.get("适用说明", "")
                    li = doc.add_paragraph(style="List Bullet")
                    li.add_run(no).bold = True
                    if content:
                        li.add_run(" " + content)
                    if note:
                        li.add_run(f"（{note}）")
                else:
                    doc.add_paragraph(f"- {law}", style="List Bullet")

        # 整改措施
        measures = s.get("measures") or []
        if isinstance(measures, str):
            measures = [measures]
        if measures:
            p = doc.add_paragraph()
            pr = p.add_run("【整改措施】")
            pr.bold = True
            for m in measures:
                if isinstance(m, dict):
                    mtype = m.get("措施类型", "措施")
                    content = m.get("措施内容", "")
                    owner = m.get("责任主体", "")
                    deadline = m.get("时限要求", "")
                    meta = "　".join(x for x in [owner, deadline] if x)
                    li = doc.add_paragraph(style="List Bullet")
                    li.add_run(f"[{mtype}] ").bold = True
                    li.add_run(content)
                    if meta:
                        li.add_run(f"（{meta}）")
                else:
                    doc.add_paragraph(f"- {m}", style="List Bullet")

        # 参考案例
        refs = s.get("references") or []
        if isinstance(refs, str):
            refs = [p.strip() for p in refs.split("\n") if p.strip()]
        if refs:
            p = doc.add_paragraph()
            pr = p.add_run("【参考案例】")
            pr.bold = True
            for ref in refs:
                doc.add_paragraph(f"- {ref}", style="List Bullet")

    # 执行日志
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("四、执行日志")
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    set_cn_font(r, "微软雅黑")
    for x in log:
        doc.add_paragraph(x, style="List Number")

    return doc


def save_report(
    zone_name: str,
    shots: list[dict],
    log: list[str],
    is_mock: bool,
    quality: dict | None = None,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(__file__).resolve().parents[2] / "data" / "robot_inspections" / f"{ts}_{zone_name}"
    base.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for s in shots:
        lvl = s.get("severity", "未知")
        counts[lvl] = counts.get(lvl, 0) + 1

    # 生成 Word 报告（下载物）
    docx_path = base / "隐患排查建议报告.docx"
    try:
        doc = _build_docx_report(zone_name, shots, log, is_mock, counts)
        doc.save(str(docx_path))
    except Exception as exc:
        # docx 生成失败时回退到 markdown，避免巡检流程中断
        docx_path = base / "隐患排查建议报告.md"
        docx_path.write_text(f"# 隐患排查建议报告 · {zone_name}\n\n（Word 生成失败：{exc}，已降级为纯文本）\n", encoding="utf-8")

    # 同时保留一份 markdown 详情（内部备用，不作为下载物）
    md = [f"# 机器狗巡检报告 · {zone_name}", ""]
    md.append(f"- 时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    md.append(f"- 采集照片：{len(shots)} 张")
    md.append(f"- 运行模式：{'演示模式（样本图）' if is_mock else '真机模式'}")
    md.append(f"- 等级分布：{ '、'.join(f'{k} {v}' for k, v in counts.items()) or '无' }")
    md.append("")
    md.append(_render_findings(shots) if shots else "_本次未采集到照片_")
    if quality:
        md.append(_render_quality_md(quality))
    md.append("\n## 执行日志\n")
    md.extend(f"{i}. {x}" for i, x in enumerate(log, 1))
    (base / "report.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "zone": zone_name,
        "time": datetime.now().isoformat(timespec="seconds"),
        "mock": is_mock,
        "counts": counts,
        "quality": quality or {},
        "shots": shots,
        "log": log,
    }
    (base / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return docx_path


# ---------------------------------------------------------------------------
# 固定航点导航巡检（航点即巡检点）
# ---------------------------------------------------------------------------
def navigate_waypoints_iter(
    client: RobotClient,
    config: NavConfig,
    *,
    question: str,
    agent=None,
    diagnose: bool = True,
    zone_name: str = "航点巡检",
) -> Generator[dict, None, None]:
    """按固定航点闭环导航，每个航点到达后拍照 + 诊断（航点即巡检点）。

    与 execute_plan_iter 的区别：走位不再靠「速度 × 时间」开环估算，而是
    调用 WaypointNavigator 闭环导航（读位置 → 朝向纠偏 → 前进 → 到达判定）。
    诊断与报告逻辑复用现有实现。

    快照协议与 execute_plan_iter 一致：
        {"status", "gallery", "photo", "answer", "streaming_text", "report", "done"}
    """
    shots: list[dict] = []
    log: list[str] = []
    gallery: list[tuple[str, str]] = []
    stats: dict[str, Any] = {
        "步骤总数": len(config.waypoints),
        "成功": 0,
        "失败": 0,
        "重试次数": 0,
        "有效照片": 0,
        "无效照片": 0,
        "提前结束": "",
    }

    def snap(status: str, done: bool = False, streaming: str = "",
             streaming_label: str = "", done_title: str = "",
             photo: tuple | None = None, answer: str | None = None) -> dict:
        return {
            "status": status,
            "gallery": list(gallery),
            "photo": photo,
            "answer": answer,
            "streaming_text": streaming,
            "report": _render_running_report(
                zone_name, shots, log, client.mock, done, streaming, streaming_label, done_title
            ),
            "done": done,
        }

    def add_shot(shot: dict, label: str) -> tuple | None:
        ok, detail = shot_is_valid(shot)
        if not ok:
            stats["无效照片"] += 1
            log.append(f"❌ 抓拍无效（{label}）：{detail}")
            return None
        path = shot.get("path_win") or shot.get("path_linux")
        stats["有效照片"] += 1
        caption = f"航点 {label} · 巡检点"
        shots.append({"path": path, "file": shot.get("file"), "label": label})
        gallery.append((path, caption))
        log.append(f"✅ 抓拍有效（{label}）：`{shot.get('file')}`（{detail}）")
        return (path, caption)

    # 复用 execute_plan_iter 的单张流式诊断闭包（避免重复实现）
    # 这里重新定义一个轻量版，行为与 execute_plan_iter 内一致。
    def diagnose_shot(item: dict, index: int) -> Generator[dict, None, None]:
        photo_md = _embed_photo_md(item["path"], index, item["label"])
        yield snap(
            f"🧠 正在诊断第 {index} 张（{item['file']}）…",
            streaming=photo_md,
            streaming_label=f"📷 第 {index} 张 · {item['label']}",
            photo=(item["path"], f"第 {index} 张 · {item['label']}"),
        )
        try:
            for ev in _stream_diagnose_one(agent, item["path"], question):
                if ev.get("type") == "stream":
                    yield snap(
                        f"🧠 第 {index} 张诊断生成中…",
                        streaming=photo_md + "\n\n" + ev["text"],
                        streaming_label=f"🧠 第 {index} 张 · 诊断生成中（逐字流式）",
                    )
                elif ev.get("type") == "result":
                    result = ev["result"]
                    structured = result.get("structured_output") or {}
                    item["verdict"] = structured.get("隐患定性", "")
                    item["laws"] = structured.get("法律依据", [])
                    item["measures"] = structured.get("整改措施", [])
                    item["references"] = structured.get("参考案例", [])
                    item["thought"] = structured.get("思维链", "")
                    item["retrieved_laws"] = result.get("retrieved_laws", [])
                    item["citation_check"] = structured.get("引用校验")
                    item["severity"] = classify_severity(item["verdict"])
                    item["raw_answer"] = result.get("raw_answer", "")
                    item["ok_diag"] = bool(structured)
            if not item.get("ok_diag"):
                raise RuntimeError("诊断未产出结构化结论")
        except Exception as exc:
            item["error"] = str(exc)
            item["severity"] = "诊断失败"
            item["ok_diag"] = False

        log.append(
            f"诊断第 {index} 张 → "
            f"{'✅ ' + item.get('severity', '') if item.get('ok_diag') else '❌ ' + item.get('error', '')}"
        )
        gallery[index - 1] = (gallery[index - 1][0], f"{gallery[index - 1][1]} · {item.get('severity', '诊断失败')}")
        body_md = _render_one_verdict(item, index) if item.get("verdict") or item.get("error") else ""
        yield snap(
            f"🧠 第 {index} 张诊断完成（{item.get('severity', '—')}）",
            answer=photo_md + "\n\n" + body_md,
            photo=None,
        )

    nav = WaypointNavigator(
        client,
        config,
        on_progress=lambda step, name, pose, dist: log.append(
            f"  导航中 [{name}] step={step} dist={dist:.2f}m"
        ) if pose is not None else None,
    )

    # 任务原点标定：把「点开始这一刻」狗的位姿记为原点与朝向基准，
    # 使航点坐标恒为「从当前位姿出发」，不再依赖上电朝向。
    yield snap("正在标定导航起点（记录当前位姿为原点）…")
    if not nav._calibrate():
        yield snap("⚠️ 无法读取当前位置，导航起点标定失败", done=True,
                   done_title="⚠️ 标定失败")
        final = snap("⚠️ 标定失败：无法读取狗当前位置", done=True)
        final["report"] += f"\n\n---\n❌ 位置读取失败，请确认狗已连接且 /pose 数据有效"
        yield final
        return

    yield snap(f"开始航点巡检，共 {len(config.waypoints)} 个航点…")

    # ---- 阶段一：先导航到所有航点并拍照（不诊断，导航阶段不被云端 LLM 阻塞） ----
    for i, wp in enumerate(config.waypoints, 1):
        yield snap(f"[{i}/{len(config.waypoints)}] 正在导航到「{wp.name}」({wp.x}, {wp.y})…")
        result = nav.navigate_to(wp)
        log.append(
            f"航点「{wp.name}」→ {'✅ ' + result.message if result.reached else '❌ ' + result.message}"
            f"（{result.steps} 步 / {result.elapsed_s:.1f}s）"
        )
        if not result.reached:
            stats["失败"] += 1
            yield snap(f"[{i}/{len(config.waypoints)}] 「{wp.name}」导航失败，跳过巡检")
            continue

        stats["成功"] += 1
        # 航点即巡检点：到达后拍照（只拍照，诊断留到阶段二统一做）
        yield snap(f"[{i}/{len(config.waypoints)}] 「{wp.name}」已到达，正在抓拍…")
        shot = client.capture(wp.name)
        added = add_shot(shot, wp.name)
        if added:
            yield snap(f"[{i}/{len(config.waypoints)}] 「{wp.name}」抓拍完成", photo=added)
        else:
            stats["失败"] += 1
            yield snap(f"[{i}/{len(config.waypoints)}] 「{wp.name}」抓拍无效")

    # ---- 阶段二：统一批量诊断（复用流式，逐张增量上屏） ----
    if diagnose and agent is not None and shots:
        yield snap(f"导航完成，开始统一诊断 {len(shots)} 张照片…")
        for i, item in enumerate(shots, 1):
            yield from diagnose_shot(item, i)

    report_path = save_report(zone_name, shots, log, client.mock, quality=stats)
    done_title = "✅ 航点巡检完成" if stats["失败"] == 0 else "⚠️ 航点巡检完成（部分航点失败）"
    final = snap(done_title, done=True, done_title=done_title)
    final["report"] += _render_quality_md(stats)
    final["report"] += f"\n\n---\n📄 报告已保存：`{report_path}`"
    final["report_path"] = str(report_path)
    final["quality"] = stats
    yield final
