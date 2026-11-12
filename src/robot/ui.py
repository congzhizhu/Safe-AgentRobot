# -*- coding: utf-8 -*-
"""
机器狗巡检面板（Gradio Tab · 聊天式界面）

与主对话「💬 智能对话与检测」保持一致的 ChatInterface 聊天界面：
- 顶部一行紧凑连接状态 + 急停按钮
- 中间一个聊天窗口，用户用自然语言指挥狗 / 选区域一键巡检 / 手动微调
- 智能体回复复用主对话 chat_with_agent 同款富格式模板（思维链折叠 +
  🚨隐患定性 + 📖法律依据逐条原文 + 🛠️整改措施 + 🗂️参考案例 + 🔍RAG溯源），
  并支持逐字流式打字机输出
- 报告下载区缩小为一条紧凑卡片 + 小尺寸下载按钮

三条业务链路（功能不变，全部迁入对话流）：
A. 区域巡检（一键）     选区域 → 狗走位+四向扫描 → 逐张诊断 → 出报告
B. 自然语言指令         直接对话下指令 → 智能体解析 → 执行 → 诊断
C. 手动微调             快捷按钮：站立/坐下/趴下/前进/后退/左转/右转/拍照诊断/急停
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import gradio as gr

from robot.client import RobotClient
from robot.navigator import NavConfig, WaypointNavigator
from robot.patrol import (
    DEFAULT_SHOTS,
    ZONES,
    classify_severity,
    describe_plan,
    execute_plan_iter,
    navigate_waypoints_iter,
    plan_with_llm,
    validate_plan,
    zone_choices,
    _embed_photo_md,
    _render_citation_check,
    _render_one_verdict,
)


def _status_md(snap) -> str:
    if snap.mock:
        return (
            "**🔌 连接状态**：⚪ 控狗服务未启动 · 🧪 演示模式（样本图）\n\n"
            f"> {snap.note}\n\n"
            "启动真机：WSL 里执行 `cd /mnt/d/Safety_Agent/robot && python3 robot_service.py --nic auto`"
        )
    if not snap.ok:
        return f"**🔌 连接状态**：❌ {snap.note}"
    cam = {True: "✅ 正常", False: "❌ 异常", None: "⚪ 未测试"}.get(snap.camera_ok, "⚪ 未测试")
    link = "🟢 已连接" if snap.connected else "🟡 服务在线 / 狗未握手"
    parts = [
        f"**🔌 连接状态**：🟢 控狗服务在线（127.0.0.1:8010）· 机器狗 {link} · 摄像头 {cam}",
    ]
    if snap.note:
        parts.append(f"> {snap.note}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 与主对话 chat_with_agent 一致的富格式渲染
# ---------------------------------------------------------------------------
def _render_law_item(law) -> str:
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


def _render_assistant_reply(result: dict) -> str:
    """把 agent.analyze() 的结果渲染成主对话同款结构化富格式。"""
    structured = result.get("structured_output") or {}
    raw_answer = result.get("raw_answer", "未能生成结果")

    reply = ""
    if isinstance(structured, dict) and ("隐患定性" in structured or "思维链" in structured):
        thought = structured.get("思维链", "")
        if thought:
            reply += (
                "<details open class='sa-details'><summary>🧠 <b>分析推理过程</b>（点击折叠）</summary>\n\n"
                f"{thought}\n\n</details>\n\n---\n\n"
            )

        verdict = structured.get("隐患定性", "")
        if verdict:
            reply += f"### 🚨 隐患定性\n\n{verdict}\n\n"

        laws = structured.get("法律依据", [])
        if laws:
            reply += "### 📖 法律依据（逐条原文引用）\n\n"
            for i, law in enumerate(laws, 1):
                reply += f"**{i}.** " + _render_law_item(law) + "\n"
            reply += "\n"

        check_md = _render_citation_check(structured.get("引用校验"))
        if check_md:
            reply += "### 🧾 法条引用核验\n\n" + check_md

        measures = structured.get("整改措施", [])
        if measures:
            reply += "### 🛠️ 整改措施（可执行 · 可复核）\n\n"
            for m in measures:
                reply += "- " + _render_measure(m)
            reply += "\n"

        refs = structured.get("参考案例", [])
        if isinstance(refs, str):
            # LLM 偶发把参考案例输出成字符串而非数组，拆行兜底，避免逐字符渲染
            refs = [p.strip() for p in refs.split("\n") if p.strip()]
        if refs:
            reply += "### 🗂️ 参考案例\n\n"
            for ref in refs:
                reply += f"- {ref}\n"
            reply += "\n"
    else:
        reply += f"{raw_answer}\n\n"

    cases = result.get("similar_cases", [])
    laws = result.get("retrieved_laws", [])
    if cases or laws:
        reply += (
            "<details class='sa-details sa-rag'><summary>🔍 <b>检索增强来源 (RAG Sources)</b>（点击展开）</summary>\n\n"
        )
        if cases:
            reply += "**📷 相似案卷：**\n"
            for i, case in enumerate(cases, 1):
                reply += f"{i}. 相似度 `{case.get('score', 0):.2f}`\n"
            reply += "\n"
        if laws:
            reply += "**📜 命中法条：**\n"
            for i, law in enumerate(laws, 1):
                if isinstance(law, dict):
                    src = law.get("source_file", "未知源")
                    content = law.get("legal_basis", "")
                    score = law.get("score", 0)
                    reply += f"{i}. `{src}`（距离 {score:.2f}）\n   > {content[:160]}{'…' if len(content) > 160 else ''}\n\n"
                else:
                    reply += f"{i}. {law}\n"
        reply += "</details>\n"

    return reply


# ---------------------------------------------------------------------------
# 聊天处理函数（供 ChatInterface 调用，流式）
# ---------------------------------------------------------------------------
def _strip_report_path(report: str) -> str:
    """去掉报告末尾的「报告已保存：路径」行。

    下载入口由独立 File 组件提供，聊天流里重复显示绝对路径既拥挤又易泄露
    本地目录结构，因此在渲染前统一剥离（含其前的分隔线）。
    """
    text = str(report or "")
    marker = "📄 报告已保存"
    if marker in text:
        text = text.split(marker)[0]
    text = text.rstrip()
    if text.endswith("---"):
        text = text[:-3].rstrip()
    return text


def _make_inspect_chat_fn(client: RobotClient, agent):
    """自然语言指令 → 智能体解析 → 执行 → 拍一张立即回传+立即流式回答。"""

    def inspect_fn(message, history) -> Generator[tuple[str, list, str], None, None]:
        text = (message or "").strip()
        if not text:
            yield "请输入指令，例如「向前走 2 秒，然后原地扫描一圈」，或点击上方「区域巡检」/快捷按钮。", [], None
            return

        ops, say = plan_with_llm(agent, text)
        if not ops:
            yield f"⚠️ {say}\n\n换个说法试试，或点击上方「区域巡检」一键执行。", [], None
            return

        plan = validate_plan(ops)
        if not plan:
            yield "⚠️ 没有可执行的计划，请重新描述指令。", [], None
            return

        yield f"**🧠 智能体理解**：{say}\n\n**动作计划**\n\n{describe_plan(plan)}\n\n---\n\n", [], None

        question = (
            "请分析这张机器狗巡检拍摄的现场照片，判断是否存在安全隐患并进行隐患定性，"
            "同时给出法律依据、整改措施与参考案例。"
        )
        for snap in execute_plan_iter(
            client, plan, question=question, agent=agent, diagnose=True, zone_name="自然语言指令"
        ):
            gallery = snap.get("gallery") or []
            if snap.get("done"):
                report = _strip_report_path(snap.get("report", ""))
                report_path = snap.get("report_path")
                yield report, gallery, report_path
                return
            # 逐张流式回答：answer 非空时推这张的富格式结论
            answer = snap.get("answer")
            if answer:
                yield answer, gallery, None
                continue
            status = snap.get("status", "")
            # 诊断生成中：只推流式打字机增量文本（不带标题，避免闪烁）
            if "诊断" in status and "完成" not in status:
                seg = (snap.get("streaming_text") or "").strip()
                if seg:
                    yield seg, gallery, None
                continue
            if status:
                yield f"\n\n> {status}\n", gallery, None

    return inspect_fn


def _make_zone_chat_fn(client: RobotClient, agent):
    """区域巡检一键：选区域 → 走位扫描 → 拍一张立即回传+立即流式回答。"""

    def zone_fn(zone_key: str, shots: int, history) -> Generator[tuple[str, list], None, None]:
        zone = ZONES.get(zone_key)
        if not zone:
            yield "⚠️ 请先选择巡检区域。", []
            return

        ops = []
        for op in zone.route:
            if op["op"] == "scan":
                ops.append({"op": "scan", "shots": int(shots), "settle": op.get("settle", 1.3)})
            else:
                ops.append(dict(op))
        plan = validate_plan(ops)

        yield (
            f"### 🚀 开始巡检 · {zone.name}\n\n"
            f"{zone.desc}\n\n**动作计划**\n\n{describe_plan(plan)}\n\n---\n\n",
            [],
        )

        question = f"请分析这张机器狗巡检拍摄的现场照片（区域：{zone.name}），判断是否存在安全隐患并进行隐患定性。"
        for snap in execute_plan_iter(
            client, plan, question=question, agent=agent, diagnose=True, zone_name=zone.key
        ):
            gallery = snap.get("gallery") or []
            if snap.get("done"):
                report = _strip_report_path(snap.get("report", ""))
                yield report, gallery
                return
            answer = snap.get("answer")
            if answer:
                yield answer, gallery
                continue
            status = snap.get("status", "")
            if "诊断" in status and "完成" not in status:
                seg = (snap.get("streaming_text") or "").strip()
                if seg:
                    yield seg, gallery
                continue
                continue
            if status:
                yield f"\n\n> {status}\n", gallery

    return zone_fn


def _make_manual_chat_fn(client: RobotClient, agent):
    """手动微调：单步动作 + 拍照诊断（流式 + 照片回传）。"""

    def manual_fn(action: str, question: str, history) -> Generator[tuple[str, list], None, None]:
        if action == "shot":
            yield "> 📷 正在抓拍…", []
            shot = client.capture("manual")
            if not shot.get("ok"):
                yield f"❌ 抓拍失败：{shot.get('error')}", []
                return
            path = shot.get("path_win") or shot.get("path_linux")
            gallery = [(path, shot.get("file", "现场照片"))]
            try:
                stream_fn = getattr(agent, "analyze_stream", None)
                if stream_fn is not None:
                    accumulated = ""
                    result = None
                    for ev in stream_fn(
                        image_path=path,
                        question=question or "请分析这张现场照片是否存在安全隐患。",
                        force_hazard=True,
                    ):
                        etype = ev.get("type")
                        if etype == "raw_delta":
                            accumulated += ev.get("text", "")
                            yield accumulated, gallery
                        elif etype == "field_text":
                            accumulated = ev.get("text", "")
                            yield accumulated, gallery
                        elif etype == "done":
                            result = ev.get("result")
                    if result is None:
                        raise RuntimeError("流式诊断未返回最终结果")
                    yield _embed_photo_md(path, 1, shot.get("file", "现场照片")) + "\n\n" + _render_assistant_reply(result), gallery
                else:
                    result = agent.analyze(image_path=path, question=question or "请分析这张现场照片是否存在安全隐患。")
                    yield _embed_photo_md(path, 1, shot.get("file", "现场照片")) + "\n\n" + _render_assistant_reply(result), gallery
            except Exception as exc:
                yield f"❌ 诊断失败：{exc}", gallery
            return

        ok, msg = client.act(action)
        yield f"{'✅' if ok else '❌'} `{action}` — {msg}", []

    return manual_fn


# ---------------------------------------------------------------------------
# 航点导航
# ---------------------------------------------------------------------------
def _nav_config_files(project_root: Path) -> list[str]:
    """列出 configs/ 下所有航点配置文件（nav_*.yaml），返回展示名列表。"""
    cfg_dir = project_root / "configs"
    files = sorted(cfg_dir.glob("nav_*.yaml")) if cfg_dir.exists() else []
    return [str(f) for f in files]


def _nav_config_choices(project_root: Path) -> list[str]:
    """返回下拉框选项：显示文件名，值为完整路径（用 value/label 分离）。"""
    return [str(f.name) for f in Path(project_root, "configs").glob("nav_*.yaml")] if (project_root / "configs").exists() else []


# ---------------------------------------------------------------------------
# 构建巡检 Tab
# ---------------------------------------------------------------------------
def build_robot_tab(agent, project_root: Path) -> None:
    """在调用方的 `with gr.Tab(...)` 上下文中构建聊天式巡检面板。"""
    client = RobotClient(project_root=project_root, allow_mock=True)

    # ---------------- 顶部：紧凑状态 + 急停 ----------------
    status_md = gr.Markdown(_status_md(client.snapshot()))
    with gr.Row():
        refresh_btn = gr.Button("🔄 刷新连接状态", size="sm", scale=1)
        estop_btn = gr.Button("🛑 急停（Damp）", variant="stop", size="sm", scale=1)

    # ---------------- 快捷控件区（紧凑，聊天框上方） ----------------
    with gr.Row():
        zone_dd = gr.Dropdown(
            choices=zone_choices(),
            value="scan_only",
            label="🗺️ 区域巡检（一键）",
            scale=3,
        )
        shots_sl = gr.Slider(1, 6, value=DEFAULT_SHOTS, step=1, label="每个巡视位拍几张", scale=2)
        zone_btn = gr.Button("🚀 开始巡检", variant="primary", scale=1, size="sm")

    with gr.Row():
        nav_dd = gr.Dropdown(
            choices=_nav_config_choices(project_root),
            value=(_nav_config_choices(project_root) or [None])[0],
            label="🧭 航点导航（闭环 · 航点即巡检点）",
            scale=3,
        )
        nav_info = gr.Markdown(
            "航点导航按 `configs/nav_*.yaml` 里的航点序列闭环导航，"
            "每个航点到达后自动拍照 + 诊断。航点坐标是相对狗上电原点的偏移。",
            scale=4,
        )
        nav_btn = gr.Button("🧭 开始航点导航", variant="secondary", scale=1, size="sm")

    with gr.Row():
        b_stand = gr.Button("🧍 站立", size="sm")
        b_sit = gr.Button("🪑 坐下", size="sm")
        b_down = gr.Button("⬇️ 趴下", size="sm")
        b_fwd = gr.Button("⬆️ 前进 1s", size="sm")
        b_back = gr.Button("⬇️ 后退 1s", size="sm")
        b_left = gr.Button("⬅️ 左转 90°", size="sm")
        b_right = gr.Button("➡️ 右转 90°", size="sm")
        b_shot = gr.Button("📷 拍一张并诊断", size="sm", variant="primary")

    # ---------------- 报告下载区（缩小：一条紧凑行） ----------------
    report_file = gr.File(
        label="📄 报告下载（隐患排查建议报告.docx）",
        visible=True,
        scale=0,
    )

    # ---------------- 主聊天界面（与主对话同款） ----------------
    with gr.Column(elem_classes=["chat-panel"]):
        # 现场照片画廊：狗拍到的照片实时回传显示在这里
        gallery = gr.Gallery(
            label="📷 现场照片（实时回传）",
            columns=2,
            rows=1,
            height="auto",
            object_fit="cover",
            allow_preview=True,
        )
        chat = gr.ChatInterface(
            fn=_make_inspect_chat_fn(client, agent),
            multimodal=False,
            additional_outputs=[gallery, report_file],
            description=(
                "用自然语言指挥机器狗巡检：例如「向前走 2 秒，然后原地扫描一圈」；"
                "也可点击上方「区域巡检」一键走位扫描，或用快捷按钮单步微调。"
                "诊断结论为**法条原文逐字引用**，并支持逐字流式输出；"
                "狗拍到的照片会实时显示在上方「现场照片」画廊。"
            ),
            examples=[
                {"text": "向前走 2 秒，然后原地扫描 4 个方向各拍一张"},
                {"text": "左转 90 度，拍一张"},
                {"text": "坐下"},
            ],
            fill_height=True,
        )

    # ---------------- 事件绑定 ----------------
    refresh_btn.click(lambda: _status_md(client.snapshot()), outputs=status_md, queue=False)

    def _estop():
        ok, msg = client.estop()
        return f"### 🛑 急停已下发\n\n{'✅' if ok else '❌'} {msg}"

    estop_btn.click(_estop, outputs=status_md, queue=False)

    # 手动微调按钮：直接执行动作，结果写入状态区
    def _act(action):
        ok, msg = client.act(action)
        return f"{'✅' if ok else '❌'} `{action}` — {msg}"

    def _move(vx, vy, vyaw, duration):
        ok, msg = client.move(vx, vy, vyaw, duration)
        return f"{'✅' if ok else '❌'} {msg}"

    b_stand.click(lambda: _act("stand"), outputs=status_md, queue=False)
    b_sit.click(lambda: _act("sit"), outputs=status_md, queue=False)
    b_down.click(lambda: _act("standdown"), outputs=status_md, queue=False)
    b_fwd.click(lambda: _move(0.3, 0, 0, 1.0), outputs=status_md, queue=False)
    b_back.click(lambda: _move(-0.25, 0, 0, 1.0), outputs=status_md, queue=False)
    b_left.click(lambda: _move(0, 0, 0.9, 1.64), outputs=status_md, queue=False)
    b_right.click(lambda: _move(0, 0, -0.9, 1.64), outputs=status_md, queue=False)

    # 拍照并诊断按钮：走手动流式，结果写状态区 + 照片写画廊
    def _shot_diag():
        shot = client.capture("manual")
        if not shot.get("ok"):
            return f"❌ 抓拍失败：{shot.get('error')}", []
        path = shot.get("path_win") or shot.get("path_linux")
        gallery = [(path, shot.get("file", "现场照片"))]
        try:
            stream_fn = getattr(agent, "analyze_stream", None)
            if stream_fn is not None:
                result = None
                for ev in stream_fn(
                    image_path=path,
                    question="请分析这张现场照片是否存在安全隐患。",
                    force_hazard=True,
                ):
                    if ev.get("type") == "done":
                        result = ev.get("result")
                if result is None:
                    raise RuntimeError("流式诊断未返回最终结果")
            else:
                result = agent.analyze(image_path=path, question="请分析这张现场照片是否存在安全隐患。")
            structured = result.get("structured_output") or {}
            verdict = structured.get("隐患定性", "") or result.get("raw_answer", "")[:400]
            severity = classify_severity(verdict)
            item = {
                "file": shot.get("file"),
                "verdict": verdict,
                "severity": severity,
                "thought": structured.get("思维链", ""),
                "laws": structured.get("法律依据", []),
                "measures": structured.get("整改措施", []),
                "references": structured.get("参考案例", []),
                "retrieved_laws": result.get("retrieved_laws", []),
            }
            return _embed_photo_md(path, 1, shot.get("file", "现场照片")) + "\n\n" + _render_one_verdict(item, 1), gallery
        except Exception as exc:
            return f"❌ 诊断失败：{exc}", gallery

    b_shot.click(_shot_diag, outputs=[status_md, gallery])

    # 区域巡检按钮：跑完整流程，报告写回下载区 + 照片写画廊
    def _run_zone_quick(zone_key, shots):
        zone = ZONES.get(zone_key)
        if not zone:
            return "⚠️ 请先选择巡检区域", "", []
        ops = []
        for op in zone.route:
            if op["op"] == "scan":
                ops.append({"op": "scan", "shots": int(shots), "settle": op.get("settle", 1.3)})
            else:
                ops.append(dict(op))
        plan = validate_plan(ops)
        question = f"请分析这张机器狗巡检拍摄的现场照片（区域：{zone.name}），判断是否存在安全隐患并进行隐患定性。"
        report_path = None
        final_report = ""
        gallery = []
        for snap in execute_plan_iter(client, plan, question=question, agent=agent, diagnose=True, zone_name=zone.key):
            final_report = snap.get("report", "")
            report_path = snap.get("report_path", report_path)
            gallery = snap.get("gallery") or gallery
        return final_report, report_path, gallery

    zone_btn.click(
        _run_zone_quick,
        inputs=[zone_dd, shots_sl],
        outputs=[status_md, report_file, gallery],
    )

    # 航点导航按钮：加载航点配置 → 闭环导航 → 每个航点拍照诊断 → 报告
    def _run_waypoint_quick(nav_name):
        if not nav_name:
            return "⚠️ 请先在 `configs/` 下创建航点配置文件（如 nav_waypoints.yaml）", "", []
        cfg_path = project_root / "configs" / nav_name
        if not cfg_path.exists():
            return f"⚠️ 航点配置文件不存在：{cfg_path}", "", []
        try:
            config = NavConfig.from_yaml(cfg_path)
        except Exception as exc:
            return f"⚠️ 航点配置解析失败：{exc}", "", []
        if not config.waypoints:
            return "⚠️ 航点配置里没有 waypoints，请检查文件。", "", []

        question = "请分析这张机器狗巡检拍摄的现场照片，判断是否存在安全隐患并进行隐患定性。"
        report_path = None
        final_report = ""
        gallery = []
        for snap in navigate_waypoints_iter(
            client, config, question=question, agent=agent, diagnose=True, zone_name="航点导航"
        ):
            final_report = snap.get("report", "")
            report_path = snap.get("report_path", report_path)
            gallery = snap.get("gallery") or gallery
        return final_report, report_path, gallery

    nav_btn.click(
        _run_waypoint_quick,
        inputs=[nav_dd],
        outputs=[status_md, report_file, gallery],
    )
