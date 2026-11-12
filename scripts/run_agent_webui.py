# -*- coding: utf-8 -*-
"""Safety_Agent WebUI —— 高级视觉版
暗色科技风 + 动态渐变头图 + 玻璃拟态卡片 + 结构化报告渲染。
Gradio 6 注意：theme 与 css 必须传给 launch()，不能传给 Blocks()。
"""
import gradio as gr
import os
import re
import sys
from pathlib import Path
import requests

# 将 src 目录加入环境变量以导入你的 agent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from agent.workflow import build_agent_from_env
except Exception as e:
    print(f"无法加载 Agent 工作流，错误详情: {e}")
    sys.exit(1)

# 初始化智能体
print("正在初始化智能体，这可能需要一些时间（尤其是加载模型或连接数据库时）...")
agent = build_agent_from_env()
print("智能体初始化完成！")

# 机器狗巡检面板（可选：导入失败不影响原有对话/知识库/语音功能）
try:
    from robot.ui import build_robot_tab

    ROBOT_PANEL_OK = True
    ROBOT_PANEL_ERR = ""
except Exception as e:  # pragma: no cover
    ROBOT_PANEL_OK = False
    ROBOT_PANEL_ERR = str(e)
    print(f"机器狗面板未启用（{e}）")

try:
    from agent.streaming import format_partial_chain
    from voice.config import VoiceConfig
    from voice.pipeline import VoicePipeline
    VOICE_ENABLED = True
except Exception as exc:  # pragma: no cover - 语音依赖缺失时降级为纯文本模式
    print(f"[WARN] 语音模块未启用：{exc}")
    VOICE_ENABLED = False


# ---------------------------------------------------------------------------
# 业务逻辑
# ---------------------------------------------------------------------------

def _escape_md(text: str) -> str:
    return str(text or "").replace("|", "\\|")


def _md(text) -> str:
    """转成安全的 markdown 文本（换行保留）。"""
    s = str(text or "").strip()
    return s.replace("\r\n", "\n")


# 思维链各步骤的图标映射（未命中则用默认）
_STEP_ICONS = {
    "视觉分析": "👁️",
    "图像分析": "👁️",
    "法规匹配": "📖",
    "法规检索": "📖",
    "隐患判定": "🚨",
    "隐患定性": "🚨",
    "风险研判": "🧭",
    "整改建议": "🛠️",
    "整改措施": "🛠️",
    "综合结论": "✅",
    "案例比对": "🗂️",
}


def _render_thought_steps(steps) -> str:
    """把思维链（[{步骤, 内容}, ...]）渲染成步骤时间线卡片。"""
    if isinstance(steps, dict):
        steps = [steps]
    if not isinstance(steps, list):
        return _md(steps) + "\n\n"

    html = ["<div class='sa-steps'>"]
    for i, s in enumerate(steps, 1):
        if isinstance(s, dict):
            name = str(s.get("步骤") or s.get("step") or f"步骤 {i}")
            content = _md(s.get("内容") or s.get("content") or "")
        else:
            name = f"步骤 {i}"
            content = _md(s)
        icon = _STEP_ICONS.get(name, "🔹")
        body = content.replace("\n", "<br>") if content else "—"
        html.append(
            "<div class='sa-step'>"
            f"<div class='sa-step-rail'><span class='sa-step-num'>{i}</span></div>"
            "<div class='sa-step-card'>"
            f"<div class='sa-step-name'>{icon} {name}</div>"
            f"<div class='sa-step-body'>{body}</div>"
            "</div>"
            "</div>"
        )
    html.append("</div>")
    return "\n".join(html) + "\n\n"


def _render_law_item(law) -> str:
    """把一条法律依据渲染成卡片式 Markdown（含原文引用块）。兼容多种键名。"""
    if isinstance(law, dict):
        no = law.get("条款号") or law.get("法规名称") or law.get("条文") or ""
        content = law.get("条款内容") or law.get("原文") or law.get("内容") or ""
        apply_note = law.get("适用说明") or law.get("说明") or ""
        block = f"**📜 {no}**\n\n" if no else ""
        if content:
            block += f"> {_md(content)}\n\n"
        if apply_note:
            block += f"*📌 {_md(apply_note)}*\n"
        return block or "-（空条目）\n"
    return f"- {_md(law)}\n"


def _render_measure(m) -> str:
    if isinstance(m, dict):
        mtype = m.get("措施类型") or m.get("类型") or "措施"
        content = _md(m.get("措施内容") or m.get("内容") or "")
        owner = m.get("责任主体") or ""
        deadline = m.get("时限要求") or m.get("时限") or ""
        block = f"**🔧 [{mtype}]** {content}\n\n" if content else f"**🔧 [{mtype}]**\n\n"
        meta = []
        if owner:
            meta.append(f"👤 责任主体：**{owner}**")
        if deadline:
            meta.append(f"⏱️ 时限：**{deadline}**")
        if meta:
            block += " ｜ ".join(meta) + "\n"
        return block
    return f"- {_md(m)}\n"


def _render_report(structured: dict) -> str:
    """把结构化结果渲染成报告（兼容「字段还在生成中」的部分值）。"""
    if not isinstance(structured, dict) or not structured:
        return ""

    reply = ""

    # 思维链（步骤时间线，默认折叠）
    thought = structured.get("思维链")
    if thought:
        count = len(thought) if isinstance(thought, list) else 1
        reply += (
            "<details class='sa-details sa-thought'><summary>🧠 <b>分析推理过程</b>"
            f"（{count} 个步骤 · 点击展开/折叠）</summary>\n\n"
            + _render_thought_steps(thought)
            + "</details>\n\n"
        )

    # 隐患定性
    verdict = structured.get("隐患定性") or ""
    if verdict:
        reply += f"### 🚨 隐患定性\n\n{_md(verdict)}\n\n"

    # 法律依据（逐条卡片 + 原文引用）
    laws_output = structured.get("法律依据") or []
    if laws_output:
        reply += "### 📖 法律依据（逐条原文引用）\n\n"
        if isinstance(laws_output, list):
            for i, law in enumerate(laws_output, 1):
                reply += f"**{i}.** " + _render_law_item(law) + "\n"
        else:
            reply += _md(laws_output) + "\n"
        reply += "\n"

    # 整改措施
    measures = structured.get("整改措施") or []
    if measures:
        reply += "### 🛠️ 整改措施（可执行 · 可复核）\n\n"
        if isinstance(measures, list):
            for m in measures:
                reply += "- " + _render_measure(m)
        else:
            reply += _md(measures) + "\n"
        reply += "\n"

    # 参考案例
    refs = structured.get("参考案例") or []
    if refs:
        reply += "### 🗂️ 参考案例\n\n"
        items = refs if isinstance(refs, list) else [refs]
        for ref in items:
            if isinstance(ref, dict):
                title = ref.get("案例名称") or ref.get("名称") or ref.get("case_id") or "案例"
                desc = _md(ref.get("描述") or ref.get("内容") or ref.get("issue_text") or "")
                reply += f"- **{title}**" + (f"：{desc}" if desc else "") + "\n"
            else:
                reply += f"- {_md(ref)}\n"
        reply += "\n"

    return reply


def _collect_spoken_text(structured: dict) -> str:
    """从结构化结果里拼一段适合朗读的文本（跳过思维链与参考案例）。"""
    chunks: list[str] = []

    verdict = structured.get("隐患定性") or ""
    if verdict:
        chunks.append(f"隐患定性：{_md(verdict)}")

    laws = structured.get("法律依据") or []
    if isinstance(laws, list):
        for law in laws:
            if isinstance(law, dict):
                no = law.get("条款号") or ""
                content = law.get("条款内容") or ""
                line = f"{no} {content}".strip()
                if line:
                    chunks.append(line)
            elif law:
                chunks.append(str(law))
    elif laws:
        chunks.append(str(laws))

    measures = structured.get("整改措施") or []
    if isinstance(measures, list):
        for m in measures:
            if isinstance(m, dict):
                content = m.get("措施内容") or ""
                owner = m.get("责任主体") or ""
                deadline = m.get("时限要求") or ""
                line = content
                if owner:
                    line += f"，责任主体：{owner}"
                if deadline:
                    line += f"，时限：{deadline}"
                if line:
                    chunks.append(line)
            elif m:
                chunks.append(str(m))
    elif measures:
        chunks.append(str(measures))

    return "\n".join(chunk for chunk in chunks if str(chunk).strip())


def _rag_sources(cases: list, laws: list) -> str:
    """渲染 RAG 检索来源（折叠面板，避免喧宾夺主）。"""
    if not cases and not laws:
        return ""

    out = (
        "<details class='sa-details sa-rag'><summary>🔍 <b>检索增强来源 (RAG Sources)</b>（点击展开）</summary>\n\n"
    )
    if cases:
        out += "**📷 相似案卷：**\n"
        for i, case in enumerate(cases, 1):
            if isinstance(case, dict):
                src = case.get("source_file", "未知源")
                snippet = _escape_md(
                    (case.get("issue_text") or case.get("legal_basis") or "").strip().replace("\n", " ")
                )
                # 文本检索命中的案例可能没有配图，优先展示案卷来源与隐患描述
                out += (
                    f"{i}. `{src}`（得分 `{case.get('score', 0):.2f}`）\n"
                    f"   > {snippet[:140]}{'…' if len(snippet) > 140 else ''}\n"
                )
            else:
                out += f"{i}. {case}\n"
        out += "\n"
    if laws:
        out += "**📜 命中法条：**\n"
        for i, law in enumerate(laws, 1):
            if isinstance(law, dict):
                src = law.get("source_file", "未知源")
                content = law.get("legal_basis", "")
                score = law.get("score", 0)
                out += f"{i}. `{src}`（距离 {score:.2f}）\n   > {content[:160]}{'…' if len(content) > 160 else ''}\n\n"
            else:
                out += f"{i}. {law}\n"
    out += "</details>\n"
    return out


def _stream_reply(message, history=None, sink: dict | None = None):
    """流式应答生成器：模型边生成，界面边刷新。

    产出约定：每一次 yield 都是「当前为止的完整文本」，由调用方负责刷新界面。

    sink: 可选字典，用于把最终的结构化结果带出去（供语音播报复用），
          键为 structured / raw_answer / cases / laws。
    """
    text_query = message.get("text", "请分析图片存在的安全隐患。")
    files = message.get("files", [])

    has_image = bool(files)
    intent = agent.detect_intent(question=text_query, has_image=has_image)

    # 无图 + 判定为隐患分析：不硬拦（知识库里有大量文字案例可依据），
    # 只加一行说明，交给智能体基于案例库/法规库作答。
    no_image_notice = ""
    if not has_image and intent == "hazard_analysis":
        no_image_notice = (
            "> ℹ️ 未上传现场照片，以下依据**隐患案例库与法规条文**、结合文字描述分析；"
            "如需对照现场实况，请补传照片。\n\n"
        )

    image_path = files[0] if files else None

    structured: dict = {}
    thought_stream = ""
    raw_answer = ""
    result_payload: dict = {}
    error_message = ""

    try:
        for event in agent.analyze_stream(image_path=image_path, question=text_query):
            etype = event.get("type")

            if etype == "status":
                if event.get("stage") == "intent":
                    structured = {}
                    thought_stream = ""
                continue

            if etype == "field_start":
                thought_stream = ""
                continue

            if etype == "field_text":
                key = event.get("key")
                text = event.get("text") or ""
                if key == "思维链":
                    # 思考过程逐字上屏
                    thought_stream = format_partial_chain(text)
                    yield _compose_progress(thought_stream, structured)
                continue

            if etype == "field_done":
                key = event.get("key")
                value = event.get("value")
                if key:
                    structured[key] = value
                if key == "思维链":
                    thought_stream = ""
                yield _compose_progress(thought_stream, structured)
                continue

            if etype == "done":
                result_payload = event.get("result", {}) or {}
                raw_answer = result_payload.get("raw_answer", "")
                if result_payload.get("structured_output"):
                    structured = result_payload["structured_output"]
                break

            if etype == "warn":
                error_message = event.get("message", "")
    except Exception as exc:  # noqa: BLE001
        error_message = f"推理过程异常：{exc}"

    # 最终渲染
    if structured:
        reply = _render_report(structured)
    else:
        reply = _md(_strip_intent_prefix(raw_answer)) + "\n\n"

    if error_message:
        reply += f"> ⚠️ {error_message}\n"

    cases = result_payload.get("similar_cases", []) or []
    laws = result_payload.get("retrieved_laws", []) or []

    # 知识库命中情况（让人一眼确认到底检索到了什么）
    kb_line = f"> 🔍 知识库检索：法规库 **{len(laws)}** 条｜隐患案例库 **{len(cases)}** 条\n\n"
    reply = no_image_notice + kb_line + reply
    reply += _rag_sources(cases, laws)

    if sink is not None:
        sink.update(
            {
                "structured": structured or {},
                "raw_answer": raw_answer,
                "cases": cases,
                "laws": laws,
            }
        )

    yield reply


def _compose_progress(thought_stream: str, structured: dict) -> str:
    """把「思考中的片段 + 已完成的字段」拼成中间态文本。"""
    parts: list[str] = []

    if thought_stream:
        parts.append(
            "<details class='sa-details sa-thought' open><summary>🧠 <b>正在推理…</b></summary>\n\n"
            + _md(thought_stream)
            + "\n\n</details>\n\n"
        )

    finished = {k: v for k, v in (structured or {}).items() if k != "思维链"}
    if finished:
        parts.append(_render_report(finished))

    if not parts:
        parts.append("🧠 **已收到问题，正在分析…**")

    return "".join(parts)


def _strip_intent_prefix(text: str) -> str:
    """去掉模型输出开头的意图回显（如 legal_only / hazard_analysis）。"""
    return re.sub(r"^\s*(greeting|legal_only|hazard_analysis)\s*[:：，,]?\s*", "", text or "", flags=re.IGNORECASE)


def _build_pipeline():
    """构造语音链路；模块不可用时返回 None。"""
    if not VOICE_ENABLED:
        return None
    from voice.config import VoiceConfig
    from voice.pipeline import VoicePipeline

    cfg = VoiceConfig.from_env()
    return VoicePipeline(agent=agent, config=cfg), cfg


def _asr_to_input(audio_path, current_text):
    """只做语音识别：把结果填进输入框，不触发智能体推理。"""
    if not VOICE_ENABLED:
        return current_text, "⚠️ 语音模块未启用，请先安装依赖：`pip install dashscope sounddevice soundfile`"
    if not audio_path:
        return current_text, "⚠️ 请先用 🎙️ 麦克风录一段语音，再点识别"

    built = _build_pipeline()
    if built is None:
        return current_text, "⚠️ 语音模块未启用"
    pipeline, _cfg = built
    try:
        question = pipeline.understand(Path(audio_path).read_bytes())
    except Exception as exc:  # noqa: BLE001
        return current_text, f"⚠️ 语音识别失败：{exc}"

    question = _strip_intent_prefix((question or "").strip())
    if not question:
        return current_text, "⚠️ 没听清，请重新录制"

    return question, "✅ 识别完成 — 可直接在输入框里修改，改好后点「🚀 发送」"


def _synthesize(structured: dict, raw_answer: str):
    """把回答合成成播报音频，返回 (音频路径, 错误信息)。"""
    spoken = _collect_spoken_text(structured) if structured else ""
    if not spoken.strip():
        spoken = _md(_strip_intent_prefix(raw_answer))
    if not spoken.strip():
        return None, ""
    try:
        from voice.audio_io import save_audio

        built = _build_pipeline()
        if built is None:
            return None, "语音模块未启用"
        pipeline, cfg = built
        return str(save_audio(pipeline.tts.synthesize(spoken), cfg.tts_format)), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _chat_submit(text, image_path, history, want_tts):
    """对话主流程：智能体边思考边生成；结束后按勾选合成语音播报。

    yield: (chatbot 历史, 输入框清空值, 播报音频路径, 状态文本)
    """
    history = list(history or [])
    question = (text or "").strip()
    files = [image_path] if image_path else []

    if not question and not files:
        yield history, "", None, "⚠️ 请输入问题，或先用麦克风识别一段语音"
        return

    default_question = "请分析图片存在的安全隐患。"
    user_display = question or default_question
    if files:
        user_display = f"{question or default_question}\n\n📷 现场照片：`{Path(image_path).name}`"
    history = history + [{"role": "user", "content": user_display}]
    yield history, "", None, ""

    history = history + [{"role": "assistant", "content": "🧠 **已收到问题，正在分析…**"}]
    yield history, "", None, ""

    message = {"text": question or default_question, "files": files}
    sink: dict = {}
    final = ""
    for chunk in _stream_reply(message, history, sink=sink):
        final = chunk
        history[-1]["content"] = chunk
        yield history, "", None, ""

    if not want_tts:
        yield history, "", None, ""
        return

    # 播报：先提示正在合成，避免界面像是卡住
    history[-1]["content"] = final + "\n\n> 🔊 正在合成语音…"
    yield history, "", None, ""

    if not VOICE_ENABLED:
        history[-1]["content"] = final + "\n\n> ⚠️ 语音模块未启用，无法播报"
        yield history, "", None, "⚠️ 语音模块未启用"
        return

    audio_path_out, err = _synthesize(sink.get("structured") or {}, sink.get("raw_answer", ""))
    if err:
        history[-1]["content"] = final + f"\n\n> ⚠️ 语音合成失败：{err}"
        yield history, "", None, f"⚠️ 语音合成失败：{err}"
    elif audio_path_out is None:
        # 没有可朗读的内容（例如提示性回答），直接收尾，不假装播报过
        yield history, "", None, "（本条无可朗读内容，已跳过播报）"
    else:
        yield history, "", audio_path_out, "🔊 播报已生成"


def _clear_chat():
    return [], "", None, ""


def upload_library_file(file, lib_type):
    """通过 Gradio 界面上传文件到知识库"""
    if file is None:
        return "请先选择需要上传的文件。"
    file_path = file.name
    try:
        url = "http://127.0.0.1:8001/upload"
        files = {'file': open(file_path, 'rb')}
        data = {'lib_type': "laws" if lib_type == "法律法规库" else "hazards"}
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            return "✅ " + response.json().get("message", "上传并处理成功！")
        else:
            return f"❌ 上传失败: {response.text}"
    except Exception as e:
        return f"⚠️ 发生错误: {str(e)}"


API_BASE = "http://127.0.0.1:8001"


def kb_fetch(lib_type):
    try:
        resp = requests.get(f"{API_BASE}/kb", params={"lib_type": lib_type})
        if resp.status_code != 200:
            return [], f"查询失败: {resp.text}"
        cases = resp.json().get("cases", [])
        simplified: list[list[str]] = []
        for c in cases:
            simplified.append([
                c.get("id", ""),
                c.get("source_file", ""),
                (c.get("legal_basis", c.get("issue_text", "")) or "")[:200],
            ])
        return simplified, f"✅ 查询成功，共 {len(simplified)} 条记录"
    except Exception as e:
        return [], f"⚠️ 请求出错: {e}"


def kb_create(lib_type, content, source_file):
    if not content:
        return [], "❌ 内容不能为空"
    try:
        resp = requests.post(f"{API_BASE}/kb/create", json={
            "lib_type": lib_type,
            "content": content,
            "source_file": source_file or "manual",
        })
        if resp.status_code != 200:
            return [], f"❌ 新增失败: {resp.text}"
        return kb_fetch(lib_type)
    except Exception as e:
        return [], f"⚠️ 请求出错: {e}"


def kb_update(lib_type, case_id, content, source_file):
    if not case_id or not content:
        return [], "❌ ID 和内容均不能为空"
    try:
        resp = requests.post(f"{API_BASE}/kb/update", json={
            "lib_type": lib_type,
            "case_id": case_id,
            "content": content,
            "source_file": source_file or None,
        })
        if resp.status_code != 200:
            return [], f"❌ 更新失败: {resp.text}"
        return kb_fetch(lib_type)
    except Exception as e:
        return [], f"⚠️ 请求出错: {e}"


def kb_delete(lib_type, case_id):
    if not case_id:
        return [], "❌ ID 不能为空"
    try:
        resp = requests.post(f"{API_BASE}/kb/delete", json={"case_id": case_id})
        if resp.status_code != 200:
            return [], f"❌ 删除失败: {resp.text}"
        return kb_fetch(lib_type)
    except Exception as e:
        return [], f"⚠️ 请求出错: {e}"


# ---------------------------------------------------------------------------
# 界面：暗色科技风 · 玻璃拟态 · 动态渐变
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div class="sa-hero">
  <div class="sa-hero-glow"></div>
  <div class="sa-hero-inner">
    <div class="sa-logo">
      <div class="sa-logo-icon">⛑️</div>
      <div>
        <h1>安全生产管理智能体</h1>
        <p>Safety Production Agent · 多模态隐患研判 &nbsp;·&nbsp; 法规精准溯源</p>
      </div>
    </div>
    <div class="sa-badges">
      <span class="sa-badge sa-badge-red">🔍 隐患识别</span>
      <span class="sa-badge sa-badge-blue">📖 法条精引</span>
      <span class="sa-badge sa-badge-cyan">🧠 Qwen-VL 多模态</span>
      <span class="sa-badge sa-badge-purple">📚 RAG 溯源</span>
    </div>
  </div>
  <div class="sa-hero-line"></div>
</div>
"""

FOOTER_HTML = """
<div class="sa-footer">
  <span>⚙️ 云端推理 · 阿里云百炼</span><i>·</i>
  <span>🗂️ 本地知识库 · ChromaDB</span><i>·</i>
  <span>⚖️ 法条原文逐字引用，拒绝模糊表述</span>
</div>
"""

CHAT_CSS = """
/* ============ 全局 ============ */
:root {
  --page-padding: 14px;
  --sa-bg: #0b1220;
  --sa-card: rgba(255,255,255,0.045);
  --sa-border: rgba(148, 190, 255, 0.16);
  --sa-cyan: #38e1ff;
  --sa-blue: #4f7cff;
  --sa-purple: #a78bfa;
  --sa-red: #ff6b6b;
  --sa-amber: #fbbf24;
}
body {
  margin: 0;
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(79,124,255,0.16), transparent 60%),
    radial-gradient(900px 500px at -10% 25%, rgba(56,225,255,0.10), transparent 55%),
    radial-gradient(700px 500px at 110% 80%, rgba(167,139,250,0.10), transparent 55%),
    var(--sa-bg) !important;
}
.gradio-container {
  max-width: 100% !important;
  padding: var(--page-padding) !important;
  min-height: 100vh;
}

/* ============ Hero 头图 ============ */
.sa-hero {
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  padding: 26px 30px 22px;
  margin-bottom: 14px;
  background:
    linear-gradient(120deg, rgba(15, 30, 60, 0.92) 0%, rgba(20, 45, 80, 0.88) 45%, rgba(35, 25, 70, 0.88) 100%);
  border: 1px solid var(--sa-border);
  box-shadow: 0 18px 50px rgba(3, 10, 30, 0.55), inset 0 1px 0 rgba(255,255,255,0.06);
}
.sa-hero-glow {
  position: absolute; inset: -40%;
  background: conic-gradient(from 0deg,
    transparent 0%, rgba(56,225,255,0.14) 12%, transparent 25%,
    transparent 50%, rgba(79,124,255,0.14) 62%, transparent 75%,
    transparent 100%);
  animation: sa-rotate 14s linear infinite;
  pointer-events: none;
}
@keyframes sa-rotate { to { transform: rotate(360deg); } }
.sa-hero-inner { position: relative; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; }
.sa-logo { display: flex; align-items: center; gap: 16px; }
.sa-logo-icon {
  font-size: 40px; width: 66px; height: 66px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(56,225,255,0.22), rgba(79,124,255,0.22));
  border: 1px solid rgba(56,225,255,0.35);
  box-shadow: 0 0 26px rgba(56,225,255,0.25);
}
.sa-logo h1 {
  margin: 0 0 4px 0; font-size: 26px; font-weight: 800; letter-spacing: 1px;
  color: #eaf6ff;
  text-shadow: 0 0 18px rgba(56,225,255,0.55), 0 0 4px rgba(56,225,255,0.35);
}
.sa-logo p { margin: 0; color: #9fb3d9; font-size: 13px; letter-spacing: 0.5px; }
.sa-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.sa-badge {
  font-size: 12.5px; padding: 5px 12px; border-radius: 999px;
  backdrop-filter: blur(6px);
  border: 1px solid var(--sa-border);
  color: #cfe3ff;
  background: rgba(255,255,255,0.05);
  transition: transform .18s ease, box-shadow .18s ease;
}
.sa-badge:hover { transform: translateY(-2px); }
.sa-badge-red   { border-color: rgba(255,107,107,.45); color: #ffc9c9; box-shadow: 0 0 14px rgba(255,107,107,.12); }
.sa-badge-blue  { border-color: rgba(79,124,255,.5);  color: #c7d8ff; box-shadow: 0 0 14px rgba(79,124,255,.14); }
.sa-badge-cyan  { border-color: rgba(56,225,255,.5);  color: #bff2ff; box-shadow: 0 0 14px rgba(56,225,255,.14); }
.sa-badge-purple{ border-color: rgba(167,139,250,.5); color: #e2d8ff; box-shadow: 0 0 14px rgba(167,139,250,.14); }
.sa-hero-line {
  position: relative; height: 2px; margin-top: 18px; border-radius: 2px;
  background: linear-gradient(90deg, transparent, var(--sa-cyan), var(--sa-blue), var(--sa-purple), transparent);
  background-size: 200% 100%;
  animation: sa-flow 6s linear infinite;
}
@keyframes sa-flow { to { background-position: 200% 0; } }

/* ============ 页脚 ============ */
.sa-footer {
  display: flex; justify-content: center; align-items: center; gap: 10px;
  margin-top: 14px; padding: 10px 0 2px;
  color: #7d90b5; font-size: 12.5px;
  border-top: 1px solid rgba(148,190,255,0.10);
}
.sa-footer i { color: #3d4f75; font-style: normal; }

/* ============ Tab 导航 ============ */
.gradio-container .tab-container button,
.gradio-container [class*="tab-nav"] button {
  border-radius: 12px 12px 0 0 !important;
  font-weight: 600;
}
.gradio-container .tab-container button.selected,
.gradio-container [class*="tab-nav"] button.selected {
  background: linear-gradient(180deg, rgba(56,225,255,0.16), rgba(79,124,255,0.10)) !important;
  color: #eaf6ff !important;
  text-shadow: 0 0 12px rgba(56,225,255,0.45);
}

/* ============ 区块 / 面板（玻璃拟态）============ */
.gradio-container .form,
.gradio-container .block,
.gradio-container .panel {
  background: var(--sa-card) !important;
  border: 1px solid var(--sa-border) !important;
  border-radius: 14px !important;
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 28px rgba(2, 8, 24, 0.35);
}

/* ============ 聊天气泡 ============ */
.chat-panel { display: flex; flex-direction: column; gap: 12px; min-height: calc(100vh - 250px); }
.chat-panel .chatbot { flex: 1 1 auto; min-height: 62vh; }
.chat-panel .message-row { width: 100% !important; }
.chat-panel textarea { min-height: 130px; }

/* 用户气泡：蓝紫渐变 */
.message-row.user-row .bubble {
  background: linear-gradient(135deg, rgba(79,124,255,0.85), rgba(56,225,255,0.65)) !important;
  border: 1px solid rgba(140,180,255,0.5) !important;
  color: #ffffff !important;
  border-radius: 16px 16px 4px 16px !important;
  box-shadow: 0 6px 22px rgba(79,124,255,0.30);
}
/* 助手气泡：深色玻璃 */
.message-row.assistant-row .bubble,
.message-row.bot-row .bubble {
  background: linear-gradient(150deg, rgba(18, 32, 58, 0.96), rgba(13, 22, 42, 0.92)) !important;
  border: 1px solid rgba(56,225,255,0.22) !important;
  color: #dbe8ff !important;
  border-radius: 16px 16px 16px 4px !important;
  box-shadow: 0 10px 30px rgba(2, 10, 30, 0.45);
}

/* ============ 聊天内 Markdown 排版 ============ */
.assistant-row .message-content h3 {
  padding: 6px 12px;
  border-left: 3px solid var(--sa-cyan);
  background: linear-gradient(90deg, rgba(56,225,255,0.10), transparent 75%);
  border-radius: 4px;
  color: #aee9ff !important;
}
.assistant-row .message-content blockquote {
  background: rgba(56,225,255,0.06);
  border-left: 3px solid var(--sa-cyan);
  padding: 8px 14px;
  border-radius: 0 8px 8px 0;
  color: #b9d8f5 !important;
}
.assistant-row .message-content code {
  background: rgba(167,139,250,0.16);
  color: #d9c9ff;
  border: 1px solid rgba(167,139,250,0.25);
  border-radius: 5px;
  padding: 1px 6px;
}
.assistant-row .message-content hr {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(148,190,255,0.4), transparent);
}
details.sa-details {
  border: 1px solid rgba(167,139,250,0.3);
  border-radius: 10px;
  padding: 10px 14px;
  background: rgba(167,139,250,0.06);
  margin: 8px 0;
}
details.sa-details summary {
  cursor: pointer;
  color: #cfc2ff;
  font-weight: 600;
  user-select: none;
}
details.sa-rag { border-color: rgba(148,190,255,0.25); background: rgba(148,190,255,0.05); }
details.sa-rag summary { color: #a9c4ea; }

/* ============ 聊天内嵌照片（机器狗回传） ============ */
.assistant-row .message-content img,
.bot-row .message-content img {
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  max-height: 380px;
  object-fit: cover;
  border-radius: 12px;
  margin: 6px 0 12px;
  border: 1px solid rgba(56,225,255,0.28);
  box-shadow: 0 8px 24px rgba(2,10,30,0.45);
  cursor: zoom-in;
}

/* ============ 思维链步骤时间线 ============ */
details.sa-thought { border-color: rgba(79,124,255,0.35); background: rgba(79,124,255,0.07); }
details.sa-thought summary { color: #a9c4ff; }
.sa-steps { display: flex; flex-direction: column; gap: 10px; margin: 10px 2px 4px; }
.sa-step { display: flex; gap: 10px; align-items: stretch; }
.sa-step-rail { display: flex; flex-direction: column; align-items: center; }
.sa-step-num {
  width: 26px; height: 26px; flex: 0 0 auto;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%;
  font-size: 13px; font-weight: 800; color: #061224;
  background: linear-gradient(135deg, var(--sa-cyan), var(--sa-blue));
  box-shadow: 0 0 12px rgba(56,225,255,0.4);
}
.sa-step:not(:last-child) .sa-step-rail::after {
  content: ""; flex: 1 1 auto; width: 2px; margin-top: 4px;
  background: linear-gradient(180deg, rgba(56,225,255,0.5), rgba(79,124,255,0.15));
  border-radius: 2px;
}
.sa-step-card {
  flex: 1 1 auto; min-width: 0;
  border: 1px solid rgba(148,190,255,0.18);
  border-radius: 10px;
  background: linear-gradient(150deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  padding: 8px 12px 10px;
}
.sa-step-name {
  font-weight: 700; font-size: 13.5px; color: #9fe8ff;
  letter-spacing: 0.5px; margin-bottom: 4px;
}
.sa-step-body { font-size: 13.5px; line-height: 1.75; color: #cfe0f5; word-break: break-word; }

/* ============ 按钮发光 ============ */
.gradio-container button.primary,
.gradio-container .primary {
  background: linear-gradient(135deg, var(--sa-blue), var(--sa-cyan)) !important;
  border: none !important;
  color: #061224 !important;
  font-weight: 700;
  box-shadow: 0 4px 18px rgba(56,225,255,0.35);
  transition: all .2s ease;
}
.gradio-container button.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 26px rgba(56,225,255,0.55);
}
.gradio-container button.secondary:hover {
  border-color: var(--sa-cyan) !important;
  color: var(--sa-cyan) !important;
}

/* ============ 滚动条 ============ */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(79,124,255,0.5), rgba(56,225,255,0.45));
  border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(56,225,255,0.7); }

/* ============ 数据表格 ============ */
.gradio-container table { border-collapse: collapse; }
.gradio-container tbody tr:hover td { background: rgba(56,225,255,0.05); }
"""

with gr.Blocks(title="安全生产管理智能体", fill_height=True) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tab("💬 智能对话与检测"):
        gr.Markdown(
            "上传现场照片（或截图）+ 输入问题，智能体进行多模态隐患分析；也可直接咨询法规条文。"
            "法条回答均为**原文逐字引用**，标注条款号与数值标准。\n\n"
            "**🎙️ 语音输入**：① 麦克风录制 → ② 点「识别语音」（**只识别，不会开始思考**）"
            " → ③ 文字自动填入输入框，改好 → ④ 点「发送」，智能体才开始推理 → ⑤ 勾选「语音播报」后自动朗读回答。"
        )

        # ---------------- 界面组件 ----------------
        with gr.Column(elem_classes=["chat-panel"]):
            chatbot = gr.Chatbot(
                show_label=False,
                height="56vh",
                autoscroll=True,
                render_markdown=True,
                placeholder="👋 上传现场照片或直接提问；也可以先用麦克风说一句话。",
            )

            with gr.Row():
                chat_image = gr.Image(
                    label="📷 现场照片（可选）",
                    sources=["upload", "clipboard"],
                    type="filepath",
                    height=190,
                    scale=1,
                )
                chat_input = gr.Textbox(
                    label="💬 问题（语音识别结果会填到这里，可修改后再发送）",
                    lines=4,
                    scale=4,
                    placeholder="例如：有限空间作业前气体检测的氧含量合格标准是多少？",
                )

            with gr.Row():
                mic_audio = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎙️ 麦克风",
                    scale=2,
                )
                with gr.Column(scale=2, min_width=300):
                    with gr.Row():
                        asr_btn = gr.Button("🎤 识别语音", scale=1)
                        send_btn = gr.Button("🚀 发送", variant="primary", scale=1)
                        clear_btn = gr.Button("🗑️ 清空", scale=1)
                    voice_status = gr.Markdown("")

            with gr.Row():
                auto_speak = gr.Checkbox(value=True, label="🔊 回答后语音播报", scale=1)
                voice_out = gr.Audio(label="🔉 语音播报", autoplay=True, scale=3)

        gr.Examples(
            examples=[
                "有限空间作业前气体检测的氧含量合格标准是多少？检测频率有什么要求？",
                "未对从业人员进行安全生产教育和培训，依据哪部法律哪一条处罚？罚款金额是多少？",
                "你好，你能做什么？",
            ],
            inputs=chat_input,
        )

        asr_btn.click(
            _asr_to_input,
            inputs=[mic_audio, chat_input],
            outputs=[chat_input, voice_status],
        )

        _submit_cfg = dict(
            fn=_chat_submit,
            inputs=[chat_input, chat_image, chatbot, auto_speak],
            outputs=[chatbot, chat_input, voice_out, voice_status],
        )
        send_btn.click(**_submit_cfg)
        chat_input.submit(**_submit_cfg)
        clear_btn.click(
            _clear_chat,
            outputs=[chatbot, chat_input, voice_out, voice_status],
        )

    with gr.Tab("📥 知识库录入"):
        gr.Markdown("### 上传新资料到系统数据库\n支持 PDF / Word / TXT，入库后自动完成解析、分块与向量化。")
        with gr.Row():
            file_input = gr.File(
                label="选择文件 (建议 PDF/Word/TXT)",
                file_types=None,
            )
            lib_type = gr.Radio(["法律法规库", "安全隐患库"], label="目标知识库", value="法律法规库")

        upload_button = gr.Button("🚀 上传并开始分析/入库")
        output_txt = gr.Textbox(label="操作状态")

        upload_button.click(
            fn=upload_library_file,
            inputs=[file_input, lib_type],
            outputs=output_txt
        )

    with gr.Tab("🗄️ 知识库管理"):
        gr.Markdown("### 查看 / 增删改 知识库条目")
        lib_select = gr.Dropdown(["laws", "hazards"], value="laws", label="库类型 (laws=法律法规, hazards=安全隐患)")
        kb_table = gr.Dataframe(headers=["id", "source_file", "legal_basis"], interactive=False)
        status_box = gr.Textbox(label="操作状态", interactive=False)

        with gr.Row():
            refresh_btn = gr.Button("🔄 刷新列表")
            del_id = gr.Textbox(label="删除/更新用 ID", lines=1)
            del_btn = gr.Button("🗑️ 删除")
        with gr.Row():
            new_content = gr.Textbox(label="内容 (法律条文或隐患描述)", lines=4)
            new_source = gr.Textbox(label="来源文件", value="manual")
        with gr.Row():
            create_btn = gr.Button("➕ 新增")
            update_btn = gr.Button("✏️ 更新")

        def _wrap_fetch(lib_type):
            data, msg = kb_fetch(lib_type)
            return data, msg

        refresh_btn.click(_wrap_fetch, inputs=lib_select, outputs=[kb_table, status_box])

        create_btn.click(
            lambda lt, c, s: kb_create(lt, c, s),
            inputs=[lib_select, new_content, new_source],
            outputs=[kb_table, status_box],
        )

        update_btn.click(
            lambda lt, cid, c, s: kb_update(lt, cid, c, s),
            inputs=[lib_select, del_id, new_content, new_source],
            outputs=[kb_table, status_box],
        )

        del_btn.click(
            lambda lt, cid: kb_delete(lt, cid),
            inputs=[lib_select, del_id],
            outputs=[kb_table, status_box],
        )

        # 首次加载自动刷新
        demo.load(_wrap_fetch, inputs=lib_select, outputs=[kb_table, status_box])

    with gr.Tab("🤖 机器狗巡检"):
        if ROBOT_PANEL_OK:
            build_robot_tab(agent=agent, project_root=PROJECT_ROOT)
        else:
            gr.Markdown(
                "### ⚠️ 机器狗面板未能加载\n\n"
                f"`{ROBOT_PANEL_ERR}`\n\n"
                "请确认 `src/robot/` 下的 client.py / patrol.py / ui.py 存在，且已安装 requests。"
            )

    gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    # Gradio 6：theme / css 必须传给 launch() 而不是 Blocks()
    # allowed_paths：放行机器人抓拍照片目录，使聊天内嵌图片（/gradio_api/file=）可加载
    _shots_dir = str((PROJECT_ROOT / "data" / "robot_shots").resolve())
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(PROJECT_ROOT.resolve()), _shots_dir],
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="blue",
            neutral_hue="slate",
            font=["system-ui", "-apple-system", "Segoe UI", "PingFang SC", "Microsoft YaHei", "sans-serif"],
        ),
        css=CHAT_CSS,
    )
