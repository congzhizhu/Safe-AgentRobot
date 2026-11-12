from __future__ import annotations

import json
import base64
import os
import re
from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import requests
from langchain_core.documents import Document

from retriever.clip_engine import CLIPCaseEngine, SimilarCase

try:
    from agent.streaming import StreamingJSONParser, format_partial_chain, iter_sse_deltas
except Exception:  # pragma: no cover - 兼容 src/agent 被当作顶层包的导入方式
    from streaming import StreamingJSONParser, format_partial_chain, iter_sse_deltas

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception as exc:  # pragma: no cover
    raise RuntimeError("缺少 langchain-core，请先安装 langchain。") from exc


# 文本模式 / 图文模式共用的系统提示模板（与 analyze / analyze_stream 保持一致）
_TEXT_SYSTEM_TEMPLATE = (
    "{system_prompt}\n\n"
    "【意图识别与任务处理】\n"
    "请先根据用户问题进行意图识别：greeting / legal_only / hazard_analysis。\n"
    "1) greeting：问候、身份、能力问题。请简短回答，不输出 JSON。\n"
    "2) legal_only：法规咨询、条文解释、处罚依据。请基于 [法条检索] 专业作答，不输出 JSON。\n"
    "3) hazard_analysis：隐患排查、风险研判、整改建议。若无图片则明确说明结论不确定性，再给出通用排查建议。\n\n"
    "【法条引用具体性要求（最高优先级）】\n"
    "引用法条时必须给出：条款号（完整规范格式）+ 从 [法条检索] 中逐字摘录的条款原文（保留数值、单位、时限）。\n"
    "严禁使用“根据相关规定”“按照有关标准”等模糊表述；严禁只写法规名不写条款号。\n"
    "若检索结果中没有可支撑的条款，必须明确说明“检索库中未找到直接对应的条款”，不得虚构条款号或编造条文。\n\n"
    "【整改措施具体性要求】\n"
    "每条整改建议必须包含：措施类型、具体措施内容（落实到设备/动作/参数）、责任主体（岗位级别）、时限与复查方式。\n"
    "严禁“加强管理”“提高认识”等空泛表述。\n\n"
    "【思维链分析要求】（仅在需要进行隐患排查时使用）\n"
    "在给出最终结论前，请先在 JSON 的 \"思维链\" 字段中进行一步步推理：\n"
    "1. 视觉分析：图中观察到了哪些具体的物理特征或作业状态？\n"
    "2. 法规匹配：这些特征对应了检索到的哪些法律条款？逐条引用条款号比对。\n"
    "3. 风险判定：结合图像证据和法条，确定隐患的严重程度。\n"
    "4. 措施建议：基于隐患原因，提出针对性的技术与管理建议。\n\n"
    "如果是隐患排查，你必须严格输出 JSON 对象，包含如下键：\n"
    "思维链, 隐患定性, 法律依据, 整改措施, 参考案例。\n"
    "其中 法律依据数组每一项必须包含 条款号/条款内容/适用说明 三键（条款内容为检索原文逐字摘录）；\n"
    "整改措施数组每一项必须包含 措施类型/措施内容/责任主体/时限要求 四键。\n\n"
    "\n\n[相似案例检索]\n{retrieved_cases}"
    "\n\n[法条检索]\n{retrieved_laws}"
)

_TEXT_USER_TEMPLATE = "用户问题：{question}\n请基于检索上下文给出专业解答。如果涉及法律条文，请直接引用检索到的内容。"


@dataclass
class AgentConfig:
    system_prompt_path: str | Path = "prompts/system_role_prompt.txt"
    api_base: str = "http://127.0.0.1:8000/v1"
    model_name: str = "qwen3vl_lora_local"
    # 纯文本模型：用于自然语言指令解析等不需要视觉的文本任务。
    # 多模态模型（如 qwen3-vl-plus）在纯文本 JSON 结构化输出上的指令遵循较弱，
    # 因此指令解析单独走一个纯文本、指令遵循强的模型。
    text_model_name: str = "qwen-plus"
    api_key: str = ""
    project_root: str | Path = "."
    top_k: int = 3
    legal_top_k: int = 6
    cases_metadata_path: str | Path = "data/data_processed/cases_metadata.json"
    legal_db_dir: str | Path = "data/data_processed/chroma_legal"
    cases_db_dir: str | Path = "data/data_processed/chroma_cases"
    bge_model_name: str | None = None

    def __post_init__(self) -> None:
        if self.bge_model_name is None:
            # 优先使用项目本地模型目录（ModelScope 下载），其次环境变量，最后默认 HF 名称
            local_bge = Path(self.project_root).resolve() / "models" / "bge-small-zh-v1.5"
            if (local_bge / "config.json").exists():
                self.bge_model_name = str(local_bge)
            else:
                self.bge_model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
        if not self.text_model_name:
            self.text_model_name = os.getenv("QWEN_TEXT_MODEL_NAME", "qwen-plus")


class IntentType:
    GREETING = "greeting"
    LEGAL_ONLY = "legal_only"
    HAZARD_ANALYSIS = "hazard_analysis"


class SafetyProductionAgent:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.project_root = Path(self.config.project_root).resolve()
        self.image_retriever = CLIPCaseEngine(project_root=self.project_root, remote=True)
        # 案例文本检索：让纯文字/语音提问（无图）也能命中知识库中的隐患案例
        self.case_text_retriever = CaseTextRetriever(
            project_root=self.project_root,
            metadata_path=self.config.cases_metadata_path,
            persist_directory=self.config.cases_db_dir,
            embedding_model_name=self.config.bge_model_name,
        )
        self.legal_retriever = LegalClauseRetriever(
            project_root=self.project_root,
            metadata_path=self.config.cases_metadata_path,
            persist_directory=self.config.legal_db_dir,
            embedding_model_name=self.config.bge_model_name,
        )
        self.system_prompt = self._load_system_prompt(self.config.system_prompt_path)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_prompt}\n\n"
                    "【意图识别与任务处理】\n"
                    "请先根据用户问题进行意图识别：\n"
                    "- greeting: 问候、身份、能力说明。\n"
                    "- legal_only: 仅法规咨询、条文解释、处罚依据、流程要求。\n"
                    "- hazard_analysis: 现场隐患分析、风险研判、整改建议（特别是含图片）。\n\n"
                    "处理规则：\n"
                    "1) greeting -> 简短自然语言回答，不输出 JSON。\n"
                    "2) legal_only -> 优先依据 [法条检索] 作答，不输出隐患分析 JSON。\n"
                    "3) hazard_analysis -> 结合 [相似案例检索] 与 [法条检索] 输出结构化 JSON。\n\n"
                    "【思维链分析要求】\n"
                    "在给出最终结论前，请先在 JSON 的 \"思维链\" 字段中进行一步步推理：\n"
                    "1. 视觉分析：图中观察到了哪些具体的物理特征或作业状态？\n"
                    "2. 法规匹配：这些特征对应了检索到的哪些法律条款？逐条引用条款号比对。\n"
                    "3. 风险判定：结合图像证据和法条，确定隐患的严重程度。\n"
                    "4. 措施建议：基于隐患原因，提出针对性的技术与管理建议。\n\n"
                    "【法条引用具体性要求（最高优先级）】\n"
                    "“法律依据”数组中每一项必须是包含如下键的对象：\n"
                    "  - 条款号：完整规范格式，如《中华人民共和国安全生产法》第三十三条第一款；\n"
                    "  - 条款内容：从 [法条检索] 上下文中逐字摘录的原文，保留原文中的数值、单位、时限，不得改写或概括；\n"
                    "  - 适用说明：一句话说明本案例中哪个行为/状态触发了该条款。\n"
                    "严禁出现“根据相关规定”“按照有关标准”等模糊表述；严禁只写法规名不写条款号。\n"
                    "若 [法条检索] 中没有可支撑的条款，必须明确写“检索库中未找到直接对应的条款”，不得虚构。\n\n"
                    "【整改措施具体性要求】\n"
                    "“整改措施”数组中每一项必须是包含如下键的对象：\n"
                    "  - 措施类型：技术措施 / 管理措施 / 个体防护 / 应急措施；\n"
                    "  - 措施内容：具体到设备、动作、参数（如“加装 1.2m 高固定式临边防护栏杆并挂设密目安全网”，禁止“加强防护”类空话）；\n"
                    "  - 责任主体：岗位级别（现场作业负责人 / 属地安全管理人员 / 企业主要负责人）；\n"
                    "  - 时限要求：整改时限与复查方式（如“立即停工整改，24 小时内由安全科现场复查”）。\n\n"
                    "你必须严格输出 JSON 对象，且包含如下键：\n"
                    "思维链, 隐患定性, 法律依据, 整改措施, 参考案例。\n"
                    "其中 法律依据/整改措施/参考案例 必须是数组。\n"
                    "\n\n[相似案例检索]\n{retrieved_cases}"
                    "\n\n[法条检索]\n{retrieved_laws}",
                ),
                (
                    "user",
                    "<image>\n用户问题：{question}\n图像路径：{image_path}\n请基于图片与检索上下文给出结构化结论。",
                ),
            ]
        )

    # ------------------------------------------------------------------
    # 公共准备流程：解析图片 -> 意图识别 -> 双路检索 -> 构造消息
    # analyze()（一次性返回）与 analyze_stream()（边思考边生成）共用，
    # 避免两份逻辑各自漂移。
    # ------------------------------------------------------------------

    def _retrieve_cases(
        self,
        question: str,
        image_abs_path: str | None,
        is_legal_only: bool,
        is_greeting: bool,
    ) -> list[SimilarCase]:
        """检索隐患案例：有图走 CLIP 以图搜图，无图（语音/纯文字）走案例文本库。

        原先无图提问时案例库完全不参与，模型只能看到空的 [相似案例检索]，
        从而误判为"知识库未加载"。这里补上文本通路，保证知识库管理中的
        隐患案例在纯文字提问下同样参与作答。
        """
        if is_greeting:
            return []

        if image_abs_path and not is_legal_only:
            try:
                cases = self.image_retriever.search_similar_cases(
                    query_image=image_abs_path,
                    top_k=self.config.top_k,
                )
            except Exception:
                cases = []
            if cases:
                return cases

        # 无图、以图搜图无命中、或纯法规咨询：改用案例文本库语义检索
        try:
            return self.case_text_retriever.search(query=question, top_k=self.config.top_k)
        except Exception:
            return []
    # ------------------------------------------------------------------
    def _prepare(
        self, image_path: str | Path | None, question: str, force_hazard: bool = False
    ) -> dict[str, Any]:
        # 清理 image_path 并解析绝对路径
        image_abs_path = self._resolve_image_path(image_path) if image_path else None

        # 意图识别逻辑：统一入口，优先识别寒暄，再区分法条咨询与隐患分析。
        # force_hazard=True 时（如机器狗巡检）跳过意图识别的 LLM 调用，
        # 直接按 hazard_analysis 处理，避免每张照片多一次网络往返。
        if force_hazard:
            intent = IntentType.HAZARD_ANALYSIS
        else:
            intent = self.detect_intent(question=question, has_image=bool(image_abs_path))
        is_greeting = intent == IntentType.GREETING
        is_legal_only = intent == IntentType.LEGAL_ONLY

        similar_cases = self._retrieve_cases(
            question=question,
            image_abs_path=image_abs_path,
            is_legal_only=is_legal_only,
            is_greeting=is_greeting,
        )

        # 法律库根据是否为闲聊稍微调控 depth
        legal_top_k = 0 if is_greeting else (self.config.legal_top_k * 2 if is_legal_only else self.config.legal_top_k)

        laws = []
        if legal_top_k > 0:
            laws = self.legal_retriever.search(query=question, top_k=legal_top_k)

        return {
            "image_abs_path": image_abs_path,
            "intent": intent,
            "similar_cases": similar_cases,
            "laws": laws,
            "retrieved_cases": self._format_similar_cases(similar_cases),
            "retrieved_laws": self._format_laws(laws),
        }

    def _build_payload_messages(
        self,
        image_abs_path: str | None,
        question: str,
        retrieved_cases: str,
        retrieved_laws: str,
    ) -> list[dict[str, Any]]:
        """构造最终发给 OpenAI 兼容接口的 messages（有图时转 base64）。"""
        prompt_kwargs = {
            "system_prompt": self.system_prompt,
            "retrieved_cases": retrieved_cases,
            "retrieved_laws": retrieved_laws,
            "question": question,
        }

        if not image_abs_path:
            # 纯文本模式：手动格式化，避免 LangChain 版本差异
            payload_messages = [
                {"role": "system", "content": _TEXT_SYSTEM_TEMPLATE.format(**prompt_kwargs)},
                {"role": "user", "content": _TEXT_USER_TEMPLATE.format(**prompt_kwargs)},
            ]
            return payload_messages

        # 含图片模式：先走 LangChain 模板，再把 <image> 占位符换成 base64
        prompt_kwargs["image_path"] = image_abs_path
        formatted_messages = self.prompt.format_messages(**prompt_kwargs)

        payload_messages: list[dict[str, Any]] = []
        for message in formatted_messages:
            role = self._to_openai_role(message.type)
            if role == "user":
                with open(image_abs_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    base64_image = f"data:image/jpeg;base64,{encoded_string}"
                payload_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": str(message.content)},
                            {"type": "image_url", "image_url": {"url": base64_image}},
                        ],
                    }
                )
            else:
                payload_messages.append({"role": role, "content": str(message.content)})

        return payload_messages

    def analyze_stream(
        self,
        image_path: str | Path | None,
        question: str = "请分析该现场隐患",
        force_hazard: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """边思考边生成的分析入口（生成器）。

        事件序列：

        - ``{"type": "status", "stage": "intent", ...}``  意图识别完成
        - ``{"type": "status", "stage": "retrieved", ...}`` RAG 检索完成
        - ``{"type": "raw_delta", "text": "..."}``        原始 token 增量
        - ``{"type": "field_start", "key": "思维链"}``     某字段开始生成
        - ``{"type": "field_text", "key": ..., "text": ...}`` 该字段当前可读文本
        - ``{"type": "field_done", "key": ..., "value": ...}`` 该字段已完整
        - ``{"type": "done", "result": {...}}``           全流程结束

        界面只要消费 ``field_text``，就能做到思维链逐字上屏。

        force_hazard=True 时跳过意图识别的额外 LLM 调用（机器狗巡检等
        固定为隐患分析的场景），直接把 intent 视为 hazard_analysis，可显著
        降低单张照片的诊断延迟。
        """
        prepared = self._prepare(
            image_path=image_path, question=question, force_hazard=force_hazard
        )

        yield {
            "type": "status",
            "stage": "intent",
            "intent": prepared["intent"],
            "has_image": bool(prepared["image_abs_path"]),
        }

        payload_messages = self._build_payload_messages(
            image_abs_path=prepared["image_abs_path"],
            question=question,
            retrieved_cases=prepared["retrieved_cases"],
            retrieved_laws=prepared["retrieved_laws"],
        )

        yield {
            "type": "status",
            "stage": "retrieved",
            "similar_cases": [item.__dict__ for item in prepared["similar_cases"]]
            if prepared["similar_cases"]
            else [],
            "retrieved_laws": prepared["laws"],
        }

        parser = StreamingJSONParser()
        raw_parts: list[str] = []

        try:
            for delta in self._chat_completion_stream(payload_messages):
                raw_parts.append(delta)
                yield {"type": "raw_delta", "text": delta}
                for event in parser.feed(delta):
                    yield event
        except Exception as exc:
            # 流式失败时降级为一次性生成，保证功能可用
            yield {"type": "warn", "message": f"流式推理失败，已降级为一次性生成：{exc}"}
            answer = self._chat_completion(payload_messages)
            raw_parts = [answer]
            yield {"type": "raw_delta", "text": answer}
            for event in parser.feed(answer):
                yield event

        for event in parser.finish():
            yield event

        raw_answer = "".join(raw_parts)
        structured = dict(parser.fields) if parser.fields else self._parse_json_output(raw_answer)
        structured = self._normalize_structured(structured)

        yield {
            "type": "done",
            "result": {
                "question": question,
                "image_path": str(image_path) if image_path else None,
                "intent": prepared["intent"],
                "similar_cases": [item.__dict__ for item in prepared["similar_cases"]]
                if prepared["similar_cases"]
                else [],
                "retrieved_laws": prepared["laws"],
                "raw_answer": raw_answer,
                "structured_output": structured,
            },
        }

    def analyze(self, image_path: str | Path | None, question: str = "请分析该现场隐患") -> dict[str, Any]:
        # 清理 image_path 并解析绝对路径
        image_abs_path = self._resolve_image_path(image_path) if image_path else None

        # 意图识别逻辑：统一入口，优先识别寒暄，再区分法条咨询与隐患分析
        intent = self.detect_intent(question=question, has_image=bool(image_abs_path))
        is_greeting = intent == IntentType.GREETING
        is_legal_only = intent == IntentType.LEGAL_ONLY

        similar_cases = self._retrieve_cases(
            question=question,
            image_abs_path=image_abs_path,
            is_legal_only=is_legal_only,
            is_greeting=is_greeting,
        )

        # 法律库根据是否为闲聊稍微调控 depth
        legal_top_k = 0 if is_greeting else (self.config.legal_top_k * 2 if is_legal_only else self.config.legal_top_k)

        laws = []
        if legal_top_k > 0:
            laws = self.legal_retriever.search(
                query=question,
                top_k=legal_top_k,
            )

        retrieved_cases = self._format_similar_cases(similar_cases)
        retrieved_laws = self._format_laws(laws)

        payload_messages = self._build_payload_messages(
            image_abs_path=image_abs_path,
            question=question,
            retrieved_cases=retrieved_cases,
            retrieved_laws=retrieved_laws,
        )

        answer = self._chat_completion(payload_messages)

        parsed = self._parse_json_output(answer)
        return {
            "question": question,
            "image_path": str(image_path) if image_path else None,
            "intent": intent,
            "similar_cases": [item.__dict__ for item in similar_cases] if similar_cases else [],
            "retrieved_laws": laws,
            "raw_answer": answer,
            "structured_output": parsed,
        }

    def detect_intent(self, question: str, has_image: bool) -> str:
        """Public intent classifier: LLM-first with regex fallback."""
        llm_intent = self._detect_intent_with_llm(question=question, has_image=has_image)
        if llm_intent in {IntentType.GREETING, IntentType.LEGAL_ONLY, IntentType.HAZARD_ANALYSIS}:
            return llm_intent
        return self._detect_intent(question=question, has_image=has_image)

    def _detect_intent_with_llm(self, question: str, has_image: bool) -> str | None:
        system_prompt = (
            "你是意图分类器，只做分类不做回答。"
            "你只能输出一个 JSON 对象，格式为: "
            '{"intent":"greeting|legal_only|hazard_analysis","confidence":0~1,"reason":"..."}'
            "。"
            "分类规则："
            "greeting=问候/身份/闲聊；"
            "legal_only=法规条文、合规解释、处罚依据；"
            "hazard_analysis=隐患排查、风险研判、整改建议。"
            "若有图片且语义不明确，优先 hazard_analysis。"
            "禁止输出除 JSON 外的任何文本。"
        )
        user_prompt = (
            f"问题: {question}\n"
            f"是否有图片: {'是' if has_image else '否'}\n"
            "请只返回 JSON。"
        )
        try:
            raw = self._chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=120,
                timeout=12,
            )
            parsed = self._parse_json_output(raw)
            intent = str(parsed.get("intent", "")).strip()
            if intent in {IntentType.GREETING, IntentType.LEGAL_ONLY, IntentType.HAZARD_ANALYSIS}:
                return intent
        except Exception:
            return None
        return None

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 300,
        model: str | None = None,
    ) -> str:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = {
            "model": model or self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"].get("content", ""))

    def _chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> Iterator[str]:
        """以 SSE 方式请求对话接口，逐段产出文本增量。"""
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        body = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "max_tokens": max_tokens,
            "stream": True,
        }

        with requests.post(url, headers=headers, json=body, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for delta in iter_sse_deltas(resp.iter_lines()):
                yield delta

    @staticmethod
    def _format_similar_cases(items: list[Any]) -> str:
        if not items:
            return "未检索到相似案例。"
        lines: list[str] = []
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"[{idx}] 来源: {item.source_file} | case_id: {item.case_id} | score: {item.score:.3f}\n"
                f"隐患描述: {item.issue_text}\n"
                f"法定依据: {item.legal_basis}\n"
                f"整改建议: {item.suggestion}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_laws(items: list[dict[str, Any]]) -> str:
        if not items:
            return "未检索到法条。"
        lines: list[str] = []
        for idx, item in enumerate(items, start=1):
            source = item.get("source_file", "未知文件")
            # 格式化输出更加友好
            lines.append(
                f"[{idx}] [{source}] (相似距离: {item['score']:.3f})\n"
                f"{item['legal_basis']}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _to_openai_role(role: str) -> str:
        mapping = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        return mapping.get(role, role)

    @staticmethod
    def _load_system_prompt(path: str | Path) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"未找到 system prompt: {p}")
        return p.read_text(encoding="utf-8")

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    def _detect_intent(self, question: str, has_image: bool) -> str:
        q = self._normalize_text(question)

        greeting_patterns = [
            r"你好|您好|hi|hello",
            r"你是谁|你是做什么|你能做什么|怎么用",
            r"介绍一下|自我介绍|help|帮助",
        ]
        legal_patterns = [
            r"法律|法规|条例|条款|依据|处罚|罚则|违法|合规",
            r"是否违法|是否合法|怎么规定|如何规定|第[一二三四五六七八九十0-9]+条",
        ]
        hazard_patterns = [
            r"隐患|危险|风险|违章|整改|排查|评估|研判|定级",
            r"图片|图中|照片|现场|作业|施工|设备|防护",
            r"看一下|帮我分析|判断一下",
        ]

        if any(re.search(p, q) for p in greeting_patterns):
            return IntentType.GREETING

        legal_hit = any(re.search(p, q) for p in legal_patterns)
        hazard_hit = any(re.search(p, q) for p in hazard_patterns)

        if has_image:
            # 有图时优先按隐患分析路由，除非明确仅问法规且没有隐患语义。
            if legal_hit and not hazard_hit:
                return IntentType.LEGAL_ONLY
            return IntentType.HAZARD_ANALYSIS

        if hazard_hit and not legal_hit:
            return IntentType.HAZARD_ANALYSIS
        if legal_hit:
            return IntentType.LEGAL_ONLY
        return IntentType.HAZARD_ANALYSIS

    def _resolve_image_path(self, image_path: str | Path | None) -> str:
        if not image_path:
            return ""
        path = Path(image_path)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {path}")
        return str(path)

    @staticmethod
    def _normalize_structured(structured: dict[str, Any]) -> dict[str, Any]:
        """归一化 LLM 产出的结构化字段，防止类型错乱导致下游渲染异常。

        LLM 有时不遵守 JSON 契约，把「参考案例/法律依据/整改措施」输出成字符串
        （甚至混入未闭合的 JSON 片段）而非数组。下游 `for x in items` 遍历字符串会
        逐字符输出（表现成「来/源/：/现/场/安/全/检/查」）。这里把已知应为数组的字段
        强制归一到 list，并对混入后续字段的脏字符串做截断清洗。
        """
        if not isinstance(structured, dict):
            return structured
        result = dict(structured)
        # 必须为数组的字段
        list_fields = ("参考案例", "法律依据", "整改措施", "思维链", "引用校验")
        for key in list_fields:
            val = result.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                continue
            if isinstance(val, str):
                result[key] = SafetyProductionAgent._split_field_text(val)
            elif isinstance(val, dict):
                # 单个 dict 包成列表
                result[key] = [val]
            else:
                result[key] = [val]
        return result

    @staticmethod
    def _split_field_text(val: str) -> list[str]:
        """把一个字段的字符串值拆成列表，并截断混入的后续字段泄漏。

        典型脏数据：LLM 把「参考案例」字段写成纯文本，且引号未闭合，导致后面本该是
        独立字段的「思维链/隐患定性/法律依据/整改措施/参考案例」整体被吞进这个字符串。
        这里在遇到这些字段名标记（顶格或带冒号）时截断，只保留前面的合法内容。
        """
        # 截断标记：出现这些字段名（通常意味着后续是泄漏进来的其它字段）即停止
        leak_markers = (
            "思维链", "隐患定性", "法律依据", "整改措施", "参考案例", "引用校验",
        )
        # 找到最早出现的泄漏点
        cut = len(val)
        for marker in leak_markers:
            # 匹配「字段名:」或「字段名: [」等模式（允许全角冒号）
            for sep in (": ", "：", ":", "：[", ":[", ":{"):
                idx = val.find(marker + sep)
                if idx != -1:
                    cut = min(cut, idx)
        val = val[:cut]

        # 按换行拆分，过滤空行和明显的 JSON 结构残片（{ } [ ] 开头或仅标点）
        parts = [p.strip() for p in val.split("\n") if p.strip()]
        cleaned = [p for p in parts if not p.lstrip().startswith(("{", "}", "[", "]", ","))]
        return cleaned

    @staticmethod
    def _parse_json_output(text: str) -> dict[str, Any]:
        text = text.strip()

        # 剥掉 markdown 代码围栏
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)

        candidates = [text]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

        for cand in candidates:
            # 1) 原样解析
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return SafetyProductionAgent._normalize_structured(parsed)
            except Exception:
                pass
            # 2) 修复常见语法错误后再解析（分号分隔对象、尾逗号等）
            repaired = SafetyProductionAgent._repair_json_text(cand)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return SafetyProductionAgent._normalize_structured(parsed)
            except Exception:
                pass
            # 3) json5 宽松解析（容忍尾逗号/单引号/注释等）
            try:
                import json5  # type: ignore

                parsed = json5.loads(repaired)
                if isinstance(parsed, dict):
                    return SafetyProductionAgent._normalize_structured(parsed)
            except Exception:
                pass
        return {}

    @staticmethod
    def _repair_json_text(s: str) -> str:
        """修复大模型输出 JSON 时最常见的语法错误（含全角标点）。"""
        # 对象/数组后误用分号或全角逗号/分号: } ; 、 } ， } ； -> },
        s = re.sub(r"([}\]])\s*[;；，]\s*", r"\1,", s)
        # 冒号误用全角: "key"： -> "key":
        s = re.sub(r'("\s*)\s*：\s*', r"\1: ", s)
        # 去掉对象/数组后的尾逗号
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s


class LegalClauseRetriever:
    def __init__(
        self,
        project_root: str | Path,
        metadata_path: str | Path,
        persist_directory: str | Path,
        embedding_model_name: str,
        local: bool = False,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.metadata_path = self.project_root / metadata_path
        self.persist_directory = self.project_root / persist_directory
        self.local = local  # True: 本进程直接读写 Chroma（仅入库 API 进程使用）
        self.embedding_model_name = embedding_model_name
        self.use_lexical_fallback = False
        self.store = None
        self.embedding = None
        self.lexical_cases = self._load_cases()
        self._metadata_mtime = self._stat_mtime()

        # 远程模式（Agent/WebUI 进程默认）：不直接打开 Chroma。
        # Chroma 不支持多进程并发访问同一目录，统一经 8001 API 检索。
        if not self.local:
            self.api_base = os.getenv("RETRIEVAL_API_BASE", "http://127.0.0.1:8001")
            return

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        offline_mode = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        if offline_mode:
            self.use_lexical_fallback = True
            return

        try:
            embeddings_cls = self._load_hf_embeddings_cls()
            self.embedding = embeddings_cls(
                model_name=embedding_model_name,
                model_kwargs={"device": "cuda" if self._has_cuda() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            chroma_cls = self._load_chroma_cls()
            self.store = chroma_cls(
                collection_name="legal_clauses",
                embedding_function=self.embedding,
                persist_directory=str(self.persist_directory),
            )
            self._ensure_indexed()
        except Exception:
            self.use_lexical_fallback = True

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.local:
            remote = self._search_remote(query, top_k)
            # 注意：远程接口可能返回空列表（向量库为空或检索服务异常），
            # 此时必须回退本地词法检索，否则模型会认为"法条库未加载"。
            if remote:
                return remote
            return self._search_lexical(query=query, top_k=top_k)

        # 本地模式（入库 API 进程）
        self._refresh_if_stale()
        if self.use_lexical_fallback or self.store is None:
            return self._search_lexical(query=query, top_k=top_k)

        # Chroma 相似度搜索返回 (doc, score)，其中 score 是 L2 距离。
        # 距离越小表示越相似。
        results = self.store.similarity_search_with_score(query, k=max(top_k, 5))

        output: list[dict[str, Any]] = []
        for doc, score in results:
            output.append(
                {
                    "case_id": str(doc.metadata.get("case_id", "law_clause")),
                    "source_file": str(doc.metadata.get("source_file", "未知文件")),
                    "legal_basis": doc.page_content,
                    "score": float(score),
                }
            )
        # 按距离升序排列（最相似的在前）
        output.sort(key=lambda item: item["score"])
        return output[:top_k]

    def _search_remote(self, query: str, top_k: int) -> list[dict[str, Any]] | None:
        """经入库 API 检索；失败返回 None 以便回退到词法检索。"""
        try:
            resp = requests.post(
                f"{self.api_base}/search/laws",
                json={"query": query, "top_k": top_k},
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                return [
                    {
                        "case_id": str(it.get("case_id", "law_clause")),
                        "source_file": str(it.get("source_file", "未知文件")),
                        "legal_basis": str(it.get("legal_basis", "")),
                        "score": float(it.get("score", 0.0)),
                    }
                    for it in results
                ]
        except Exception:
            pass
        return None

    def _search_lexical(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: list[dict[str, Any]] = []
        for case in self.lexical_cases:
            legal_basis = str(case.get("legal_basis", "")).strip()
            if not legal_basis:
                continue
            tokens = self._tokenize(legal_basis)
            if not tokens:
                continue

            overlap = query_tokens.intersection(tokens)
            if not overlap:
                continue

            similarity = len(overlap) / max(len(query_tokens), 1)
            scored.append(
                {
                    "case_id": str(case.get("id", "")),
                    "source_file": str(case.get("source_file", "")),
                    "legal_basis": legal_basis,
                    "score": float(1.0 - similarity),
                }
            )

        scored.sort(key=lambda item: item["score"])
        return scored[: max(top_k, 1)]

    def _ensure_indexed(self) -> None:
        existing = self.store._collection.count()
        if existing > 0:
            return

        cases = self._load_cases()
        docs: list[Document] = []
        for case in cases:
            legal_basis = str(case.get("legal_basis", "")).strip()
            if not legal_basis:
                continue
            docs.append(
                Document(
                    page_content=legal_basis,
                    metadata={
                        "case_id": str(case.get("id", "")),
                        "source_file": str(case.get("source_file", "")),
                    },
                )
            )
        if docs:
            self.store.add_documents(docs)

    def _load_cases(self) -> list[dict[str, Any]]:
        if not self.metadata_path.exists():
            return []
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return raw.get("cases", []) if isinstance(raw, dict) else []

    def _stat_mtime(self) -> float:
        try:
            return self.metadata_path.stat().st_mtime
        except OSError:
            return 0.0

    def _refresh_if_stale(self) -> None:
        """入库 API（独立进程）更新 metadata 后，运行中的服务必须重载才能看到新数据：
        重载词法索引，并重新打开 Chroma 向量库以加载其他进程新写入的向量。"""
        mtime = self._stat_mtime()
        if mtime == self._metadata_mtime:
            return
        self._metadata_mtime = mtime
        self.lexical_cases = self._load_cases()
        if self.use_lexical_fallback or self.embedding is None:
            return
        try:
            chroma_cls = self._load_chroma_cls()
            self.store = chroma_cls(
                collection_name="legal_clauses",
                embedding_function=self.embedding,
                persist_directory=str(self.persist_directory),
            )
        except Exception:
            pass

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """分词：中文按 2-gram 切分，英文/数字按词切分。

        旧实现用 `[\\u4e00-\\u9fa5]{2,}` 会把一整段连续中文当成一个 token，
        导致查询与文档的 token 几乎不可能相交，词法兜底形同虚设（永远返回空）。
        改为 2-gram 后，"高处作业" 与 "高处作业安全规范" 才能正确匹配。
        """
        normalized = re.sub(r"\s+", "", text or "")
        tokens: set[str] = set()
        for run in re.findall(r"[\u4e00-\u9fa5]+", normalized):
            if len(run) == 1:
                tokens.add(run)
                continue
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
        tokens.update(re.findall(r"[A-Za-z]+", normalized))
        tokens.update(re.findall(r"\d+", normalized))
        return tokens

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _load_chroma_cls() -> Any:
        candidates = [
            ("langchain_chroma", "Chroma"),
            ("langchain_community.vectorstores", "Chroma"),
            ("langchain.vectorstores", "Chroma"),
        ]
        for module_name, attr in candidates:
            try:
                module = import_module(module_name)
                return getattr(module, attr)
            except Exception:
                continue
        raise RuntimeError("未找到 Chroma 实现，请安装 langchain-chroma 或 langchain-community。")

    @staticmethod
    def _load_hf_embeddings_cls() -> Any:
        candidates = [
            ("langchain_huggingface", "HuggingFaceEmbeddings"),
            ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
            ("langchain.embeddings", "HuggingFaceEmbeddings"),
        ]
        for module_name, attr in candidates:
            try:
                module = import_module(module_name)
                return getattr(module, attr)
            except Exception:
                continue
        raise RuntimeError(
            "未找到 HuggingFaceEmbeddings 实现，请安装 langchain-huggingface 或 langchain-community。"
        )


class CaseTextRetriever:
    """隐患案例文本检索器。

    案例库原本只支持 CLIP「以图搜图」，纯文字提问（语音问答、法规咨询）永远检索不到，
    模型看到空的 [相似案例检索] 会误判为"知识库未加载"。本类基于案例元数据
    （隐患描述 / 法定依据 / 整改建议）做语义检索 + 词法兜底，让知识库管理中的
    隐患案例在无图提问时同样参与作答。

    进程模型与 LegalClauseRetriever 完全一致：
    - local=True（入库 API 进程，8001）：直接读写 Chroma，是索引的唯一所有者；
    - local=False（Agent / WebUI 进程）：经 /search/cases 远程检索，失败回退本地词法。
    """

    COLLECTION_NAME = "hazard_cases_text"

    def __init__(
        self,
        project_root: str | Path,
        metadata_path: str | Path,
        persist_directory: str | Path,
        embedding_model_name: str,
        local: bool = False,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.metadata_path = self.project_root / metadata_path
        self.persist_directory = self.project_root / persist_directory
        self.embedding_model_name = embedding_model_name
        self.local = local
        self.use_lexical_fallback = False
        self.store = None
        self.embedding = None
        self.lexical_cases: list[dict[str, Any]] = self._load_cases()
        self._metadata_mtime = self._stat_mtime()

        if not self.local:
            self.api_base = os.getenv("RETRIEVAL_API_BASE", "http://127.0.0.1:8001")
            return

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        offline_mode = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        if offline_mode:
            self.use_lexical_fallback = True
            return

        try:
            embeddings_cls = LegalClauseRetriever._load_hf_embeddings_cls()
            self.embedding = embeddings_cls(
                model_name=embedding_model_name,
                model_kwargs={"device": "cuda" if LegalClauseRetriever._has_cuda() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            chroma_cls = LegalClauseRetriever._load_chroma_cls()
            self.store = chroma_cls(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_directory),
            )
            self._ensure_indexed()
        except Exception:
            self.use_lexical_fallback = True

    # ------------------------------------------------------------------
    # 检索入口
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> list[SimilarCase]:
        if self.local:
            self._refresh_if_stale()
            if not self.use_lexical_fallback and self.store is not None:
                try:
                    hits = self._search_vector(query, top_k)
                except Exception:
                    hits = []
                if hits:
                    return hits
            return self._search_lexical(query=query, top_k=top_k)

        remote = self._search_remote(query, top_k)
        if remote:
            return remote
        return self._search_lexical(query=query, top_k=top_k)

    def _search_vector(self, query: str, top_k: int) -> list[SimilarCase]:
        results = self.store.similarity_search_with_score(query, k=max(top_k, 5))
        output: list[SimilarCase] = []
        for doc, score in results:
            metadata = doc.metadata or {}
            output.append(
                SimilarCase(
                    case_id=str(metadata.get("case_id", "")),
                    image_id=str(metadata.get("image_id", "")),
                    image_path=str(metadata.get("image_path", "")),
                    issue_text=str(metadata.get("issue_text", "")),
                    legal_basis=str(metadata.get("legal_basis", "")),
                    suggestion=str(metadata.get("suggestion", "")),
                    source_file=str(metadata.get("source_file", "未知文件")),
                    score=float(score),
                )
            )
        output.sort(key=lambda item: item.score)
        return output[:top_k]

    def _search_remote(self, query: str, top_k: int) -> list[SimilarCase] | None:
        try:
            resp = requests.post(
                f"{self.api_base}/search/cases",
                json={"query": query, "top_k": top_k},
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                return [
                    SimilarCase(
                        case_id=str(item.get("case_id", "")),
                        image_id=str(item.get("image_id", "")),
                        image_path=str(item.get("image_path", "")),
                        issue_text=str(item.get("issue_text", "")),
                        legal_basis=str(item.get("legal_basis", "")),
                        suggestion=str(item.get("suggestion", "")),
                        source_file=str(item.get("source_file", "未知文件")),
                        score=float(item.get("score", 0.0)),
                    )
                    for item in data.get("results", [])
                ]
        except Exception:
            pass
        return None

    def _search_lexical(self, query: str, top_k: int = 3) -> list[SimilarCase]:
        query_tokens = LegalClauseRetriever._tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for case in self.lexical_cases:
            haystack = " ".join(
                str(case.get(key, "") or "")
                for key in ("issue_text", "legal_basis", "suggestion")
            )
            tokens = LegalClauseRetriever._tokenize(haystack)
            if not tokens:
                continue
            overlap = query_tokens.intersection(tokens)
            if not overlap:
                continue
            similarity = len(overlap) / max(len(query_tokens), 1)
            scored.append((similarity, case))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._to_case(case, score=1.0 - sim) for sim, case in scored[: max(top_k, 1)]]

    # ------------------------------------------------------------------
    # 索引维护（仅入库 API 进程调用）
    # ------------------------------------------------------------------
    @staticmethod
    def _doc_text(case: dict[str, Any]) -> str:
        parts: list[str] = []
        issue = str(case.get("issue_text", "") or "").strip()
        legal = str(case.get("legal_basis", "") or "").strip()
        suggestion = str(case.get("suggestion", "") or "").strip()
        if issue:
            parts.append(f"隐患描述：{issue}")
        if legal:
            parts.append(f"法定依据：{legal}")
        if suggestion:
            parts.append(f"整改建议：{suggestion}")
        return "\n".join(parts)

    @staticmethod
    def _metadata_of(case: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_id": str(case.get("id", "")),
            "image_id": str(case.get("image_id", "")),
            "image_path": str(case.get("image_path", "")),
            "issue_text": str(case.get("issue_text", "") or ""),
            "legal_basis": str(case.get("legal_basis", "") or ""),
            "suggestion": str(case.get("suggestion", "") or ""),
            "source_file": str(case.get("source_file", "未知文件")),
        }

    def _ensure_indexed(self) -> None:
        try:
            if self.store is None or self.store._collection.count() > 0:
                return
        except Exception:
            return
        self.reindex()

    def reindex(self) -> int:
        """按 cases_metadata.json 全量重建案例文本索引（可重复调用）。"""
        if self.store is None:
            return 0
        cases = self._load_cases()
        docs: list[Document] = []
        ids: list[str] = []
        for idx, case in enumerate(cases):
            text = self._doc_text(case)
            if not text:
                continue
            case_id = str(case.get("id", "") or f"case_{idx}")
            ids.append(case_id)
            docs.append(Document(page_content=text, metadata=self._metadata_of(case)))

        # 先清空旧索引：Chroma 对已存在的 id 会抛 DuplicateIDError
        try:
            existing = self.store._collection.get()
            if existing and existing.get("ids"):
                self.store._collection.delete(ids=existing["ids"])
        except Exception:
            pass

        if docs:
            self.store.add_documents(docs, ids=ids)
        self.lexical_cases = cases
        return len(docs)

    def upsert_case(self, case: dict[str, Any]) -> None:
        """单个案例写入/更新（知识库 CRUD 或文件上传后调用）。"""
        if self.store is None:
            return
        text = self._doc_text(case)
        if not text:
            return
        self.delete_case(str(case.get("id", "")))
        self.store.add_documents(
            [Document(page_content=text, metadata=self._metadata_of(case))],
            ids=[str(case.get("id", ""))],
        )

    def delete_case(self, case_id: str) -> None:
        if self.store is None:
            return
        try:
            self.store._collection.delete(where={"case_id": case_id})
        except Exception:
            pass
        try:
            self.store.delete(ids=[case_id])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------
    def _to_case(self, case: dict[str, Any], score: float) -> SimilarCase:
        return SimilarCase(
            case_id=str(case.get("id", "")),
            image_id=str(case.get("image_id", "")),
            image_path=str(case.get("image_path", "")),
            issue_text=str(case.get("issue_text", "") or ""),
            legal_basis=str(case.get("legal_basis", "") or ""),
            suggestion=str(case.get("suggestion", "") or ""),
            source_file=str(case.get("source_file", "未知文件")),
            score=float(score),
        )

    def _load_cases(self) -> list[dict[str, Any]]:
        if not self.metadata_path.exists():
            return []
        try:
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return raw.get("cases", []) if isinstance(raw, dict) else []

    def _stat_mtime(self) -> float:
        try:
            return self.metadata_path.stat().st_mtime
        except OSError:
            return 0.0

    def _refresh_if_stale(self) -> None:
        mtime = self._stat_mtime()
        if mtime == self._metadata_mtime:
            return
        self._metadata_mtime = mtime
        self.lexical_cases = self._load_cases()
        if self.use_lexical_fallback or self.embedding is None:
            return
        try:
            chroma_cls = LegalClauseRetriever._load_chroma_cls()
            self.store = chroma_cls(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embedding,
                persist_directory=str(self.persist_directory),
            )
        except Exception:
            pass


def build_agent_from_env() -> SafetyProductionAgent:
    project_root = os.getenv("PROJECT_ROOT", ".")
    cfg = AgentConfig(
        system_prompt_path=os.getenv("SYSTEM_PROMPT_PATH", "prompts/system_role_prompt.txt"),
        api_base=os.getenv("QWEN_API_BASE", "http://127.0.0.1:8000/v1"),
        model_name=os.getenv("QWEN_MODEL_NAME", "qwen3vl_lora_local"),
        text_model_name=os.getenv("QWEN_TEXT_MODEL_NAME", "qwen-plus"),
        api_key=os.getenv("QWEN_API_KEY", ""),
        top_k=int(os.getenv("RETRIEVE_TOP_K", "3")),
        project_root=project_root,
    )
    return SafetyProductionAgent(cfg)
