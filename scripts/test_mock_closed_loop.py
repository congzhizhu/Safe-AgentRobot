# -*- coding: utf-8 -*-
"""
mock 模式闭环自测：不控真狗，验证「下指令 → 计划 → 巡检执行 → 拍照(mock) → 诊断 → 报告」全链路。

运行：
    cd /d D:/Safety_Agent
    .venv/Scripts/python.exe scripts/test_mock_closed_loop.py

验证点：
1. RobotClient 在 8010 服务不可达时自动进入 mock 模式（用样本图冒充狗拍的照片）
2. plan_with_llm 把自然语言解析成动作计划（真调云端 Qwen-VL）
3. execute_plan_iter 逐步执行计划，逐张诊断（真调云端 Qwen-VL 多模态）
4. 最终产出 report.md + summary.json
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for p in (str(SRC),):
    if p not in sys.path:
        sys.path.insert(0, p)

# 从 .env 加载环境变量（云端百炼 API 配置）
def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

load_dotenv(PROJECT_ROOT / ".env")

from agent.workflow import build_agent_from_env
from robot.client import RobotClient
from robot.patrol import plan_with_llm, describe_plan, execute_plan_iter


def main() -> None:
    print("=" * 60)
    print("[0] 构建智能体（云端 Qwen-VL）")
    agent = build_agent_from_env()
    print(f"    模型: {agent.config.model_name}")
    print(f"    API : {agent.config.api_base}")
    print()

    print("[1] 构建 RobotClient（应进入 mock 模式，因为 8010 未启动）")
    client = RobotClient(project_root=PROJECT_ROOT, allow_mock=True)
    snap = client.snapshot()
    print(f"    mock={snap.mock}, connected={snap.connected}")
    print(f"    提示: {snap.note}")
    print()
    assert snap.mock, "预期进入 mock 模式但未进入"

    text = "巡检一圈，拍 4 张照片"
    print(f"[2] 自然语言指令: {text!r}")
    print("    正在让智能体解析为动作计划（真调云端 Qwen-VL）...")
    ops, say = plan_with_llm(agent, text)
    print(f"    智能体理解: {say}")
    print(f"    计划步骤: {len(ops)} 个")
    print(describe_plan(ops))
    print()
    assert ops, "计划解析失败，无可用动作"

    print("[3] 执行计划（mock 拍照 + 云端 Qwen-VL 逐张诊断）...")
    print("-" * 60)
    question = "请分析这张机器狗巡检拍摄的现场照片，判断是否存在安全隐患并进行隐患定性。"
    final = None
    for snap_dict in execute_plan_iter(
        client, ops, question=question, agent=agent, diagnose=True, zone_name="mock自测"
    ):
        status = snap_dict["status"]
        n_gallery = len(snap_dict["gallery"])
        print(f"    [进度] {status}  照片数={n_gallery}")
        final = snap_dict

    print("-" * 60)
    print("[4] 结果")
    report_path = final.get("report_path", "")
    print(f"    报告路径: {report_path}")
    if report_path and Path(report_path).exists():
        print("    ✅ report.md 已生成")
        print()
        print("    === 报告摘要（前 40 行）===")
        lines = Path(report_path).read_text(encoding="utf-8").splitlines()
        for ln in lines[:40]:
            print("    " + ln)
    else:
        print("    ⚠️ 未生成报告")

    print()
    print("=" * 60)
    print("✅ mock 闭环自测完成")


if __name__ == "__main__":
    main()
