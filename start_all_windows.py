#!/usr/bin/env python3
"""Safety_Agent Windows 一键启动器
用法:  .venv\\Scripts\\python.exe start_all_windows.py
停止:  Ctrl + C
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

    if not os.getenv("QWEN_API_KEY"):
        print("[ERROR] 未配置 QWEN_API_KEY，请在 .env 中填入阿里云百炼 API Key。")
        sys.exit(1)

    busy = [p for p in (8001, 7860) if _port_in_use(p)]
    if busy:
        labels = {8001: "8001 (知识库管理 API)", 7860: "7860 (Agent WebUI)"}
        print("[ERROR] 端口已被占用:", "、".join(labels[p] for p in busy))
        print("        很可能已有一个 Safety_Agent 实例在运行。处理办法：")
        print("          1) 直接使用已运行的实例：浏览器打开 http://127.0.0.1:7860")
        print("          2) 或关闭旧的命令行窗口（Ctrl + C）后再重新运行本脚本")
        sys.exit(1)


    python = sys.executable
    env = os.environ.copy()

    procs: list[subprocess.Popen] = []

    def cleanup(*_args) -> None:
        print("\n[INFO] 正在关闭所有服务...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("[1/2] 启动数据入库与知识库管理 API (http://127.0.0.1:8001) ...")
    procs.append(
        subprocess.Popen(
            [python, "-m", "uvicorn", "run_ingestion_api:app", "--host", "127.0.0.1", "--port", "8001"],
            cwd=str(PROJECT_ROOT / "scripts"),
            env=env,
        )
    )

    print("[2/2] 启动 Agent WebUI (http://127.0.0.1:7860)，首次启动需下载 BGE/CLIP 模型，请耐心等待 ...")
    procs.append(
        subprocess.Popen(
            [python, str(PROJECT_ROOT / "scripts" / "run_agent_webui.py")],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    )

    print("=" * 60)
    print("  模型推理: 阿里云百炼 API (云端)")
    print("  入库服务: http://127.0.0.1:8001")
    print("  WebUI  : http://127.0.0.1:7860")
    print("  停止服务: Ctrl + C")
    print("=" * 60)

    try:
        while True:
            for p in procs:
                if p.poll() is not None:
                    cleanup()
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
