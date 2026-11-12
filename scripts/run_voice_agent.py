# -*- coding: utf-8 -*-
"""Safety_Agent 语音交互入口（面向机器狗集成）。

典型用法：

    # 1) 交互式：说一句，智能体边思考边回答并朗读
    .venv\\Scripts\\python.exe scripts\\run_voice_agent.py

    # 2) 持续循环 + 唤醒词（先说"你好小狗"，再说问题）
    .venv\\Scripts\\python.exe scripts\\run_voice_agent.py --loop --wake

    # 3) 不接麦克风，直接用文本测试（CI / 调试用）
    .venv\\Scripts\\python.exe scripts\\run_voice_agent.py --text "有限空间作业气体检测标准？"

    # 4) 带一张机器狗摄像头抓拍的现场图
    .venv\\Scripts\\python.exe scripts\\run_voice_agent.py --image data\\data_processed\\images\\xxx.jpg
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_dotenv(path: Path) -> None:
    """最小化 .env 加载：不引入 python-dotenv 依赖。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety_Agent 语音交互（机器狗）")
    parser.add_argument("--text", type=str, default=None,
                        help="直接以文本提问，跳过录音与语音识别（调试/自测用）")
    parser.add_argument("--image", type=Path, default=None,
                        help="可选：现场图片路径（机器狗摄像头抓拍）")
    parser.add_argument("--audio", type=Path, default=None,
                        help="可选：用已有音频文件代替麦克风输入")
    parser.add_argument("--loop", action="store_true", help="持续对话循环")
    parser.add_argument("--wake", action="store_true",
                        help="持续循环时要求先说唤醒词（默认取 .env 的 WAKE_WORDS）")
    parser.add_argument("--seconds", type=float, default=None,
                        help="固定录音秒数；不传则按静音自动结束")
    parser.add_argument("--no-tts", action="store_true", help="关闭语音朗读，只输出文本")
    parser.add_argument("--asr-provider", type=str, default=None,
                        choices=["bailian", "funasr", "whisper", "mock"], help="覆盖 ASR_PROVIDER")
    parser.add_argument("--tts-provider", type=str, default=None,
                        choices=["bailian", "edge", "pyttsx3", "mock"], help="覆盖 TTS_PROVIDER")
    parser.add_argument("--tts-voice", type=str, default=None, help="覆盖 TTS_VOICE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache" / "huggingface")

    if args.asr_provider:
        os.environ["ASR_PROVIDER"] = args.asr_provider
    if args.tts_provider:
        os.environ["TTS_PROVIDER"] = args.tts_provider
    if args.tts_voice:
        os.environ["TTS_VOICE"] = args.tts_voice

    from agent.workflow import build_agent_from_env
    from voice.config import VoiceConfig
    from voice.pipeline import VoicePipeline
    from voice.robot_dog import ConsoleRobotDog, HttpRobotDog

    print("正在初始化智能体...")
    agent = build_agent_from_env()
    cfg = VoiceConfig.from_env()

    robot = ConsoleRobotDog()
    if cfg.robot_http_base:
        robot = HttpRobotDog(cfg.robot_http_base, token=cfg.robot_http_token)

    pipeline = VoicePipeline(agent=agent, config=cfg, robot=robot,
                             auto_tts=False if args.no_tts else None)

    print(f"ASR: {cfg.asr_provider} ({cfg.asr_model})   TTS: {cfg.tts_provider} ({cfg.tts_voice})")
    print("准备好了，请说话（Ctrl+C 退出）。")

    try:
        if args.loop:
            pipeline.loop(wake_word=args.wake)
        else:
            audio_bytes = None
            if args.audio:
                audio_bytes = args.audio.read_bytes()
            pipeline.run_turn(audio=audio_bytes, text=args.text,
                              image_path=args.image, listen_seconds=args.seconds)
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        if hasattr(robot, "close"):
            try:
                robot.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
