# -*- coding: utf-8 -*-
"""语音交互模块（可集成到机器狗）。

组成：
    config   —— 语音相关配置（全部可用环境变量覆盖）
    asr      —— 语音识别（语音 -> 文本）
    tts      —— 语音合成（文本 -> 语音）
    audio_io —— 麦克风录音 / 扬声器播放 / 流式断句
    pipeline —— 把「录音 -> ASR -> 智能体流式推理 -> TTS 朗读」串起来
    robot_dog—— 机器狗动作适配层（LED / 头部 / 耳朵等状态反馈）

设计目标：
    1. 与现有 Agent 解耦：不改动 ``src/agent/workflow.py`` 的推理语义，
       只在其之上叠加语音输入输出；
    2. 可离线：TTS 可选 edge / pyttsx3，ASR 可选 funasr / whisper，
       机器狗在无网环境下也能跑；
    3. 低延迟：模型边生成边按句合成播放，而不是等整段说完。
"""
from .config import VoiceConfig, load_voice_config
from .asr import SpeechRecognizer, transcribe_file
from .tts import SpeechSynthesizer
from .audio_io import record_seconds, record_until_silence, play_audio, SentenceSplitter
from .pipeline import VoicePipeline
from .robot_dog import RobotDog, ConsoleRobotDog, HttpRobotDog

__all__ = [
    "VoiceConfig",
    "load_voice_config",
    "SpeechRecognizer",
    "transcribe_file",
    "SpeechSynthesizer",
    "record_seconds",
    "record_until_silence",
    "play_audio",
    "SentenceSplitter",
    "VoicePipeline",
    "RobotDog",
    "ConsoleRobotDog",
    "HttpRobotDog",
]
