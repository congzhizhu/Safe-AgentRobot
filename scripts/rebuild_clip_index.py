#!/usr/bin/env python3
"""重建 CLIP 图片索引（Windows 版）"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever.clip_engine import CLIPCaseEngine

engine = CLIPCaseEngine(project_root=PROJECT_ROOT)
count = engine.index_images()
print(f"CLIP 索引完成，共索引 {count} 张图片。")
