# 贡献指南

感谢你对 Safety_Agent 的关注。本项目欢迎 bug 修复、文档改进与功能完善。在提交贡献前，请阅读以下约定。

## 环境准备

1. Fork 本仓库并克隆到本地。
2. 按 [DEVELOPMENT.md](DEVELOPMENT.md) 搭建开发环境：
   ```bash
   pip install -r requirements.txt
   cp .env.example .env   # 填入你的阿里云百炼 API Key
   ```

## 提交规范

- **分支**：从 `main` 切出功能分支，命名如 `feat/xxx`、`fix/xxx`、`docs/xxx`。
- **提交信息**：使用简洁的中文描述，说明「做了什么」。
- **敏感信息**：严禁提交 `.env`、`.env.bak` 或任何包含真实 API Key 的内容。`.env` 模板请用 `.env.example` 维护。
- **大数据**：`data/`、`models/` 等目录已被 `.gitignore` 忽略，请勿强制提交。

## 代码约定

- 机器狗控制相关改动（`robot/`、`src/robot/`）需在真机或 mock 模式下验证闭环后再提交。
- 导航参数（`configs/nav_waypoints.yaml`）为经验初值，改动时请在注释中说明理由与验证结果。
- 涉及 LLM 结构化输出的解析，务必在 `src/agent/workflow.py` 的归一化逻辑内做类型兜底，避免渲染层逐字符崩坏。

## 提交前自检

```bash
# 确认没有敏感文件被暂存
git diff --cached --name-only | grep -E "\.env|\.bak|models/|node_modules" && echo "有泄漏，先移除"

# 语法检查
python -m py_compile src/robot/*.py src/agent/*.py
```

## 报告问题

提交 Issue 时请包含：环境信息（Python 版本、OS）、复现步骤、期望与实际行为、相关日志或截图。

---

本项目遵循 MIT 开源协议，详见 [LICENSE](LICENSE)。
