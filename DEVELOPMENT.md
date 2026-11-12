# 安全生产管理智能体 (Safety_Agent) 开发文档

本项目是一个基于 **Qwen3-VL** 多模态大模型与 **RAG (检索增强生成)** 技术构建的智能化安全生产管理系统。它通过结合视觉识别、相似案例检索以及法律法规检索，为安全生产现场的隐患排查提供专业、合规的决策支持；并可驱动宇树 Go2 机器狗执行固定航点自动巡检，生成隐患排查建议报告。

---

## 🏗️ 1. 系统架构与技术栈

### 核心技术栈
- **多模态模型**: Qwen-VL 系列 (Qwen2-VL / Qwen3-VL)，通过阿里云百炼 API 调用
- **纯文本模型**: Qwen-Plus（用于自然语言指令解析等文本任务，与多模态模型分离）
- **微调框架**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（LoRA 微调与模型合并）
- **向量数据库**: [ChromaDB](https://www.trychroma.com/)（存储法律法规与案例特征）
- **检索模型**: CLIP（图像检索）、BGE/sentence-transformers（文本向量化）
- **机器狗**: [unitree_sdk2py](https://github.com/unitreerobotics/unitree_sdk2py) + CycloneDDS（宇树 Go2）
- **前端交互**: Gradio（WebUI）、TypeScript（Web App）
- **报告生成**: python-docx

### 模块结构
- `src/agent/`: 智能体核心逻辑，工作流调度、意图识别与 CoT（思维链）推理。
- `src/retriever/`: 混合检索引擎，图像相似度匹配与法律条文/案例检索。
- `src/data_processor/`: PDF/Word 数据提取与清洗。
- `src/robot/`: 机器狗客户端（client.py）、闭环导航器（navigator.py）、巡检编排（patrol.py）、WebUI（ui.py）。
- `src/voice/`: 语音识别（ASR）与语音合成（TTS）。
- `robot/`: 机器狗控制服务（robot_service.py，WSL 侧运行，HTTP 接口）。
- `configs/`: 训练配置 + 固定航点导航配置（nav_waypoints.yaml）。
- `scripts/`: 全流程操作脚本（预处理 → 训练 → 检索 → 运行）。

---

## 🛠️ 2. 环境搭建

### 2.1 基础环境
- **Python**: 3.10+
- **OS**: Linux（推荐）/ Windows WSL2（机器狗控制服务需在 WSL 侧运行）

### 2.2 安装依赖
```bash
# 建议创建虚拟环境
conda create -n safety_agent python=3.10
conda activate safety_agent

# 安装核心依赖
pip install -r requirements.txt

# 语音功能（可选）
pip install -r requirements-voice.txt
```

### 2.3 配置
```bash
cp .env.example .env
# 编辑 .env，至少填入 QWEN_API_KEY（阿里云百炼 API Key）
```

---

## 📂 3. 数据流程 (Data Pipeline)

### 3.1 原始数据预处理
系统支持从 PDF 规范、Word 文档中提取结构化知识。
```bash
python scripts/run_data_preprocess.py
```

### 3.2 向量库构建
将解析后的数据注入向量数据库。
```bash
python scripts/add_laws_to_db.py
```

---

## ⚡ 4. 模型训练 (LoRA Fine-tuning)

本项目采用 LoRA 技术在特定安全场景下提升模型表现。

```bash
# 启动训练任务
bash scripts/train_qwen3vl_lora.sh
# 模型合并导出
bash scripts/export_lora_model.sh
```

配置文件位于 `configs/llamafactory/`。

---

## 🤖 5. 机器狗巡检模块

### 5.1 分层架构

```
┌──────────────────────────────────────────────┐
│ 巡检编排层 patrol.py（航点即巡检点）            │
│  navigate_waypoints_iter / execute_plan_iter  │
└───────────────────┬──────────────────────────┘
                    │ 调 move / capture / pose
┌───────────────────▼──────────────────────────┐
│ 导航层 navigator.py（纯追踪闭环）              │
│  WaypointNavigator / NavConfig / NavResult    │
└───────────────────┬──────────────────────────┘
                    │ 通过 client.py（HTTP）
┌───────────────────▼──────────────────────────┐
│ 执行层 robot/robot_service.py（WSL 侧）        │
│  SportClient / ObstaclesAvoidClient           │
└───────────────────┬──────────────────────────┘
                    │ unitree_sdk2py + CycloneDDS
┌───────────────────▼──────────────────────────┐
│ 宇树 Go2 机器狗（内置 4D LiDAR）               │
└──────────────────────────────────────────────┘
```

### 5.2 执行层 HTTP 接口（robot_service.py）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/move` | POST | 连续速度控制 `{vx, vy, vyaw, duration}`（内部 10Hz 重发） |
| `/avoid_move` | POST | 避障移动（ObstaclesAvoidClient） |
| `/capture` | POST | 抓一帧落盘，返回路径 |
| `/estop` | POST | 急停（Damp） |
| `/pose` | GET | 返回世界坐标位置与偏航（订阅 `rt/sportmodestate`） |

### 5.3 客户端（src/robot/client.py）

`RobotClient` 封装上述 HTTP 接口，核心方法：`health()`、`move()`、`avoid_move()`、`estop()`、`capture()`、`get_pose()`。

### 5.4 固定航点导航（src/robot/navigator.py）

`WaypointNavigator` 采用纯追踪（Pure Pursuit）控制律实现闭环导航：

1. `_calibrate()`：把「点开始导航那一刻」的狗位姿标定为任务原点。
2. `_local()`：把航点坐标做局部坐标变换（相对任务原点）。
3. `navigate_to()`：循环读取 `/pose` → 计算朝向误差 → 朝向纠偏 → 前进 → 到达判定。
4. 超时保护：单航点超过 `max_steps` / `max_time_s` 标记失败，避免死循环。

航点配置见 [configs/nav_waypoints.yaml](../configs/nav_waypoints.yaml)。关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| arrival_radius | 0.2 m | 到达判定距离阈值 |
| heading_threshold_deg | 18° | 朝向对准死区 |
| turn_deg_per_sec | 55°/s | 实测原地转角速度 |
| v_max | 0.45 m/s | 前进线速度上限 |
| use_avoid | false | 是否用避障移动（默认关闭，见下方说明） |

> 关于 `use_avoid`：实测避障服务（ObstaclesAvoidClient）会因 LiDAR 误判/未就绪把前进速度压到接近 0，导致狗原地颤动、不直走。固定航点本就是预先规划的安全点位，默认走普通 SportClient.Move（前进可靠）。

### 5.5 巡检编排（src/robot/patrol.py）

- `navigate_waypoints_iter()`：航点导航 + 到位拍照 + 统一诊断，生成巡检报告。
- `execute_plan_iter()`：自然语言指令 → 动作计划 → 执行 → 拍照 + 流式诊断。
- `plan_with_llm()`：用纯文本模型（Qwen-Plus）解析自然语言指令为动作计划。
- `save_report()` / `_build_docx_report()`：生成含现场照片的 Word 隐患报告。

---

## 🎤 6. 语音交互模块（src/voice/）

- `asr.py`：语音识别，支持 bailian（云端 qwen3-asr-flash）/ funasr / whisper / mock。
- `tts.py`：语音合成，支持 bailian / edge-tts / pyttsx3 / mock。
- `pipeline.py`：语音对话编排。
- `robot_dog.py`：语音控制机器狗。

---

## 🧠 7. 核心逻辑分析：SafetyProductionAgent

`SafetyProductionAgent` 是系统的大脑，其工作流逻辑如下：

1. **多路输入**: 接收「现场图像」+「用户提问」。
2. **意图路由**: 采用「LLM 优先 + 规则兜底」的三分类策略，判断为 `greeting` / `legal_only` / `hazard_analysis`。
3. **混合检索 (Hybrid RAG)**:
   - **CLIP Retrieval**: 基于图片特征在历史案例库中寻找相似违规照片。
   - **Chroma Retrieval**: 基于文本语义检索相关法律条文与案例。
4. **思维链推理 (CoT)**: 视觉特征提取 → 法规契合点分析 → 风险等级判定 → 整改措施建议。
5. **JSON 结构化输出**: 强制模型输出预定义格式，便于下游系统集成。

### 7.1 意图识别实现位置
- 核心文件：`src/agent/workflow.py`
- Web 入口：`scripts/run_agent_webui.py`

### 7.2 模型分离策略
- 多模态诊断用 `QWEN_MODEL_NAME`（Qwen3-VL-Plus）。
- 自然语言指令解析用 `QWEN_TEXT_MODEL_NAME`（Qwen-Plus），指令遵循强于多模态模型。

### 7.3 结构化输出归一化
LLM 输出格式不稳定是常态。`_normalize_structured()` 对「参考案例 / 法律依据 / 整改措施 / 思维链 / 引用校验」等数组字段做类型归一化（字符串 → 列表），并对混入的未闭合 JSON 片段做截断清洗，避免下游渲染逐字符崩坏。

---

## 📅 8. 维护与贡献
- **数据更新**: 法律法规更新后，重新运行 `scripts/add_laws_to_db.py`。
- **模型演进**: 将用户反馈的错误案例加入 `data/qwen3vl_sft.jsonl` 进行二阶段微调。
- **导航调参**: 航点控制参数变更后需真机标定（详见 `configs/nav_waypoints.yaml` 注释）。
