# Safety_Agent 安全生产管理智能体

基于多模态大模型（Qwen-VL）的安全生产管理智能体，支持图文合规性问答、现场隐患诊断，并可驱动宇树 Go2 机器狗执行固定航点自动巡检，最终生成隐患排查建议报告。

## 核心能力

- **多模态隐患诊断**：利用 Qwen-VL 对安全生产现场图片进行合规性审核，输出隐患定性、法律依据、整改措施与参考案例。
- **混合检索（RAG）**：结合 CLIP 图像检索与 ChromaDB 法规/案例检索，为诊断提供可溯源的法条与相似案例。
- **Go2 机器狗巡检**：通过 `unitree_sdk2py` 控制宇树 Go2 机器狗，支持速度控制、原地转向、拍照、急停、位置反馈，以及闭环固定航点导航。
- **固定航点导航**：按预设航点序列自主移动到巡检点，到位后自动拍照并触发隐患诊断，形成「走位 → 拍摄 → 诊断 → 报告」闭环。
- **语音交互（可选）**：支持语音识别（ASR）与语音合成（TTS），可语音指挥机器狗。
- **隐患报告生成**：自动生成含现场照片的 Word 格式隐患排查建议报告。

## 目录结构

```text
├── configs/                # 训练配置 + 固定航点导航配置（nav_waypoints.yaml）
├── data/                   # 数据目录（Git 仅保留结构，实际数据已忽略）
├── models/                 # 本地模型目录（已忽略）
├── prompts/                # 系统角色与 Prompt 模板
├── robot/                  # 机器狗控制服务（robot_service.py，WSL 侧运行）
├── scripts/                # 工具脚本（预处理、训练、检索、运行）
├── src/                    # 核心代码
│   ├── agent/              # 智能体工作流与推理
│   ├── data_processor/     # PDF/Word 提取与清洗
│   ├── evaluation/         # 模型评估
│   ├── retriever/          # CLIP 与 ChromaDB 检索引擎
│   ├── robot/              # 机器狗客户端、导航器、巡检编排、WebUI
│   └── voice/              # 语音识别与合成
├── docs/                   # 设计与使用文档
└── web-service-app/        # Web 服务（TypeScript）
```

## 快速上手

### 1. 环境准备

```bash
# Python 3.10+
pip install -r requirements.txt

# 语音功能（可选）
pip install -r requirements-voice.txt
```

### 2. 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入你的阿里云百炼 API Key（QWEN_API_KEY）
```

### 3. 启动 WebUI（Windows 一键启动）

```bash
# Windows 直接双击，或在命令行执行
启动智能体.bat
```

首次运行会自动创建虚拟环境并安装依赖。启动后访问 <http://127.0.0.1:7860>。

### 4. 启动机器狗巡检

机器狗控制服务需在 WSL 侧运行，详见 [docs/机器狗启动步骤.md](docs/机器狗启动步骤.md)，核心三步：

```bash
# 1. 检查服务是否已在跑
curl.exe http://127.0.0.1:8010/health

# 2. 在 WSL 中启动控狗服务
wsl -d Ubuntu-22.04 -u zzc
cd /mnt/d/Safety_Agent/robot && python3 robot_service.py --nic auto --port 8010

# 3. 浏览器打开 WebUI 的「机器狗巡检」面板，刷新连接状态即可
```

## 固定航点导航

航点配置在 [configs/nav_waypoints.yaml](configs/nav_waypoints.yaml)。坐标系为「点开始导航那一刻」狗位姿的相对偏移：原点 = 开始导航时狗的位置，x 轴正方向 = 开始导航时狗前进方向。

```yaml
arrival_radius: 0.2            # 到达判定距离（米）
heading_threshold_deg: 18.0    # 朝向对准死区（度）
turn_deg_per_sec: 55.0         # 实测原地转角速度
v_max: 0.45                    # 前进线速度上限（米/秒）
use_avoid: false               # 是否启用避障移动（默认关闭）
waypoints:
  - name: "直行1米测试点"
    x: 1.0
    y: 0.0
```

导航采用纯追踪（Pure Pursuit）控制律闭环：读取位置反馈 → 朝向纠偏 → 前进 → 到达判定。设计细节与参数说明见 [docs/固定航点导航详细设计.md](docs/固定航点导航详细设计.md)。

> 注意：所有控制参数为经验初值，必须真机标定后才能用于正式巡检。

## 数据处理与模型训练

```bash
# 数据预处理
python scripts/run_data_preprocess.py
# 构建法规向量库
python scripts/add_laws_to_db.py
# LoRA 微调（可选）
bash scripts/train_qwen3vl_lora.sh
```

## 技术栈

- **多模态模型**：Qwen-VL 系列（Qwen2-VL / Qwen3-VL），云端百炼 API 调用
- **向量数据库**：ChromaDB
- **检索引擎**：CLIP（图像）、BGE/sentence-transformers（文本）
- **机器狗**：unitree_sdk2py + CycloneDDS（宇树 Go2）
- **前端**：Gradio（WebUI）、TypeScript（Web App）
- **报告**：python-docx

## 许可证

本项目遵循 MIT 开源协议。
