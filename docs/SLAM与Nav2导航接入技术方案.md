# SLAM + Nav2 接入 Safety_Agent 技术方案

> 目标：用「SLAM 建图 + Nav2 全局规划」替换现有「里程计 + 纯追踪」导航，根治航点漂移与避障不稳，为固定航点自动巡检提供厘米级、可复用的定位导航能力。
> 状态：方案设计稿，待评审后编码。所有控制参数与迁移成本需真机验证。

---

## 1. 背景与问题定义

### 1.1 现状（可验证事实）

现有导航链路为三层：

```
patrol.py（巡检编排）
   └─> navigator.py（WaypointNavigator，纯追踪闭环）
          └─> client.py（RobotClient，HTTP）
                 └─> robot_service.py（WSL 侧，SportClient + ObstaclesAvoidClient）
                        └─> unitree_sdk2py + CycloneDDS → Go2
```

定位数据来源（见 `robot/robot_service.py` 的 `_get_sport_state`）：

- 订阅 `rt/sportmodestate` 话题，取 `position`（世界坐标 x/y/z）与 `imu_state.rpy`（偏航 yaw）。
- 位置是**上电时初始化的相对世界坐标系**，属惯性/足端里程计性质。

### 1.2 现有方案的三个痛点（来自真机实测与代码）

1. **里程计漂移**：`position` 是相对上电原点的里程计融合量，长时间/长距离行走后累计漂移，导致航点判定偏差。现有代码用「点开始那一刻标定原点」缓解，但未根治漂移本身。
2. **避障不可靠**：`ObstaclesAvoidClient` 会因 LiDAR 误判/未就绪把前进速度压到接近 0，导致狗原地颤动、不直走（已用日志实证）。当前默认 `use_avoid=False`，等于放弃了动态避障。
3. **航点不可复用**：因为坐标系随每次标定变化，无法在「全局地图」里保存固定的巡检点位，每次都要重新标定。

### 1.3 目标

- 建立一张**全局占用栅格地图**，航点坐标在全局坐标系下**一次标定、长期复用**。
- 用 Nav2 的全局规划 + 局部避障，替代纯追踪的「原地转向 + 短距前进」，实现**平滑路径 + 动态避障**。
- 定位不再依赖易漂的里程计，而是用 **LiDAR + SLAM** 的全局定位。

---

## 2. 技术选型

### 2.1 总体架构

采用社区在 Go2 上验证最多的组合：

| 组件 | 选型 | 理由 |
|------|------|------|
| ROS 发行版 | ROS2 Humble（Ubuntu 22.04） | Go2 社区项目最常用，与 unitree_ros2 兼容 |
| SLAM | **slam_toolbox**（2D）或 **RTAB-Map**（3D） | Go2 自带 4D LiDAR 可直接用；RTAB-Map 也可融合相机 |
| 定位 | EKF（robot_localization）融合里程计 + IMU + LiDAR | 消除单一里程计漂移 |
| 规划 | Nav2（全局 A*/Dijkstra + 局部 DWB/MPPI） | ROS2 官方导航栈，成熟稳定 |
| 轨迹跟随 | Nav2 控制器（含 Pure-Pursuit 可选） | 与你现有控制律同源，迁移平滑 |
| 避障 | Nav2 局部代价地图 + 动态避障 | 替代不稳定的 ObstaclesAvoidClient |
| 与 Safety_Agent 衔接 | 新增一个 `Nav2Bridge` 服务 | 保持上层 patrol.py / navigator.py 接口不变 |

### 2.2 参考开源项目（可验证）

以下项目均已在 Go2 上验证 SLAM + Nav2：

| 项目 | 关键信息 | 参考价值 |
|------|---------|---------|
| `flo-77/go2_navigation_thesis` | ROS2 Jazzy + Nav2 + slam_toolbox + EKF + Gazebo，Docker 化，标注 "Verified Working" | 最完整的可复现工作流，含航点导航、动态避障 |
| `darshmenon/quadruped-robotics-stack` | 2D SLAM Toolbox + 3D RTAB-Map + Nav2，Go2 是唯一完全可用的机器人 | 提供 SLAM/Nav2 的 launch 文件范式 |
| `Sayantani-Bhattacharya/unitree_go2_nav` | 用 Go2 自带 4D LiDAR + RTAB-Map + Nav2，真机导航，115 stars | 直接对接 unitree_ros2 的真机方案 |
| `L-winder2002/Unitree-Go2-Mapping...` | Go2 + RealSense D435i + RTAB-Map，路径跟随正是 Pure-Pursuit | 印证你现有控制律可平滑迁移 |

> 关键结论：Go2 + SLAM + Nav2 是**成熟、有大量成功先例**的路线，且与你现有 Pure-Pursuit 控制律同源。

---

## 3. 总体架构改造

### 3.1 目标架构

```
┌────────────────────────────────────────────────────┐
│  Windows 侧 Safety_Agent（不改接口）                  │
│  patrol.py → navigator.py → client.py               │
└──────────────────────┬─────────────────────────────┘
                       │ HTTP（保持不变）
┌──────────────────────▼─────────────────────────────┐
│  WSL 侧 robot_service.py（保留，作为 HTTP 网关）      │
│  新增：/nav2_goal 端点（转发到 Nav2）                 │
└──────────────────────┬─────────────────────────────┘
                       │ 内部调用
┌──────────────────────▼─────────────────────────────┐
│  NEW: Nav2Bridge（ROS2 桥接层，WSL 侧）              │
│  订阅 Nav2 位姿/代价地图，发布导航目标，转发结果       │
└──────────────────────┬─────────────────────────────┘
                       │ ROS2 话题/服务
┌──────────────────────▼─────────────────────────────┐
│  ROS2 导航栈                                        │
│  unitree_ros2 → slam_toolbox/RTAB-Map → EKF → Nav2  │
└──────────────────────┬─────────────────────────────┘
                       │ DDS / 传感器
┌──────────────────────▼─────────────────────────────┐
│  Go2（4D LiDAR + IMU + 里程计 + 相机）               │
└────────────────────────────────────────────────────┘
```

### 3.2 关键设计原则：**接口不动，底层可换**

现有代码已经在 `robot_service.py` 的 docstring 里预留了这个设计意图：

> "这样换掉底层实现（比如以后上 Nav2 自主导航）不影响上层智能体。"

因此改造的核心是：**在 `robot_service.py` 与 ROS2 之间插入一个 `Nav2Bridge`，对上层暴露一个与现有 `/move`、`/pose` 语义一致的新端点 `/nav2_goal`**，让 `navigator.py` 可以无感切换到 Nav2 导航。

---

## 4. 分阶段实施计划

### 阶段一：环境与仿真验证（风险最低，先做）

**目标**：在 Gazebo 仿真里跑通 Go2 + SLAM + Nav2，验证技术可行性，不碰真机。

**做法**：
1. 参考 `flo-77/go2_navigation_thesis` 的 Docker 工作流，搭建 ROS2 + Gazebo + unitree_ros2 环境。
2. 跑通 `slam_toolbox` 建图 → `Nav2` 航点导航 → 动态避障。
3. 输出：一份可复现的环境配置 + 参数文件（`slam_params.yaml`、`nav2_params.yaml`）。

**产出物**：`nav2/slam_params.yaml`、`nav2/nav2_params.yaml`、`nav2/launch/` 目录。

### 阶段二：真机 SLAM 建图（核心价值）

**目标**：用 Go2 真机建一张巡检厂区的全局地图。

**做法**：
1. 在 Go2 机载电脑或 WSL 里运行 `unitree_ros2`（把 Go2 的 LiDAR/IMU/里程计发布成 ROS2 话题）。
2. 运行 `slam_toolbox` 在线建图，手动遥控狗走一遍厂区，保存地图（`.pgm` + `.yaml`）。
3. 验证回环检测是否正常、地图是否有漂移。

**关键决策点**（需真机确认）：
- Go2 的 4D LiDAR 是否发布标准 `/scan` 话题，还是需要转换成 2D 激光扫描。
- `sportmodestate` 的里程计是否可直接喂给 EKF，还是需要额外的 odometry 源。

### 阶段三：Nav2 航点导航接入

**目标**：用 Nav2 替代 `WaypointNavigator` 的纯追踪闭环。

**改动清单**（精确到文件/函数）：

| 文件 | 改动 | 说明 |
|------|------|------|
| `robot/robot_service.py` | 新增 `POST /nav2_goal {x, y, yaw}` 端点 | 转发给 Nav2Bridge，返回导航任务 id |
| `robot/robot_service.py` | 新增 `GET /nav2_status` | 查询导航进度/是否到达/是否失败 |
| 新增 `robot/nav2_bridge.py` | Nav2Bridge 类 | ROS2 桥接：订阅 `/amcl_pose`（或 `/map` 定位）、发布 `/goal_pose`、订阅 `/move_base` 结果 |
| `src/robot/client.py` | 新增 `nav2_goal()`、`nav2_status()` 方法 | 与现有 `move()`/`get_pose()` 并列，供 navigator 调用 |
| `src/robot/navigator.py` | 新增 `use_nav2` 配置项 | `True` 时走 Nav2，`False` 时回退现有纯追踪 |
| `configs/nav_waypoints.yaml` | 新增 `use_nav2: false` + 全局坐标系航点 | 航点从「相对标定原点」改为「全局地图坐标」 |

### 阶段四：巡检流程对接

**目标**：让 `navigate_waypoints_iter` 无缝切换到 Nav2 导航。

**做法**：保持 `patrol.py` 的快照协议不变（`{"status", "gallery", "photo", "answer", "report", "done"}`），仅替换 `WaypointNavigator.navigate_to()` 内部实现——从「读 `/pose` + 发 `/move`」改为「发 `/nav2_goal` + 轮询 `/nav2_status`」。

---

## 5. 关键改造点详解

### 5.1 定位数据源切换（核心）

**现状**：`navigator.py` 通过 `get_pose()` 拿 `sportmodestate` 的里程计位置。

**改造后**：Nav2 模式下，位置改从 **SLAM 定位**（AMCL 或 slam_toolbox 的 `/amcl_pose` / `/map→odom` TF）获取，这是**全局坐标系**下的位置，不漂移。

```python
# navigator.py 伪代码
if self.config.use_nav2:
    pose = self.client.nav2_pose()   # 全局坐标，来自 SLAM 定位
else:
    pose = self.client.get_pose()     # 里程计坐标，现有逻辑
```

### 5.2 航点坐标系变更（重要）

**现状**：航点是「相对点开始那一刻」的偏移（`_calibrate()` 标定原点）。

**改造后**：Nav2 模式下，航点是**全局地图坐标系**下的绝对坐标，一次标定长期复用。

```yaml
# configs/nav_waypoints.yaml（Nav2 模式）
use_nav2: true
map_frame: "map"          # 全局坐标系
waypoints:
  - name: "A区入口"
    x: 2.3      # 全局地图坐标（米），非相对偏移
    y: 1.8
    yaw: 1.57   # 到达朝向（可选）
```

### 5.3 避障替换

**现状**：`ObstaclesAvoidClient` 不可靠（已实测），默认关闭。

**改造后**：Nav2 的**局部代价地图 + DWB 控制器**天然具备动态避障，且是基于 LiDAR 实时数据，比 Go2 内置避障更可控。

### 5.4 回退机制（保底）

必须保留 `use_nav2: false` 的回退路径——当 ROS2/Nav2 未启动或失败时，自动回退到现有纯追踪导航，保证系统不因新方案而不可用。

---

## 6. 风险与未知项

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| Go2 的 4D LiDAR 数据格式是否可直接用于 slam_toolbox | 高 | 阶段一仿真先验证；真机检查 `/scan` 话题，必要时写 3D→2D 转换节点 |
| `sportmodestate` 里程计质量是否够 EKF 融合 | 高 | 阶段二真机实测；必要时用 LiDAR 里程计（如 Fast-LIO）替代 |
| ROS2 环境搭建复杂度（WSL 下 ROS2 与 DDS 共存） | 中 | 参考 Docker 化方案隔离环境 |
| 建图时四足机器人步态导致的 LiDAR 抖动 | 中 | 使用低速行走建图；slam_toolbox 调低运动阈值 |
| Nav2 参数（代价地图膨胀半径等）需真机调优 | 中 | 阶段三逐项标定 |
| 与现有 HTTP 架构的衔接复杂度 | 低 | 新增 Nav2Bridge 独立模块，接口隔离 |

---

## 7. 工作量评估（推测）

以下为经验估计，实际以真机验证为准：

| 阶段 | 预估工作量 | 主要风险 |
|------|-----------|---------|
| 阶段一：仿真验证 | 2-3 天 | 环境搭建、依赖冲突 |
| 阶段二：真机建图 | 2-4 天 | LiDAR 数据格式、里程计质量 |
| 阶段三：Nav2 接入 | 3-5 天 | 参数调优、桥接稳定性 |
| 阶段四：巡检对接 | 1-2 天 | 接口切换、回归测试 |

---

## 8. 事实与推测的区分

**可验证事实**：
- 现有代码已预留「换底层实现」的设计意图（`robot_service.py` docstring 明确写着"以后上 Nav2 自主导航"）。
- 现有定位用 `sportmodestate` 的里程计 position，是相对上电原点、会漂移（代码注释 + 真机实测）。
- `ObstaclesAvoidClient` 会导致狗原地颤动（已用日志实证）。
- 至少 4 个开源项目已在 Go2 上跑通 SLAM + Nav2（各项目 README 明确标注）。
- 现有 `navigator.py` 用的是 Pure-Pursuit 控制律，与 Nav2 可选控制器同源。

**个人推测 / 需真机验证**：
- Go2 的 4D LiDAR 能否直接输出 2D `/scan` 需真机确认。
- `sportmodestate` 里程计质量是否满足 EKF 融合需真机实测。
- 各阶段工作量估计为经验值，实际受硬件与 ROS 环境复杂度影响。
- Nav2 参数（膨胀半径、控制器增益）需真机逐项标定。

---

## 9. 建议的下一步

**最小可行验证（MVP）**：先做阶段一 + 阶段二的一部分——在仿真里跑通 Go2 + SLAM + Nav2 航点导航，并真机确认 Go2 的 LiDAR 数据格式。这两点一旦确认，整个方案的技术风险就基本解除。

建议按以下顺序推进：
1. 我先帮你搭一份 `flo-77/go2_navigation_thesis` 或 `unitree_go2_nav` 的源码拆解，明确它如何对接 `unitree_ros2`。
2. 确认你的 Go2 固件版本是否支持将 LiDAR 发布为 ROS2 `/scan` 话题。
3. 在 WSL 里验证 ROS2 + unitree_ros2 能否与现有 `robot_service.py` 的 DDS 通道共存。
