# Franka Panda + Pooltool 协同仿真系统

## 🎯 系统概述

这是一个完整的台球机器人协同仿真系统，深度集成了：

- **Pooltool**: 世界领先的专业台球物理引擎
- **Franka Panda**: 7-DOF高精度机械臂仿真模型
- **PyBullet**: 机械臂渲染和控制环境

### ✨ 核心特性

1. **真实台球物理**: 使用pooltool专业物理引擎，支持：
   - 精确的球-球碰撞检测
   - 复杂的旋转效应和英式击球
   - 专业的台球桌缓冲和袋口物理
   - 高精度轨迹预测

2. **精确机械臂控制**: Franka Panda 7-DOF机械臂：
   - 真实URDF模型
   - 逆运动学求解
   - 平滑轨迹规划
   - 碰撞检测

3. **完整协同仿真**: 
   - 机械臂-台球物理交互
   - 多种击球策略
   - 实时状态监控
   - 性能分析系统

## 🚀 快速开始

### 1. 环境准备

确保已安装所有依赖：

```bash
# 安装核心依赖
uv add pybullet panda3d panda3d-simplepbr panda3d-gltf
uv add cattrs attrs msgpack-numpy matplotlib pillow
uv add numba llvmlite h5py pyyaml click rich msgpack
```

### 2. 运行演示

```bash
cd examples/pooltool

# 基础演示模式 (无GUI)
python run_franka_pooltool_demo.py --mode demo --shots 3 --no-gui

# GUI模式 + 视频录制
python run_franka_pooltool_demo.py --mode demo --shots 4 --record

# 交互模式
python run_franka_pooltool_demo.py --mode interactive

# 性能测试模式
python run_franka_pooltool_demo.py --mode benchmark --shots 10
```

### 3. 命令行选项

```bash
python run_franka_pooltool_demo.py --help
```

选项说明：
- `--mode`: 运行模式 (demo/interactive/benchmark)
- `--shots`: 演示击球次数 (默认: 4)
- `--record`: 启用视频录制
- `--no-gui`: 无GUI模式运行
- `--table-type`: 台球桌类型 (POCKET/SNOOKER/BILLIARD)

## 🎮 交互模式命令

进入交互模式后可使用以下命令：

```bash
# 执行击球 (速度 水平角度 俯仰角度)
shot 4.0 0.0 0.0      # 直线击球，速度4.0m/s
shot 5.5 0.524        # 角度击球，30度
shot 3.0 -0.262 0.05  # 复合击球，-15度+3度俯仰

# 重置环境
reset

# 切换摄像头视角
camera overview       # 全景视角
camera robot_view     # 机械臂视角  
camera table_view     # 台球桌视角
camera action_view    # 动作视角

# 退出
quit
```

## 🏗️ 系统架构

### 核心组件

1. **FrankaPooltoolIntegration**: 主控制器
   - 协调机械臂和台球物理引擎
   - 管理击球流程和状态
   - 处理碰撞检测和安全控制

2. **Pooltool物理引擎**: 台球仿真
   - 精确的物理计算
   - 专业台球规则
   - 高性能轨迹预测

3. **PyBullet机械臂**: 机械臂仿真
   - Franka Panda URDF模型
   - 实时渲染和控制
   - 逆运动学求解

### 击球流程

```
1. 机械臂接近球杆 → 2. 精确瞄准定位 → 3. 执行击球动作 
                   ↓
4. 机械臂撤回观察 ← 5. 分析击球结果 ← 6. Pooltool物理模拟
```

## 📊 性能指标

典型性能表现：

- **击球执行时间**: 8-12秒 (包含完整动作序列)
- **物理仿真精度**: 毫秒级时间步长
- **机械臂控制频率**: 240Hz
- **碰撞检测**: 实时无延迟

## 🎯 演示击球类型

系统支持多种专业击球技术：

1. **直线击球**: 基础正面击球
2. **角度击球**: 30度斜击，测试精度
3. **低速控制**: 精确力度控制
4. **高速冲击**: 高力度击球
5. **精确旋转**: 带旋转的复杂击球
6. **复合击球**: 水平+俯仰角度组合

## 📁 文件结构

```
examples/pooltool/
├── franka_pooltool_integration.py    # 核心协同仿真系统
├── run_franka_pooltool_demo.py       # 演示运行脚本
├── physics_bridge.py                 # 物理引擎桥接器
├── enhanced_demo.py                  # 增强演示脚本
├── data/
│   ├── pybullet-panda/              # Franka机械臂模型
│   └── pooltool_results/            # 仿真结果数据
└── videos/                          # 录制视频输出
```

## 🔧 技术细节

### Pooltool集成

- 使用官方pooltool API
- 支持完整的台球规则系统
- 高精度物理参数配置
- 实时碰撞和轨迹计算

### 机械臂控制

- 基于PyBullet物理引擎
- 真实Franka Panda URDF模型
- 平滑关节插值控制
- 安全限位和碰撞避免

### 协同机制

- 异步物理引擎协调
- 实时状态同步
- 错误恢复和重置
- 性能监控和优化

## 🎬 视频输出

启用 `--record` 选项时，系统会自动录制视频：

- 分辨率: 1024x768 @ 30fps
- 格式: MP4 (H.264编码)
- 保存位置: `videos/franka_pooltool_demo_*.mp4`

## 📈 未来扩展

计划中的功能扩展：

1. **AI击球策略**: 集成强化学习算法
2. **多机械臂协作**: 双臂协同击球
3. **实时策略调整**: 动态击球参数优化
4. **VR/AR可视化**: 沉浸式交互体验
5. **物理参数学习**: 自适应环境配置

## 🐛 故障排除

### 常见问题

1. **依赖缺失**: 确保安装所有required packages
2. **URDF加载失败**: 检查 `data/pybullet-panda/` 目录
3. **Pooltool初始化错误**: 验证third_party/pooltool子模块
4. **GUI显示问题**: 确保X11转发(WSL)或本地显示支持

### 日志信息

系统提供详细的状态日志：
- ✅ 成功操作
- ⚠️ 警告信息  
- ❌ 错误状态
- 🎯 击球信息
- 📊 性能数据

## 📄 许可证

本项目遵循Apache 2.0许可证，与openpi项目保持一致。

---

**开发团队**: OpenPI团队  
**版本**: 3.0.0 - 完整协同集成  
**更新日期**: 2025年1月  

🎱 **享受精确的台球机器人仿真体验！** 🤖 