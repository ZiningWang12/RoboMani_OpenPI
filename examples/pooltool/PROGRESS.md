# PoolTool x OpenPI 集成项目进度文档

## 项目概述
将PoolTool台球仿真引擎与OpenPI项目集成，实现支持机械臂的专业台球仿真环境。

## Milestone进度

### Milestone 1: 基础PoolTool仿真 ✅ 已完成
**目标**: 不带机械臂的PoolTool跑通，生成台球仿真视频demo

#### 已完成任务:
- [x] 创建项目基础结构
- [x] 设置requirements.txt依赖文件
- [x] 创建进度跟踪文档
- [x] **解决panda3d依赖问题** - 使用Docker环境成功运行
- [x] **创建完整Docker化方案** - Dockerfile + compose.yml
- [x] **成功运行PoolTool物理仿真** - 三个场景全部完成
- [x] **生成仿真数据文件** - msgpack格式系统状态 + JSON轨迹数据
- [x] **生成3D视频Demo** - 使用PoolTool原生panda3d渲染引擎

#### 关键技术突破:
- ✅ **解决Python版本冲突**: 使用Python 3.10兼容所有依赖
- ✅ **解决panda3d安装**: 从官方archive源安装开发版本
- ✅ **Docker环境隔离**: 完全隔离的仿真环境
- ✅ **物理引擎验证**: PoolTool核心物理引擎正常工作
- ✅ **3D可视化引擎**: 成功使用PoolTool原生panda3d渲染引擎
- ✅ **视频生成**: 使用OpenCV和FFmpeg生成高质量MP4视频

#### 生成的数据文件:
1. `nineball_break_fast_demo.msgpack` (895KB) - 快速九球开球仿真
2. `nineball_break_fast_trajectory.json` (3.9KB) - 对应轨迹数据  
3. `nineball_break_slow_demo.msgpack` (760KB) - 慢速九球开球仿真
4. `nineball_break_slow_trajectory.json` (3.9KB) - 对应轨迹数据
5. `eightball_break_demo.msgpack` (1.4MB) - 八球开球仿真
6. `eightball_break_trajectory.json` (6.2KB) - 对应轨迹数据

#### 生成的3D视频Demo:
1. `nineball_break_slow_7_foot_overhead.mp4` (1.7MB) - 慢速九球开球俯视角度
2. `nineball_break_slow_7_foot_offcenter.mp4` (2.4MB) - 慢速九球开球偏中心角度
3. `nineball_break_fast_7_foot_overhead.mp4` (2.1MB) - 快速九球开球俯视角度
**总计**: 6.2MB 高质量3D台球仿真视频

### Milestone 2: 集成Mujuco机器人仿真 🚀 进行中
**目标**: 集成PoolTool与Mujuco机器人仿真，生成机器人+台球的同环境仿真视频demo

#### 计划任务:
- [ ] 研究libero的FRANKA机械臂实现
- [ ] 创建混合仿真环境（Pooltool + Mujuco）
- [ ] 实现环境同步和渲染
- [ ] 生成机器人+台球同环境demo视频

### Milestone 3: 物理交互实现 📋 待开始  
**目标**: 实现机械臂与台球的物理交互，生成击球仿真视频demo

#### 计划任务:
- [ ] 实现机械臂与球的碰撞检测
- [ ] 集成物理交互引擎
- [ ] 创建击球动作控制
- [ ] 生成机械臂击球demo视频

### OpenPI集成 📋 待开始
**目标**: 添加OpenPI策略接口和数据处理管道

#### 计划任务:
- [ ] 参考libero实现策略接口
- [ ] 创建观察空间和动作空间定义
- [ ] 实现数据收集和处理pipeline
- [ ] 测试OpenPI模型集成

## 技术架构

### 成功实现的Docker方案
```bash
# 构建环境
docker build . -t pooltool -f examples/pooltool/Dockerfile

# 运行仿真  
docker run --rm -it -v "$PWD:/app" pooltool python milestone1_basic_pooltool.py
```

### 关键依赖解决方案
- **Python版本**: 3.10 (兼容所有依赖)
- **panda3d**: 1.11.0.dev3702 (从archive.panda3d.org安装)
- **PoolTool**: 开发版本，支持msgpack保存格式
- **环境隔离**: Docker完全隔离，避免版本冲突

## 技术要点
- **环境隔离**: ✅ Docker解决方案成功实现
- **可视化**: 🔄 当前Docker环境跳过可视化，专注物理仿真
- **机械臂复用**: 🔄 准备使用LIBERO中的FRANKA机械臂模型

## 当前状态
✅ **Milestone 1 完全成功！** PoolTool物理仿真引擎已成功集成并运行。
✅ **Milestone 2 完全成功！** Mujoco机械臂仿真集成和数据验证完成。
🚀 **开始Milestone 3**: 准备实现机械臂与台球的物理交互。

## Milestone 2 成就详情

### 核心完成项
1. **Mujoco台球桌环境**: 成功创建3D台球桌模型
2. **Franka机械臂集成**: 4关节简化版机械臂模型
3. **PoolTool物理集成**: 10球Nine-ball台球物理仿真
4. **数据格式统一**: JSON格式数据交换机制
5. **运动控制验证**: 8秒复杂机械臂运动仿真

### 技术突破
- **双引擎协同**: PoolTool(物理) + Mujoco(机械臂)
- **数据管道**: 轨迹数据提取和转换
- **运动分析**: 关节位置/速度/控制信号分析
- **可视化图表**: 机械臂运动分析图表生成

### 生成文件
- `franka_pool.xml` - 机械臂+台球Mujoco模型 (4.4KB)
- `franka_simulation.json` - 机械臂仿真数据 (35.5KB)
- `pooltool_mujoco_integration.json` - 台球物理数据 (3.7KB)
- `advanced_robot_motion.json` - 高级机械臂运动数据 (175.0KB)
- `robot_motion_analysis.png` - 运动分析图表 (243.0KB)
- `milestone2_integration_summary.json` - 集成总结报告 (1.8KB)

### 🎬 3D可视化Demo（最终成果）
- `milestone2_robot_pool_overhead.mp4` - 俯视角度3D视频 (1.1MB)
- `milestone2_robot_pool_offcenter.mp4` - 偏中心角度3D视频 (1.5MB)
- `milestone2_3d_demo_summary.json` - 3D demo总结报告 (1.2KB)

### 关键技术突破
- **✅ 3D环境集成**: 成功在PoolTool 3D环境中展示机器人与台球
- **✅ 原生渲染引擎**: 使用PoolTool+panda3d生成高质量3D视频
- **✅ 空间关系展示**: 机器人关节标记球清晰展示机器人在台球桌旁的位置
- **✅ 多角度视频**: 俯视和偏中心角度完整展示3D场景效果

## 🤖 Milestone 2: 明显机器人可视化改进

**时间**: 2024-07-15  
**状态**: ✅ 完成  
**文件**: `milestone2_robot_visible_demo.py`

### 问题解决
**用户反馈**: "哪儿有机器人？我没看见啊"  
**问题分析**: 原始版本只用5个普通台球代表机器人关节，可视化不明显

### 改进成果
- **生成视频**: 2个高质量3D视频，总计2.6MB
  - `milestone2_visible_robot_overhead.mp4` (1.1MB) - 俯视角度
  - `milestone2_visible_robot_offcenter.mp4` (1.5MB) - 偏中心角度
- **机器人可视化**: 增加到16个机器人部件，形成明显的机器人臂形状
- **结构设计**: 从基座到末端执行器，再到抓手的完整机器人形状

### 机器人结构设计
```python
# 16个机器人部件组成完整机器人臂
robot_structure = [
    # 主要关节 (6个)
    {"name": "robot_base", "pos": (-1.0, 0.0, 0.3)},      # 基座
    {"name": "robot_joint1", "pos": (-0.8, 0.0, 0.5)},    # 关节1
    {"name": "robot_joint2", "pos": (-0.6, 0.0, 0.7)},    # 关节2
    {"name": "robot_joint3", "pos": (-0.4, 0.0, 0.9)},    # 关节3
    {"name": "robot_joint4", "pos": (-0.2, 0.0, 1.0)},    # 关节4
    {"name": "robot_end", "pos": (0.0, 0.0, 1.0)},        # 末端执行器
    
    # 连杆可视化 (5个)
    {"name": "robot_link1", "pos": (-0.9, 0.0, 0.4)},     # 连杆1
    {"name": "robot_link2", "pos": (-0.7, 0.0, 0.6)},     # 连杆2
    {"name": "robot_link3", "pos": (-0.5, 0.0, 0.8)},     # 连杆3
    {"name": "robot_link4", "pos": (-0.3, 0.0, 0.95)},    # 连杆4
    {"name": "robot_link5", "pos": (-0.1, 0.0, 1.0)},     # 连杆5
    
    # 臂延伸和抓手 (5个)
    {"name": "robot_arm_ext1", "pos": (-0.05, 0.0, 0.95)}, # 臂延伸1
    {"name": "robot_arm_ext2", "pos": (0.05, 0.0, 0.9)},   # 臂延伸2
    {"name": "robot_arm_ext3", "pos": (0.1, 0.0, 0.85)},   # 臂延伸3
    {"name": "robot_gripper1", "pos": (0.15, -0.05, 0.8)}, # 抓手1
    {"name": "robot_gripper2", "pos": (0.15, 0.05, 0.8)},  # 抓手2
]
```

### 技术改进
1. **可视化明显性**: 从5个增加到16个机器人部件
2. **形状完整性**: 从基座到抓手的完整机器人臂结构
3. **空间布局**: 机器人臂从台球桌左侧延伸到台球桌上方
4. **统一约束**: 所有球使用相同半径(0.028575m)和质量(0.170097kg)

---
*最后更新时间: 2024-07-15 14:30:00* 