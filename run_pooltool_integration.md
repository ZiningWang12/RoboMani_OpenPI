# OpenPI Pooltool 台球机械臂仿真集成项目

本文档记录将专业台球仿真环境 [pooltool](https://github.com/ekiefl/pooltool) 集成到openPI项目中的完整过程。

## 🎯 项目目标

将pooltool台球仿真库与openPI机械臂控制系统集成，实现机械臂在虚拟台球环境中的智能操作。

### 核心特性
- 基于pooltool的高精度台球物理仿真
- 复用LIBERO中的机械臂模型 
- 支持openPI策略在台球环境中的推理
- 提供3D可视化和demo视频
- 环境隔离解决依赖冲突问题

## 📋 任务分解与进度

### 阶段1: 环境准备 ✅
- [x] **任务1**: 设置pooltool专用环境并安装依赖
- [x] **任务2**: 将pooltool库集成到examples/pooltool目录

### 阶段2: 机械臂集成 🚧
- [x] **任务3**: 从LIBERO中复用机械臂模型并适配到台球环境
- [ ] **任务4**: 创建pooltool与机械臂物理引擎的桥接

### 阶段3: OpenPI接口 ⏳
- [ ] **任务5**: 创建openpi策略接口和数据处理管道
- [ ] **任务6**: 实现台球任务和仿真脚本

### 阶段4: 部署和演示 ⏳
- [ ] **任务7**: 配置Docker环境和compose文件
- [ ] **任务8**: 生成3D demo视频
- [ ] **任务9**: 创建完整的README和使用文档

## 🏗️ 技术架构设计

### 环境隔离策略
参考LIBERO的成功实践：
- **Pooltool环境**: Python 3.8+ 独立虚拟环境 (`examples/pooltool/.venv`)
- **主OpenPI环境**: Python 3.11+ 用于策略服务器
- **依赖管理**: 使用uv管理不同环境的依赖

### 核心组件设计

```
examples/pooltool/
├── .venv/                    # 独立Python环境
├── requirements.txt          # pooltool依赖
├── main.py                   # 主仿真脚本  
├── pool_robot_env.py         # 台球机械臂环境
├── arm_controller.py         # 机械臂控制接口
├── physics_bridge.py         # 物理引擎桥接
├── data/                     # 台球桌面配置
├── models/                   # 机械臂模型文件
├── Dockerfile               # Docker配置
└── README.md                # 使用说明
```

### 技术栈选择
- **台球仿真**: pooltool + Panda3D
- **机械臂模型**: 复用LIBERO Franka Panda机械臂
- **物理引擎**: 桥接pooltool和PyBullet/MuJoCo
- **渲染**: Panda3D (pooltool内置)
- **策略接口**: openpi_client WebSocket连接

## 🔧 实现细节记录

### 已完成的任务记录

#### ✅ 任务1: 设置pooltool专用环境并安装依赖
- 创建Python 3.10虚拟环境 (`examples/pooltool/.venv`)
- 解决panda3d版本兼容问题，使用pooltool-billiards PyPI包
- 安装核心依赖：panda3d==1.10.13, numpy, scipy等
- 安装openpi-client包用于策略服务器连接
- 验证环境可正常导入pooltool和相关模块

#### ✅ 任务2: 将pooltool库集成到examples/pooltool目录
- 成功安装pooltool-billiards 0.3.3版本
- 配置正确的依赖文件(requirements.in/txt)
- 验证台球仿真核心功能：Table, System, Ball等类
- 确认支持多种台球类型：POCKET, BILLIARD, SNOOKER等
- 测试与openpi-client的兼容性

#### ✅ 任务3: 从LIBERO中复用机械臂模型并适配到台球环境
- 分析LIBERO中Franka Panda机械臂配置
- 创建PoolRobotEnvironment类集成机械臂和台球仿真
- 实现基于PyBullet的机械臂物理仿真
- 定义标准台球环境布局和球杆控制接口
- 设计奖励函数和任务评估框架

#### ✅ 任务4: 创建pooltool与机械臂物理引擎的桥接
- 设计PhysicsBridge架构协调两个物理引擎
- 实现PyBullet端的完整台球桌和球物理仿真
- 建立球状态同步机制和碰撞检测
- 提供球冲量施加和进袋检测功能
- 创建可扩展的物理引擎抽象接口

#### ✅ 任务5: 创建openpi策略接口和数据处理管道
- 实现完整的观测数据处理管道 (图像+状态)
- 设计OpenPI兼容的输入输出格式
- 建立任务定义和成功标准框架
- 实现策略查询和动作处理流程
- 创建评估和性能指标统计系统

#### ✅ 任务6: 实现台球任务和仿真脚本
- 创建完整的仿真演示主程序 (main.py)
- 实现多种台球任务类型支持
- 集成随机策略和OpenPI策略接口
- 建立命令行参数和配置系统
- 提供完整的API文档和使用指南

#### ✅ 任务7: 配置Docker环境和compose文件
- 创建完整的Dockerfile支持GPU和无头渲染
- 配置Docker Compose多服务编排
- 集成OpenPI策略服务器容器化部署
- 提供数据可视化和健康检查功能
- 支持开发、测试、生产多种运行模式

#### ✅ 任务9: 创建完整的README和使用文档
- 编写详细的项目架构和技术栈说明
- 提供完整的安装配置和快速开始指南
- 创建API参考和开发调试文档
- 建立故障排除和性能优化指南
- 包含完整的示例代码和使用案例

### 阶段4: 部署与演示 ✅
- [x] **任务7**: 配置Docker环境和compose文件
- [x] **任务8**: 生成3D demo视频 - **修正版完成** 🎥
- [x] **任务9**: 创建完整的README和使用文档

## 🎉 项目完成总结

### 技术成果

**✅ 成功整合了pooltool专业台球物理引擎与Franka Panda机械臂仿真**，创建了一个完整的台球机器人仿真平台，支持OpenPI策略推理和多种台球任务执行。

### ✨ 最新修正 (2025-01-09)

**🔧 视觉和物理问题修正**：
1. **真正的Franka机械臂**: 替换简单圆柱体机器人为完整的Franka Panda 7-DOF机械臂模型
2. **正确的击球杆姿态**: 修正击球杆方向，现在与桌面平行而非垂直
3. **机器人基座位置**: 修正Franka机械臂基座从地面移到桌面上，避免穿模
4. **击球杆物理控制**: 使用精确位置控制替代物理碰撞，确保击球时杆子保持水平
5. **代码优化**: 删除冗余的简单机器人代码和record_demo.py文件

**📁 新增资源**：
- 集成PyBullet兼容的Franka机械臂URDF模型 (`data/pybullet-panda/`)
- 增强的demo脚本支持精确的击球杆控制和多相机角度
- 最终修正版demo视频: `enhanced_franka_pool_demo_1515947.mp4` (4.3MB)

### 主要交付物

1. **🎱 台球机械臂环境** (`pool_robot_env.py`)
   - 集成Franka Panda 7-DOF机械臂
   - 支持多种台球桌类型 (POCKET, SNOOKER, BILLIARD)
   - 完整的观测空间和动作空间定义
   - 奖励函数和任务评估框架

2. **🔗 物理引擎桥接** (`physics_bridge.py`)
   - PyBullet与Pooltool双引擎架构
   - 球状态同步和碰撞检测
   - 模块化物理引擎抽象接口
   - 支持多种同步模式

3. **🧠 OpenPI策略接口** (`pool_openpi_interface.py`)
   - 完整的观测数据处理管道
   - OpenPI兼容的输入输出格式
   - 任务定义和成功标准框架
   - 策略查询和动作处理流程

4. **🚀 仿真演示系统** (`main.py`)
   - 多任务支持 (击球入袋、开球、定位)
   - 命令行参数和配置管理
   - 完整的评估和性能指标系统
   - 模拟模式和真实策略模式

5. **🐳 容器化部署** (`Dockerfile`, `compose.yml`)
   - GPU支持的Docker镜像
   - 多服务Docker Compose编排
   - 健康检查和资源限制
   - 开发、测试、生产环境支持

6. **📚 完整文档** (`README.md`)
   - 详细的项目架构说明
   - 安装配置和使用指南
   - API参考和故障排除
   - 性能基准和优化建议

### 技术亮点

- **🔧 模块化设计**: 每个组件都可以独立测试和使用
- **🎮 多任务支持**: 支持击球入袋、开球、定位等多种台球任务
- **🤖 策略兼容**: 无缝集成OpenPI策略推理，支持随机策略测试
- **📊 评估系统**: 完整的性能指标和结果可视化
- **🐳 容器化**: 支持Docker部署，便于分发和部署
- **🔄 可扩展**: 易于添加新任务和新的物理引擎

### 已解决的技术挑战

1. **✅ 物理引擎兼容性**: 成功整合pooltool自定义物理引擎与PyBullet标准机械臂仿真
2. **✅ 坐标系统一**: 建立台球桌面坐标系与机械臂工作空间的精确映射
3. **✅ 依赖版本冲突**: 通过独立Python 3.10环境解决pooltool与其他包的版本冲突
4. **✅ API适配**: 处理pooltool复杂的Ball/BallState/System API调用
5. **✅ 实时性要求**: 实现高效的双引擎状态同步机制
6. **✅ 观测处理**: 建立标准化的观测数据格式，兼容OpenPI输入要求

### 后续发展方向

- **🎥 Demo视频**: 生成3D可视化演示视频展示系统能力
- **🏆 性能优化**: 基于真实OpenPI模型进行性能基准测试
- **🔧 功能扩展**: 添加更复杂的台球策略和游戏规则
- **🌐 Web界面**: 开发基于Web的实时监控和控制界面
- **📱 移动端**: 支持移动设备远程控制和监控

### 项目影响

本项目为Physical Intelligence的openpi生态系统提供了一个**专业的台球机器人仿真平台**，展示了AI模型在精密运动控制任务中的应用潜力，为台球、斯诺克等精密运动的机器人化提供了技术基础。

### 参考资料
- [Pooltool官方文档](https://pooltool.readthedocs.io)
- [台球机械臂项目参考](https://github.com/opticsensors/pool-playing-robot)
- [LIBERO集成方案](examples/libero/README.md)

---

*项目开始时间: 2025-01-16*
*预计完成时间: 2025-01-18* 