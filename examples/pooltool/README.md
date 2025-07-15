# Pool Robot Simulation - 台球机械臂仿真

> **OpenPI x Pooltool**: 专业台球环境中的机械臂仿真，支持OpenPI策略推理

## 构建需求
- **专业台球物理**: 基于Pooltool的高精度台球仿真，整合[Pooltool](https://github.com/ekiefl/pooltool)专业台球物理引擎
-  **集成Mujuco的机器人仿真**: 参考LIBERO的实现(thrid_party/libero)
-  **高效率可视化**: 评估基于Pooltool实现可视化或者基于LIBERO实现可视化
-  **OpenPI兼容**: 参考LIBERO的实现(example/libero),完整的策略接口和数据处理管道，便于openpi模型接入
- 机械臂尽量复用LIBERO中的FRANKA机械臂
- 不要走捷径使用pybullet进行仿真和可视化，要尽量使用Mujuco和Pooltool的仿真和可视化

## Docker环境使用指南 🐳

### 构建镜像
```bash
# ⚠️ 必须从项目根目录构建，包含third_party/pooltool
cd /path/to/openpi
docker build . -t pooltool -f examples/pooltool/Dockerfile
```

### 运行仿真
```bash
# 🎯 Milestone 1: 台球3D仿真视频生成
docker run --rm -v "$PWD:/app" pooltool /bin/bash -c "source /.venv/bin/activate && python examples/pooltool/milestone1_3d_video_fixed.py"

# 🤖 Milestone 2: 机械臂与台球集成仿真
docker run --rm -v "$PWD:/app" pooltool /bin/bash -c "source /.venv/bin/activate && python examples/pooltool/milestone2_mujoco_integration.py"

# 📊 Milestone 2: 数据集成验证demo
docker run --rm -v "$PWD:/app" pooltool /bin/bash -c "source /.venv/bin/activate && python examples/pooltool/milestone2_data_demo.py"

# 🎬 Milestone 2: 机器人+台球环境3D可视化demo
docker run --rm -v "$PWD:/app" pooltool /bin/bash -c "source /.venv/bin/activate && python examples/pooltool/milestone2_simple_3d_demo.py"

# 🤖 Milestone 2: 明显机器人可视化demo (16个机器人部件)
docker run --rm -v "$PWD:/app" pooltool /bin/bash -c "source /.venv/bin/activate && python examples/pooltool/milestone2_robot_visible_demo.py"

# 🐚 交互式运行 (调试用)
docker run --rm -it -v "$PWD:/app" pooltool /bin/bash
# 进入容器后激活环境: source /.venv/bin/activate

# 📊 查看生成的数据和视频
docker run --rm -v "$PWD:/app" pooltool /bin/bash -c "ls -la examples/pooltool/data/"
```

### 关键环境配置
- **Python**: 3.10 (兼容panda3d和pooltool)
- **panda3d**: 1.11.0.dev3702 (从archive.panda3d.org安装)
- **环境变量**: `PYTHONPATH=/app:/app/third_party/pooltool`
- **渲染模式**: `PANDA3D_WINDOW_TYPE=offscreen` (Docker环境)

## 备注
- pooltool项目已经在third_party/pooltool 中下载完成
- 关于python环境冲突问题，参考LIBERO仿真的解决方案，仿真器与主环境隔离解耦
- 创建一个临时文档，每做一个TODO item就简要记录更新一下文档。因为这个任务很长，等到最后再写一整个文档的话，有可能context window不够


## Milestone
1. ✅ **不带机械臂的PoolTool跑通，给出台球仿真视频demo** - 完成
2. ✅ **集成PoolTool与Mujuco机器人仿真，完成数据集成验证** - 完成
3. 🔄 **尝试机械人在仿真环境与球接触，给出仿真视频demo，测试物理交互仿真功能** - 准备中