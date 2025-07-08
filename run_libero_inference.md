# OpenPI WSL环境配置实践指南

本文档基于在Ubuntu 22.04 WSL2环境中的实际配置经验，提供完整的openPI环境搭建流程。

## 🖥️ 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU (8GB+ VRAM推荐，如RTX 4080/4090)
- **内存**: 16GB+ RAM
- **存储**: 50GB+ 可用空间

### 软件环境
- **操作系统**: Ubuntu 22.04 LTS (WSL2)
- **Python**: 3.8 (LIBERO) + 3.11+ (主环境)
- **CUDA**: 12.x
- **Docker**: 27.5+

## 📋 完整配置流程

### 第一步：基础环境检查与准备

```bash
# 1. 检查GPU环境
nvidia-smi
# 确认显示GPU信息且CUDA版本为12.x

# 2. 检查WSL版本
wsl --version
# 确保使用WSL2

# 3. 克隆项目（含子模块）
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# 如果已克隆，初始化子模块
git submodule update --init --recursive
```

### 第二步：安装uv包管理器

```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 验证安装
uv --version
```

### 第三步：系统依赖安装

```bash
# 安装必要的系统依赖
sudo apt update
sudo apt install -y patchelf libegl1-mesa-dev build-essential

# 安装Docker
sudo apt install -y docker.io
sudo usermod -aG docker $USER
sudo systemctl start docker
sudo systemctl enable docker

# 安装docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 第四步：LIBERO专用环境配置 ⭐

> **关键要点**: 由于mujoco编译问题，必须使用Python 3.8独立环境

```bash
# 创建LIBERO专用Python 3.8环境
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

# 安装LIBERO依赖
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

# 安装openpi-client和LIBERO包
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

### 第五步：修复LIBERO路径配置 ⚠️

> **重要**: LIBERO默认路径配置不正确，需要手动修复

```bash
# 创建路径修复脚本
cat > fix_libero_paths.py << 'EOF'
import os
import yaml

config_dir = os.path.expanduser("~/.libero")
config_file = os.path.join(config_dir, "config.yaml")
libero_path = os.path.join(os.getcwd(), "third_party/libero/libero/libero")

new_config = {
    "benchmark_root": libero_path,
    "bddl_files": os.path.join(libero_path, "bddl_files"),
    "init_states": os.path.join(libero_path, "init_files"),
    "datasets": os.path.join(libero_path, "..", "datasets"),
    "assets": os.path.join(libero_path, "assets"),
}

os.makedirs(config_dir, exist_ok=True)
with open(config_file, "w") as f:
    yaml.dump(new_config, f)

print("✅ LIBERO路径配置已修复")
EOF

# 运行修复脚本
python fix_libero_paths.py
rm fix_libero_paths.py
```

### 第六步：环境验证测试

```bash
# 验证LIBERO环境
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl

python -c "
from libero.libero import benchmark
from openpi_client import websocket_client_policy
print('✅ LIBERO环境验证成功')
"
```

## 🚀 LIBERO Inference测试

### 方法一：快速演示测试

```bash
cd ~/projects/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl

# 运行单任务测试
cd examples/libero
python main.py --args.num-trials-per-task 1 --args.task-suite-name libero_spatial
```

### 方法二：完整Policy Server测试

**Terminal 1: 启动Policy Server**
```bash
cd ~/projects/openpi
# 注意：需要主环境配置完整的openpi依赖
uv run scripts/serve_policy.py --env LIBERO
```

**Terminal 2: 运行LIBERO测试**
```bash
cd ~/projects/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
cd examples/libero
python main.py --args.num-trials-per-task 5 --args.task-suite-name libero_spatial
```

### 方法三：Docker方案（可选）

```bash
# 安装NVIDIA Container Toolkit（一次性）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 运行完整Docker测试
export PWD=$(pwd)
export SERVER_ARGS="--env LIBERO"
export MUJOCO_GL=egl
sudo -E docker-compose -f examples/libero/compose.yml up --build
```

## ⚠️ 关键注意事项

### 1. 环境隔离策略
- **LIBERO**: 必须使用Python 3.8环境（`examples/libero/.venv`）
- **主openpi**: 需要Python 3.11+环境
- **原因**: mujoco编译依赖冲突，Python版本要求不同

### 2. 路径配置问题
- LIBERO的`~/.libero/config.yaml`默认路径不正确
- 必须在配置后运行路径修复脚本
- 验证所有路径都指向`third_party/libero`

### 3. Mujoco渲染设置
```bash
# 优先尝试EGL（无头渲染）
export MUJOCO_GL=egl

# 如果出现渲染错误，改用GLX
export MUJOCO_GL=glx
```

### 4. 环境变量设置
每次使用LIBERO时必须设置：
```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export MUJOCO_GL=egl
```

## 🔧 常见问题与解决方案

### 问题1: Mujoco编译失败
```
RuntimeError: MUJOCO_PATH environment variable is not set
```
**解决**: 使用Python 3.8独立环境，避免混合安装

### 问题2: LIBERO路径错误
```
AssertionError: [error] .../bddl_files/... does not exist!
```
**解决**: 运行路径修复脚本更新`~/.libero/config.yaml`

### 问题3: Docker GPU访问失败
```
Error response from daemon: could not select device driver "nvidia"
```
**解决**: 安装nvidia-container-toolkit并重启Docker

### 问题4: 依赖版本冲突
```
The requested interpreter resolved to Python X.X, which is incompatible
```
**解决**: 使用正确的Python版本创建虚拟环境

### 问题5: 权限问题
```
permission denied while trying to connect to the Docker daemon socket
```
**解决**: 
```bash
sudo usermod -aG docker $USER
# 重新登录或运行
newgrp docker
```

## 📊 性能预期

根据论文数据，π₀-FAST模型在LIBERO测试中的表现：
- **Libero Spatial**: ~96% 成功率
- **Libero Object**: ~97% 成功率  
- **Libero Goal**: ~89% 成功率
- **Libero 10**: ~60% 成功率

## 🎯 测试验证清单

配置完成后，应该能够：
- [ ] ✅ LIBERO环境启动无错误
- [ ] ✅ 图像处理正常（224×224×3）
- [ ] ✅ 状态提取正确（8维向量）
- [ ] ✅ 动作执行正常（7维动作空间）
- [ ] ✅ 视频保存功能工作
- [ ] ✅ Policy server连接成功
- [ ] ✅ 任务执行并记录结果

## 💡 最佳实践

1. **环境管理**: 为不同组件维护独立的Python环境
2. **路径管理**: 始终在项目根目录执行命令
3. **依赖锁定**: 使用requirements.txt锁定版本
4. **GPU监控**: 使用`nvidia-smi`监控GPU使用情况
5. **日志记录**: 保存配置和测试日志以便调试

## 🚀 下一步开发

环境配置完成后，可以进行：
- **模型微调**: 在自己的数据上fine-tune预训练模型
- **新任务测试**: 尝试不同的LIBERO任务套件
- **性能优化**: 调整推理参数和批处理大小
- **集成开发**: 将模型集成到自己的机器人系统中

---

*本文档基于Ubuntu 22.04 WSL2 + RTX 4080的实际配置经验整理，适用于类似环境的开发者参考。* 