#!/bin/bash
# PoolTool Milestone 1 运行脚本
# 解决panda3d依赖问题，使用Docker环境

set -e

echo "=== PoolTool x OpenPI Milestone 1 运行脚本 ==="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "项目根目录: $PROJECT_ROOT"
echo "PoolTool示例目录: $SCRIPT_DIR"

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请启动Docker"
    exit 1
fi

# 检查是否有NVIDIA GPU支持
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "✅ 检测到NVIDIA GPU"
    export GPU_SUPPORT="--gpus all"
else
    echo "⚠️  未检测到NVIDIA GPU，使用CPU模式"
    export GPU_SUPPORT=""
fi

# 设置环境变量
export DISPLAY=${DISPLAY:-:0}
export PANDA3D_WINDOW_TYPE=${PANDA3D_WINDOW_TYPE:-offscreen}

echo "环境变量:"
echo "  DISPLAY=$DISPLAY"
echo "  PANDA3D_WINDOW_TYPE=$PANDA3D_WINDOW_TYPE"

# 进入项目根目录
cd "$PROJECT_ROOT"

echo ""
echo "=== 构建Docker镜像 ==="
docker build . -t pooltool -f examples/pooltool/Dockerfile

echo ""
echo "=== 运行PoolTool仿真 ==="

# 创建输出目录
mkdir -p "$PROJECT_ROOT/data/pooltool/milestone1"

# 运行Docker容器
docker run --rm -it \
    --network=host \
    $GPU_SUPPORT \
    -v "$PROJECT_ROOT:/app" \
    -v "$PROJECT_ROOT/data:/data" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY="$DISPLAY" \
    -e PANDA3D_WINDOW_TYPE="$PANDA3D_WINDOW_TYPE" \
    -e LIBGL_ALWAYS_INDIRECT=1 \
    -e MESA_GL_VERSION_OVERRIDE=3.3 \
    -w /app/examples/pooltool \
    pooltool \
    /bin/bash -c "source /.venv/bin/activate && python milestone1_basic_pooltool.py"

echo ""
echo "=== 检查生成的文件 ==="
echo "输出目录: $PROJECT_ROOT/data/pooltool/milestone1"
ls -la "$PROJECT_ROOT/data/pooltool/milestone1/"

echo ""
echo "🎉 Milestone 1 运行完成！"
echo ""
echo "生成的文件说明："
echo "  - *.pkl: PoolTool系统状态文件，可用于后续分析"
echo "  - *.json: 球轨迹数据，便于可视化和分析"
echo ""
echo "如需交互式使用，运行："
echo "  docker run --rm -it $GPU_SUPPORT -v $PROJECT_ROOT:/app pooltool /bin/bash" 