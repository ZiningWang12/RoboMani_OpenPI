#!/bin/bash
# PoolTool x OpenPI 项目环境设置脚本

set -e  # 出错时停止

echo "=== PoolTool x OpenPI 环境设置 ==="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "项目根目录: $PROJECT_ROOT"
echo "PoolTool示例目录: $SCRIPT_DIR"

# 创建虚拟环境（如果不存在）
VENV_PATH="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "创建虚拟环境..."
    python3 -m venv "$VENV_PATH"
else
    echo "虚拟环境已存在: $VENV_PATH"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source "$VENV_PATH/bin/activate"

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装基础依赖
echo "安装requirements.txt中的依赖..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# 安装pooltool（开发模式）
echo "安装pooltool（开发模式）..."
POOLTOOL_PATH="$PROJECT_ROOT/third_party/pooltool"
if [ -d "$POOLTOOL_PATH" ]; then
    pip install -e "$POOLTOOL_PATH"
    echo "✅ Pooltool安装完成"
else
    echo "❌ 错误: 找不到pooltool路径: $POOLTOOL_PATH"
    exit 1
fi

# 创建数据目录
echo "创建数据目录..."
mkdir -p "$SCRIPT_DIR/data/pooltool/milestone1"
mkdir -p "$SCRIPT_DIR/data/pooltool/milestone2"
mkdir -p "$SCRIPT_DIR/data/pooltool/milestone3"

# 验证安装
echo "验证安装..."
python -c "
import sys
sys.path.insert(0, '$POOLTOOL_PATH')
import pooltool as pt
print(f'✅ PoolTool版本: {pt.__version__}')
print(f'✅ 支持的游戏类型: {[gt.name for gt in pt.GameType]}')
import numpy as np
print(f'✅ NumPy版本: {np.__version__}')
"

echo ""
echo "🎉 环境设置完成！"
echo ""
echo "使用方法:"
echo "1. 激活虚拟环境: source $VENV_PATH/bin/activate"
echo "2. 运行Milestone 1 demo: python milestone1_basic_pooltool.py"
echo "3. 检查生成的文件: ls -la data/pooltool/milestone1/"
echo "" 