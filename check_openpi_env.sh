#!/bin/bash

# OpenPI环境快速检查脚本
# 基于WSL Ubuntu 22.04配置实践

echo "🔍 OpenPI环境配置检查"
echo "======================="

# 检查基础环境
echo ""
echo "📋 基础环境检查:"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo '❌ 未检测到GPU')"
echo "  CUDA版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null || echo '❌ CUDA不可用')"
echo "  Docker: $(docker --version 2>/dev/null || echo '❌ Docker未安装')"
echo "  uv: $(uv --version 2>/dev/null || echo '❌ uv未安装')"

# 检查项目结构
echo ""
echo "📁 项目结构检查:"
echo "  项目根目录: $(pwd)"
echo "  Git子模块: $(git submodule status | wc -l) 个"
echo "  LIBERO目录: $([ -d third_party/libero ] && echo '✅ 存在' || echo '❌ 不存在')"

# 检查LIBERO环境
echo ""
echo "🐍 LIBERO环境检查:"
LIBERO_VENV="examples/libero/.venv"
if [ -d "$LIBERO_VENV" ]; then
    echo "  虚拟环境: ✅ 存在 ($LIBERO_VENV)"
    
    # 激活环境并检查关键包
    source $LIBERO_VENV/bin/activate
    
    echo "  Python版本: $(python --version 2>/dev/null || echo '❌ Python不可用')"
    echo "  mujoco: $(python -c 'import mujoco; print(f"✅ v{mujoco.__version__}")' 2>/dev/null || echo '❌ 导入失败')"
    echo "  libero: $(python -c 'import libero; print("✅ 可用")' 2>/dev/null || echo '❌ 导入失败')"
    echo "  openpi-client: $(python -c 'from openpi_client import websocket_client_policy; print("✅ 可用")' 2>/dev/null || echo '❌ 导入失败')"
    
    deactivate
else
    echo "  虚拟环境: ❌ 不存在 ($LIBERO_VENV)"
fi

# 检查LIBERO配置
echo ""
echo "⚙️ LIBERO配置检查:"
LIBERO_CONFIG="$HOME/.libero/config.yaml"
if [ -f "$LIBERO_CONFIG" ]; then
    echo "  配置文件: ✅ 存在"
    
    # 检查关键路径
    CURRENT_DIR=$(pwd)
    EXPECTED_PATH="$CURRENT_DIR/third_party/libero/libero/libero"
    
    if grep -q "$EXPECTED_PATH" "$LIBERO_CONFIG"; then
        echo "  路径配置: ✅ 正确"
    else
        echo "  路径配置: ❌ 需要修复"
        echo "    预期路径: $EXPECTED_PATH"
        echo "    当前配置: $(grep benchmark_root $LIBERO_CONFIG | cut -d' ' -f2)"
    fi
else
    echo "  配置文件: ❌ 不存在 ($LIBERO_CONFIG)"
fi

# 检查Docker容器
echo ""
echo "🐳 Docker环境检查:"
if command -v docker >/dev/null 2>&1; then
    CONTAINERS=$(docker ps -a --filter "name=libero" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "")
    if [ -n "$CONTAINERS" ]; then
        echo "  LIBERO容器:"
        echo "$CONTAINERS" | sed 's/^/    /'
    else
        echo "  LIBERO容器: ❌ 未找到"
    fi
else
    echo "  Docker: ❌ 不可用"
fi

# 环境变量建议
echo ""
echo "🔧 环境变量建议:"
echo "  每次使用LIBERO时，请设置:"
echo "    source examples/libero/.venv/bin/activate"
echo "    export PYTHONPATH=\$PYTHONPATH:\$PWD/third_party/libero"
echo "    export MUJOCO_GL=egl"

# 总结
echo ""
echo "✨ 检查完成！"
echo "   详细配置指南请参考: run_libero_inference.md" 