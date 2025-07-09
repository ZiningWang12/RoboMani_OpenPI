# Pool Robot Simulation Docker Image
# 台球机械臂仿真Docker镜像
FROM ubuntu:22.04

LABEL maintainer="OpenPI Team"
LABEL description="Pool Robot Simulation with OpenPI and Pooltool"
LABEL version="1.0.0"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV MUJOCO_GL=egl

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python-is-python3 \
    pip \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    xvfb \
    x11-apps \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# 安装UV包管理器
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# 设置工作目录
WORKDIR /workspace

# 复制项目文件
COPY requirements.txt requirements.in ./
COPY *.py ./
COPY README.md ./

# 创建Python虚拟环境并安装依赖
RUN uv venv --python 3.10 .venv
RUN .venv/bin/pip install --upgrade pip
RUN .venv/bin/pip install -r requirements.txt

# 创建数据目录
RUN mkdir -p data/pooltool_results

# 创建启动脚本
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🎱 Starting Pool Robot Simulation Container"\n\
echo "========================================"\n\
\n\
# 激活虚拟环境\n\
source .venv/bin/activate\n\
\n\
# 验证环境\n\
echo "Verifying environment..."\n\
python -c "import pooltool, pybullet; print(\"✅ Environment verified\")" || {\n\
    echo "❌ Environment verification failed"\n\
    exit 1\n\
}\n\
\n\
# 运行仿真\n\
echo "Starting simulation with parameters: $@"\n\
python main.py "$@"\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# 设置启动脚本为入口点
ENTRYPOINT ["/workspace/start.sh"]

# 默认参数
CMD ["--task", "evaluation", "--trials", "3", "--no-gui"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import pooltool, pybullet; print('OK')" || exit 1

# 暴露端口（如果需要Web界面）
EXPOSE 8080

# 设置卷挂载点
VOLUME ["/workspace/data"] 