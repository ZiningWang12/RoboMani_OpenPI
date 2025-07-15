# Dockerfile for PoolTool台球仿真环境
# 解决panda3d依赖和可视化问题

# Build the container:
# docker build . -t pooltool -f examples/pooltool/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --gpus all pooltool /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# 安装系统依赖 - 图形渲染和构建工具
RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libxcursor-dev \
    libxinerama-dev \
    libfreetype6-dev \
    libassimp-dev \
    libopenal-dev \
    libvorbis-dev \
    libssl-dev \
    git \
    pkg-config \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# 复制requirements文件
COPY ./examples/pooltool/requirements.txt /tmp/requirements.txt

# 创建Python环境并安装依赖
# 使用Python 3.10确保兼容性
RUN uv venv --python 3.10 $UV_PROJECT_ENVIRONMENT

# 安装基础依赖（不指定具体版本，让uv解析）
RUN uv pip install \
    numpy \
    scipy \
    matplotlib \
    imageio \
    tqdm \
    msgpack \
    pyyaml \
    click \
    mujoco \
    gymnasium \
    dm-control \
    opencv-python \
    pillow \
    tyro \
    websockets \
    pytest \
    black

# 添加panda3d archive源并安装panda3d
RUN uv pip install --extra-index-url https://archive.panda3d.org/simple/ \
    --pre \
    panda3d==1.11.0.dev3702

# 安装panda3d扩展
RUN uv pip install \
    panda3d-gltf \
    panda3d-simplepbr

# 安装pooltool依赖（让poetry解析版本）
COPY ./third_party/pooltool /tmp/pooltool
RUN cd /tmp/pooltool && \
    uv pip install --extra-index-url https://archive.panda3d.org/simple/ \
    attrs \
    cattrs \
    msgpack-numpy \
    numba \
    llvmlite \
    h5py \
    h11 \
    rich

# 设置环境变量
ENV PYTHONPATH=/app:/app/third_party/pooltool
ENV PANDA3D_WINDOW_TYPE=offscreen

# 创建数据目录
RUN mkdir -p /app/data/pooltool/milestone1 \
    /app/data/pooltool/milestone2 \
    /app/data/pooltool/milestone3

# 设置工作目录权限
RUN chmod -R 755 /app

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && cd /app/examples/pooltool && python milestone1_basic_pooltool.py"] 