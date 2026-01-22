# 建议使用官方 PyTorch 运行时镜像，包含 CUDA 和 cuDNN
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 设置构建参数，控制是否使用国内源，默认不使用
ARG USE_MIRROR=false

# 设置工作目录
WORKDIR /app

# 设置时区和语言，防止中文乱码
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8

# 根据构建参数决定是否替换为清华源
RUN if [ "$USE_MIRROR" = "true" ]; then \
        sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
        sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list; \
    fi

# 安装系统级依赖 (OpenCC 需要编译工具，ffmpeg 用于音频处理)
# build-essential 和 cmake 是编译 opencc 和 pyopenjtalk 必须的
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 1. 强制 opencc 从源码编译，解决 Linux GLIBC 版本不兼容问题
# 2. 使用清华源加速
# 3. 清理缓存减小体积
RUN pip install --no-cache-dir --no-binary opencc -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip cache purge

# 复制项目文件
COPY . .

# 创建必要的挂载点目录
RUN mkdir -p GPT_SoVITS/pretrained_models custom_weights

# 暴露端口
EXPOSE 3333

# 启动命令
CMD ["python", "api_v2.py", "-a", "0.0.0.0", "-p", "3333", "-c", "GPT_SoVITS/configs/tts_infer.yaml"]
