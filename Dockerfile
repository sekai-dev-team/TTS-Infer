# 建议使用官方 PyTorch 运行时镜像，包含 CUDA 和 cuDNN
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置时区和语言，防止中文乱码
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8

# 安装系统级依赖 (OpenCC 需要编译工具，ffmpeg 用于音频处理)
# 注意：如果基础镜像很简陋，可能需要安装 build-essential
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements_infer.txt .

# 安装 Python 依赖
# 使用清华源加速，--no-cache-dir 减小体积
RUN pip install --no-cache-dir -r requirements_infer.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
COPY . .

# 创建必要的挂载点目录（占位）
# pretrained_models: 基础模型 (GPT/SoVITS/BERT)
# custom_weights: 用户微调的模型
RUN mkdir -p GPT_SoVITS/pretrained_models custom_weights

# 暴露端口
EXPOSE 9880

# 启动命令
# host 0.0.0.0 允许外部访问
CMD ["python", "api_v2.py", "-a", "0.0.0.0", "-p", "9880", "-c", "GPT_SoVITS/configs/tts_infer.yaml"]
