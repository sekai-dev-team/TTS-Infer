# GPT-SoVITS-Infer: 高性能推理精简版

本仓库是 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的**精简重构版**，专注于提供高性能、轻量化的 **TTS 推理服务**。

## 🌟 项目特性

- **专注于推理**: 移除了所有训练相关的代码、WebUI 界面、Jupyter Notebooks 以及复杂的安装脚本，代码库体积显著减小。
- **环境精简**: 重新梳理了依赖关系，移除了 `gradio` (WebUI 核心)、`tensorboard` (训练可视化) 和 `pytorch-lightning` (训练框架) 等非必要库。
- **纯 Torch 实现**: 对 `t2s_lightning_module.py` 进行了重构，解除了对 `pytorch-lightning` 的强依赖，使推理过程更加纯粹且稳定。
- **容器化友好**: 提供专门优化的 Dockerfile 和 Docker Compose 配置，支持单条命令快速部署推理 API。
- **API v2 驱动**: 默认使用 FastAPI 驱动的 API v2 接口，支持流式输出 (WAV/Media)。

## 🚀 快速开始

### 1. 准备模型
请将您的模型文件放入宿主机的指定目录：
- **基础模型**: GPT, SoVITS, BERT 等预训练权重。
- **微调模型**: 您自己微调出的 `.ckpt` 和 `.pth` 权重。

### 2. 构建与启动 (Docker)
我们建议在 `gpt-sovits-infer` 目录下执行操作：

```bash
cd gpt-sovits-infer
docker compose up -d --build
```

### 3. API 调用
服务启动后默认监听 `9880` 端口。

- **TTS 推理**: `GET /tts?text=...&text_lang=zh&ref_audio_path=...&prompt_lang=zh`
- **切换模型**:
    - `GET /set_gpt_weights?weights_path=custom_weights/your_model.ckpt`
    - `GET /set_sovits_weights?weights_path=custom_weights/your_model.pth`

## 📂 目录说明

```text
.
├── gpt-sovits-infer/   # 核心推理代码库
│   ├── api_v2.py       # API 入口
│   ├── GPT_SoVITS/     # 核心模型定义与逻辑 (已重构)
│   ├── tools/          # 必要的工具脚本 (文本切分、国际化等)
│   ├── Dockerfile      # 推理镜像构建配置
│   └── requirements.txt # 精简版依赖列表
└── README.md           # 本说明文件
```

## 📜 鸣谢与来源

本项目的核心逻辑源自 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 及其贡献者。感谢原作者为开源社区提供的优秀 TTS 模型。

本项目仅在原版基础上进行工程化精简，未改变任何核心推理算法。

---
*注：如果您需要训练模型或使用可视化界面，请前往 [官方仓库](https://github.com/RVC-Boss/GPT-SoVITS)。*