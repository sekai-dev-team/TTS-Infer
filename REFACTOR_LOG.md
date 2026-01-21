# GPT-SoVITS-Infer 重构技术文档

本文档记录了将原始 GPT-SoVITS 项目重构为轻量级推理服务过程中的所有关键改动，以便后续维护与排错。

## 1. 重构目标
- **去重型化**: 移除所有训练相关的依赖和代码，减小镜像体积。
- **去框架化**: 移除 `pytorch-lightning`，改为原生 `torch.nn.Module` 实现。
- **容器化**: 提供优化的 Docker 部署方案。
- **纯粹推理**: 仅保留 `api_v2` 接口，移除 WebUI 界面。

## 2. 核心代码改动

### 2.1 模型定义适配 (`GPT_SoVITS/AR/models/t2s_lightning_module.py`)
- **改动**: 将基类从 `LightningModule` 修改为 `torch.nn.Module`。
- **原因**: 为了移除体积巨大的 `pytorch-lightning` 及其依赖（如 `fsspec`, `tensorboard` 等）。
- **影响**: 移除了 `training_step`, `validation_step`, `configure_optimizers` 等训练专用方法，保留并优化了推理所需的 `init` 和 `forward` 逻辑。

### 2.2 推理逻辑优化 (`GPT_SoVITS/TTS_infer_pack/TTS.py`)
- **改动**: 确认并保留了对 `AP_BWE` (超分) 和 `BigVGAN` (声码器) 的调用。
- **修复**: 确保所有路径引用（如 `GPT_SoVITS/BigVGAN`）在精简后的目录结构下依然有效。

### 2.3 接口适配 (`api_v2.py`)
- **改动**: 清理了启动参数，确保其在没有原版 WebUI 环境变量的情况下也能独立运行。
- **改动**: 确认接口支持最新的 v3/v4/v2ProPlus 模型权重切换。

## 3. 依赖项调整 (`requirements.txt`)
- **移除**: `gradio`, `tensorboard`, `pytorch-lightning`, `funasr`, `modelscope`, `onnxruntime-gpu` (除非明确需要 ONNX)。
- **保留**: `torch`, `torchaudio`, `transformers`, `fastapi`, `uvicorn` 以及文本处理相关的基础库。
- **效果**: 运行环境体积显著缩小，安装速度提升。

## 4. 文件结构精简
为了保持项目整洁，删除了以下非必要内容：
- **训练脚本**: `s1_train.py`, `s2_train.py` 等。
- **数据准备**: `prepare_datasets/` 目录。
- **旧版 WebUI**: `webui.py`, `inference_webui.py` 等。
- **冗余工具**: `uvr5` (人声分离), `asr` (自动语音识别) 等在单纯推理阶段不常用的工具。
- **文档与脚本**: 删除了原项目的 `docs/`, `Docker/` (旧版), `.ipynb` 笔记本和 `.bat/.sh` 安装脚本。

## 5. 部署环境改动
- **Dockerfile**: 基于 `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` 构建。
- **磁盘优化**: 在 GitHub Actions (`deploy.yml`) 中加入了磁盘清理步骤，防止构建镜像时空间不足。
- **配置**: 预留了 `GPT_SoVITS/configs/tts_infer.yaml` 作为核心配置文件，通过挂载方式支持自定义模型。

## 6. 常见问题解决 (FAQ)
- **Q: 提示缺失 `pytorch-lightning`?**
  - A: 请检查是否误用了原版的训练相关代码，或者某些导入路径依然在引用 Lightning 特性。本项目已将其完全剥离。
- **Q: 无法切换模型?**
  - A: 确保通过 `/set_gpt_weights` 或 `/set_sovits_weights` 传入的是相对于项目根目录的正确路径。
- **Q: 缺少 ASR 或 UVR5 功能?**
  - A: 这些功能属于训练/预处理范畴，已在推理版中移除。如需使用，请在外部处理好音频后再作为参考音频传入。

---
*重构时间: 2026年1月21日*
