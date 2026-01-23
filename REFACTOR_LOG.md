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

## 7. 2026-01-22 调试与修复记录 (Docker & 兼容性)

本次调试旨在解决容器化部署中的依赖地狱、模块缺失以及新旧硬件兼容性问题。

### 7.1 依赖与构建修复
- **依赖瘦身**: 修正 `requirements.txt`，移除 `torch` 和 `torchaudio`（利用基础镜像预装版本），防止重复安装导致的镜像体积膨胀（从 16GB 降至正常水平）。添加 `x_transformers` 缺失依赖。
- **模块缺失补全**:
    - 创建 `GPT_SoVITS/utils.py`: 解决 `torch.load` 加载权重时因缺少顶层 `utils` 模块而抛出的 `ModuleNotFoundError`。
    - 创建 `GPT_SoVITS/module/distrib.py` 和 `ddp_utils.py`: 提供 `is_distributed` 和 `broadcast_tensors` 的伪实现，解决推理时对分布式训练代码的硬依赖。
- **导入路径修复**: 修改 `GPT_SoVITS/feature_extractor/cnhubert.py`，注释掉顶层 `import utils` 和 `if __name__ == "__main__":` 块，防止库加载时报错。

### 7.2 跨平台硬件兼容性 (JIT vs NVRTC)
- **问题**: 在 RTX 5060 Ti (新架构) 上运行 CUDA 11.8 容器时，PyTorch 的 JIT 编译器 (`nvrtc`) 报错 `invalid value for --gpu-architecture`。
- **原因**: 容器内的旧版 CUDA 工具链不支持新显卡架构的即时编译。
- **解决方案**: 修改 `GPT_SoVITS/module/commons.py`，注释掉 `@torch.jit.script` 装饰器。
- **效果**: 禁用 JIT 编译，回退到普通 Python/Torch 执行模式。这确保了代码既能在 RTX 50 系列新卡上运行（避开编译错误），也能在 GTX 1660 Ti 等老卡上运行（保持兼容），实现了真正的跨平台部署。

### 7.3 开发与测试优化
- **开发模式**: 更新 `docker-compose.yaml`，将宿主机根目录挂载到容器 `/app`，实现代码热重载，避免频繁重建镜像。
- **测试脚本**: 增强 `tests/test_api.py`，将超时时间延长至 300秒，以适应首次推理时的模型加载与 JIT 预热（如有）。
- **ONNXRuntime**: 确认 `onnxruntime-gpu` 因 CUDA 版本不匹配 (12 vs 11) 而回退到 CPU 模式，但不影响核心 TTS 功能的使用。

### 7.4 部署注意事项
- **模型挂载**: 必须在宿主机准备好 `GPT_SoVITS/pretrained_models/` 目录及相关模型文件（BERT, Hubert, SoVITS v4 底模），并正确挂载到容器。
- **忽略规则**: 更新 `.gitignore` 防止意外提交预训练模型大文件。

---



## 8. 2026-01-23 调试与修复记录 (NLTK 资源缺失)



本次修复解决了在容器环境中处理英文文本时，由于缺少 NLTK 语料库导致的推理失败问题。



### 8.1 NLTK 资源补齐

- **问题**: 推理英文文本时报错 `LookupError: Resource averaged_perceptron_tagger_eng not found.`。

- **原因**: 容器基础镜像未包含 NLTK 的词性标注（POS Tagger）和分词（Punkt）模型。

- **解决方案**:

    - **代码层修复**: 在 `api_v2.py` 的初始化阶段加入自动检测与下载逻辑：

      ```python

      import nltk

      try:

          nltk.data.find('taggers/averaged_perceptron_tagger_eng')

      except LookupError:

          nltk.download('averaged_perceptron_tagger_eng')

      ```

    - **镜像层修复**: 修改 `Dockerfile`，在构建过程中执行 `RUN python -m nltk.downloader averaged_perceptron_tagger_eng punkt`，确保镜像离线可用性。

- **效果**: 解决了处理英文文本时的崩溃问题，提升了服务的健壮性。



---

*上次更新: 2026年1月23日*