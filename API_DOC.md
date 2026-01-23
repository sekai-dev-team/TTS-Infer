# GPT-SoVITS-Infer API 文档

本服务提供基于 GPT-SoVITS 的语音合成 (TTS) 接口。

## 启动服务

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

- `-a`: 绑定地址，默认 `127.0.0.1`
- `-p`: 绑定端口，默认 `9880`
- `-c`: 配置文件路径

---

## 核心接口

### 1. 语音合成 (TTS)

**Endpoint**: `/tts`
**Method**: `GET` / `POST`

#### 请求参数 (POST JSON / GET Query):

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `text` | string | 是 | - | 需要合成的文本 |
| `text_lang` | string | 否 | `auto` | 文本语种 (zh/en/ja/ko/yue/auto)。设为 `auto` 则自动识别。 |
| `ref_audio_path` | string | 是 | - | 参考音频路径 |
| `prompt_text` | string | 否 | "" | 参考音频的文本。若为空，将尝试自动寻找同名 `.lab` 或 `.txt` 文件。 |
| `prompt_lang` | string | 否 | - | 参考音频的语种。若为空，将尝试从文件名解析 (格式: `文本_语种.wav`)。 |
| `streaming_mode` | bool/int | 否 | `false` | 流式返回模式。0/false: 关闭; 1/true: 高质量(较慢); 2: 中质量; 3: 低质量(最快)。 |
| `media_type` | string | 否 | `wav` | 返回音频格式 (wav/ogg/aac/raw) |
| `speed_factor` | float | 否 | 1.0 | 语速倍率 |
| `top_k` | int | 否 | 15 | 采样参数 top_k |
| `top_p` | float | 否 | 1.0 | 采样参数 top_p |
| `temperature` | float | 否 | 1.0 | 采样温度 |
| `text_split_method` | string | 否 | `cut5` | 文本切分方法 (cut0-5) |
| `batch_size` | int | 否 | 1 | 推理批大小 |

#### 返回值:
- **成功**: 返回音频二进制数据流 (HTTP 200)
- **失败**: 返回包含错误信息的 JSON (HTTP 400)

---

### 2. 模型切换 (综合)

**Endpoint**: `/set_model`
**Method**: `GET`

#### 请求参数:
| 参数名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `model_name` | string | 否 | 模型名称。将在权重目录下自动匹配同名的 `.ckpt` (GPT) 和 `.pth` (SoVITS) 文件。 |
| `gpt_path` | string | 否 | 显式指定 GPT 权重路径 |
| `sovits_path` | string | 否 | 显式指定 SoVITS 权重路径 |

*注：必须提供 `model_name` 或 同时提供 `gpt_path` 和 `sovits_path`。*

---

### 3. 单独切换权重

- **切换 GPT**: `/set_gpt_weights?weights_path=xxx.ckpt`
- **切换 SoVITS**: `/set_sovits_weights?weights_path=xxx.pth`

---

### 4. 服务控制

- **重启服务**: `/control?command=restart`
- **关闭服务**: `/control?command=exit`

---

## 示例

### Python 请求示例

```python
import requests

url = "http://127.0.0.1:9880/tts"
data = {
    "text": "你好，我是人工智能助手。",
    "text_lang": "zh",
    "ref_audio_path": "ref_audio/example.wav",
    "prompt_text": "这是一个参考音频示例",
    "prompt_lang": "zh"
}

response = requests.post(url, json=data)
with open("output.wav", "wb") as f:
    f.write(response.content)
```
