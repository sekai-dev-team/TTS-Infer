"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 15,                  # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": False,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
    "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
    "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
    "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
    "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

### 切换TTS模型 (同时切换GPT和Sovits)

endpoint: `/set_model`

GET:
```
http://127.0.0.1:9880/set_model?gpt_path=gpt_weights/my_gpt.ckpt&sovits_path=sovits_weights/my_sovits.pth
```
或者通过名称自动寻找 (在权重目录下寻找同名文件):
```
http://127.0.0.1:9880/set_model?model_name=my_model
```

RESP:
成功: 返回成功信息, http code 200
失败: 返回包含错误信息的 json, http code 400

"""

import os
import sys
import traceback
import time
from typing import Generator, Union

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from GPT_SoVITS.text.LangSegmenter.langsegmenter import LangSegmenter
from pydantic import BaseModel
import threading
import uuid
import asyncio
from contextlib import asynccontextmanager

# --- NLTK Resource Check ---
import nltk
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("NLTK resource 'averaged_perceptron_tagger_eng' not found. Downloading...")
    nltk.download('averaged_perceptron_tagger_eng')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK resource 'punkt' not found. Downloading...")
    nltk.download('punkt')
# ---------------------------

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)

# --- Auto-load models from Docker volume paths ---
def scan_and_update_model_path(config_path_attr, search_dir, extensions, config_obj):
    # PRIORITIZE searching in the specific search_dir (e.g., /app/gpt_weights)
    if os.path.exists(search_dir):
        for root, _, files in os.walk(search_dir):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    found_path = os.path.join(root, file)
                    print(f"Auto-detected custom model weight: {found_path}")
                    setattr(config_obj, config_path_attr, found_path)
                    return

    # Fallback: check if the current path in config actually exists
    current_path = getattr(config_obj, config_path_attr)
    if current_path and os.path.exists(current_path):
        return
    
    # Second fallback: check default pretrained models
    pretrained_dir = "GPT_SoVITS/pretrained_models"
    if os.path.exists(pretrained_dir):
         for root, _, files in os.walk(pretrained_dir):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    if "s1bert" in file and config_path_attr == "t2s_weights_path":
                         found_path = os.path.join(root, file)
                         setattr(config_obj, config_path_attr, found_path)
                         return
                    if "s2G" in file and config_path_attr == "vits_weights_path":
                         found_path = os.path.join(root, file)
                         setattr(config_obj, config_path_attr, found_path)
                         return

# Docker volume paths
docker_gpt_dir = "/app/gpt_weights"
docker_sovits_dir = "/app/sovits_weights"
# Local paths for testing/compatibility
local_gpt_dir = "gpt_weights"
local_sovits_dir = "sovits_weights"

scan_and_update_model_path("t2s_weights_path", docker_gpt_dir if os.path.exists(docker_gpt_dir) else local_gpt_dir, [".ckpt"], tts_config)
scan_and_update_model_path("vits_weights_path", docker_sovits_dir if os.path.exists(docker_sovits_dir) else local_sovits_dir, [".pth"], tts_config)

print(tts_config)
tts_pipeline = TTS(tts_config)

tts_lock = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_lock
    tts_lock = asyncio.Lock()
    yield

APP = FastAPI(lifespan=lifespan)


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = False
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    # Author: AkagawaTsurunaki
    # Issue:
    #   Stack overflow probabilistically occurs
    #   when the function `sf_writef_short` of `libsndfile_64bit.dll` is called
    #   using the Python library `soundfile`
    # Note:
    #   This is an issue related to `libsndfile`, not this project itself.
    #   It happens when you generate a large audio tensor (about 499804 frames in my PC)
    #   and try to convert it to an ogg file.
    # Related:
    #   https://github.com/RVC-Boss/GPT-SoVITS/issues/1199
    #   https://github.com/libsndfile/libsndfile/issues/1023
    #   https://github.com/bastibe/python-soundfile/issues/396
    # Suggestion:
    #   Or split the whole audio data into smaller audio segment to avoid stack overflow?

    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)



    # See: https://docs.python.org/3/library/threading.html
    # The stack size of this thread is at least 32768
    # If stack overflow error still occurs, just modify the `stack_size`.
    # stack_size = n * 4096, where n should be a positive integer.
    # Here we chose n = 4096.
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")

    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    prompt_text: str = req.get("prompt_text", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    # Resolve ref_audio_path
    if ref_audio_path and not os.path.exists(ref_audio_path):
        # Try finding in Docker/local volume
        docker_ref_dir = "/app/ref_audio"
        local_ref_dir = "ref_audios"
        search_dir = docker_ref_dir if os.path.exists(docker_ref_dir) else local_ref_dir
        potential_path = os.path.join(search_dir, ref_audio_path)
        if os.path.exists(potential_path):
            ref_audio_path = potential_path
            req["ref_audio_path"] = ref_audio_path # Update in request
        elif os.path.exists(search_dir) and not ref_audio_path:
             # Randomly pick one if empty? No, better error out or let logic below handle empty
             pass

    # Auto-parse prompt_text and prompt_lang logic
    if ref_audio_path and os.path.exists(ref_audio_path):
        # 1. Try reading from .lab or .txt file FIRST
        if not prompt_text:
            base_path = os.path.splitext(ref_audio_path)[0]
            for ext in [".lab", ".txt"]:
                text_path = base_path + ext
                print(f"DEBUG: Checking for prompt file at {text_path}")
                if os.path.exists(text_path):
                    try:
                        with open(text_path, "r", encoding="utf-8") as f:
                            prompt_text = f.read().strip()
                            req["prompt_text"] = prompt_text
                            print(f"Auto-detected prompt_text from {ext}: {prompt_text}")
                            break
                    except Exception as e:
                        print(f"Failed to read prompt text from {text_path}: {e}")
                else:
                    print(f"DEBUG: File not found at {text_path}")

        # 2. Fallback: Parse from filename if still missing
        if not prompt_text or not prompt_lang:
            filename = os.path.basename(ref_audio_path)
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split("_")
            if len(parts) >= 2:
                auto_lang = parts[-1]
                auto_text = "_".join(parts[:-1])
                # Mapping common language codes if needed, or assume standard codes
                if not prompt_lang:
                    prompt_lang = auto_lang
                    req["prompt_lang"] = prompt_lang
                    print(f"Auto-detected prompt_lang: {prompt_lang}")
                if not prompt_text:
                    prompt_text = auto_text
                    req["prompt_text"] = prompt_text
                    print(f"Auto-detected prompt_text from filename: {prompt_text}")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    
    # Auto detect text language if set to auto or empty
    if text_lang in [None, "", "auto"]:
        try:
            detected_langs = LangSegmenter.getTexts(text)
            if detected_langs:
                # Pick the most common lang or just the first one's lang? 
                # For GSV, usually we pick the first segment's lang as the primary text_lang
                # or use "auto" if the underlying pipeline supports it.
                # Here we set it to the first detected language to satisfy the pipeline's requirement.
                text_lang = detected_langs[0]['lang']
                req["text_lang"] = text_lang
                print(f"Auto-detected text_lang: {text_lang}")
            else:
                text_lang = "zh" # Fallback
                req["text_lang"] = text_lang
        except Exception as e:
            print(f"Failed to auto-detect language: {e}")
            text_lang = "zh"
            req["text_lang"] = text_lang

    if text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    # elif media_type == "ogg" and not streaming_mode:
    #     return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 15,                  # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "parallel_infer": False,       # bool. whether to use parallel inference.
                "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
                "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
                "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
                "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
                "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res
    
    if streaming_mode == 0:
        streaming_mode = False
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 1:
        streaming_mode = False
        return_fragment = True
        fixed_length_chunk = False
    elif streaming_mode == 2:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = False
    elif streaming_mode == 3:
        streaming_mode = True
        return_fragment = False
        fixed_length_chunk = True

    else:
        return JSONResponse(status_code=400, content={"message": f"the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)"})

    req["streaming_mode"] = streaming_mode
    req["return_fragment"] = return_fragment
    req["fixed_length_chunk"] = fixed_length_chunk

    print(f"{streaming_mode} {return_fragment} {fixed_length_chunk}")

    streaming_mode = streaming_mode or return_fragment


    try:
        # Output directory and filename components
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:6]
        import time as time_module

        if streaming_mode:
            async def streaming_generator(req, media_type):
                async with tts_lock:
                    tts_generator = tts_pipeline.run(req)
                    save_path = os.path.join(output_folder, f"tts_stream_{timestamp}_{unique_id}.{media_type if media_type != 'raw' else 'wav'}")
                    print(f"Saving stream to {save_path}")
                    
                    with open(save_path, "wb") as f_save:
                        if_frist_chunk = True
                        for sr, chunk in tts_generator:
                            if if_frist_chunk and media_type == "wav":
                                header = wave_header_chunk(sample_rate=sr)
                                f_save.write(header)
                                yield header
                                media_type = "raw"
                                if_frist_chunk = False
                            
                            data = pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                            f_save.write(data)
                            yield data

            return StreamingResponse(
                streaming_generator(req, media_type),
                media_type=f"audio/{media_type}",
            )

        else:
            async with tts_lock:
                tts_generator = tts_pipeline.run(req)
                sr, audio_data = next(tts_generator)
                audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
                
                # Explicitly close the generator to trigger the 'finally' block in TTS.run immediately,
                # ensuring resources and VRAM are released.
                tts_generator.close()

            save_path = os.path.join(output_folder, f"tts_{timestamp}_{unique_id}.{media_type}")
            with open(save_path, "wb") as f:
                f.write(audio_data)
            print(f"Audio saved to: {save_path}")

            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        print(f"TTS Failed with request: {req}")
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    parallel_infer: bool = False,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
    streaming_mode: Union[bool, int] = False,
    overlap_length: int = 2,
    min_chunk_length: int = 16,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
        "overlap_length": int(overlap_length),
        "min_chunk_length": int(min_chunk_length),
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        async with tts_lock:
            tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        async with tts_lock:
            tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_model")
async def set_model(model_name: str = None, gpt_path: str = None, sovits_path: str = None):
    try:
        if model_name:
            # 自动寻找同名模型
            docker_gpt_dir = "/app/gpt_weights"
            docker_sovits_dir = "/app/sovits_weights"
            local_gpt_dir = "gpt_weights"
            local_sovits_dir = "sovits_weights"
            
            gpt_dir = docker_gpt_dir if os.path.exists(docker_gpt_dir) else local_gpt_dir
            sovits_dir = docker_sovits_dir if os.path.exists(docker_sovits_dir) else local_sovits_dir
            
            if not gpt_path:
                for root, _, files in os.walk(gpt_dir):
                    for file in files:
                        if file.startswith(model_name) and file.endswith(".ckpt"):
                            gpt_path = os.path.join(root, file)
                            break
                    if gpt_path: break
            
            if not sovits_path:
                for root, _, files in os.walk(sovits_dir):
                    for file in files:
                        if file.startswith(model_name) and file.endswith(".pth"):
                            sovits_path = os.path.join(root, file)
                            break
                    if sovits_path: break
        
        if not gpt_path or not sovits_path:
            return JSONResponse(status_code=400, content={"message": "gpt_path and sovits_path (or model_name) are required"})
        
        async with tts_lock:
            tts_pipeline.init_gpt_sovits_weights(gpt_path, sovits_path)
        return JSONResponse(status_code=200, content={"message": "success", "gpt_path": gpt_path, "sovits_path": sovits_path})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set model failed", "Exception": str(e)})


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
