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
    "top_k": 5,                   # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": False,      # bool. whether to return a streaming response.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
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

"""

import os
import sys
import traceback
from typing import Generator, Optional

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

pretrain_models_dir = now_dir + "\\GPT_SoVITS\\pretrained_models\\"

sovits_v3_model_dir = now_dir + "\\SoVITS_weights_v3\\"

gpt_v3_model_dir = now_dir + "\\GPT_weights_v3\\"

from training_cli_1_audio_processing import slice_audio, denoise, open_asr
from training_cli_2_train_data_processing import preprocess_one_step
from training_cli_3_train_model import start_sovits_training, start_gpt_training
from training_cli_4_inference_cli_v2 import generate_audio

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import (
    get_method_names as get_cut_method_names,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument(
    "-c",
    "--tts_config",
    type=str,
    default="GPT_SoVITS/configs/tts_infer.yaml",
    help="tts_infer路径",
)
parser.add_argument(
    "-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1"
)
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
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
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
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(
        io_buffer, mode="w", samplerate=rate, channels=1, format="ogg"
    ) as audio_file:
        audio_file.write(data)
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
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(
            status_code=400, content={"message": "ref_audio_path is required"}
        )
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(
            status_code=400, content={"message": "text_lang is required"}
        )
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"
            },
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(
            status_code=400, content={"message": "prompt_lang is required"}
        )
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"
            },
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(
            status_code=400,
            content={"message": f"media_type: {media_type} is not supported"},
        )
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(
            status_code=400,
            content={"message": "ogg format is not supported in non-streaming mode"},
        )

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"text_split_method:{text_split_method} is not supported"
            },
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
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
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

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    try:
        tts_generator = tts_pipeline.run(req)

        if streaming_mode:

            def streaming_generator(tts_generator: Generator, media_type: str):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                for sr, chunk in tts_generator:
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(
                streaming_generator(
                    tts_generator,
                    media_type,
                ),
                media_type=f"audio/{media_type}",
            )

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"message": f"tts failed", "Exception": str(e)}
        )


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
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut0",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
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
        return JSONResponse(
            status_code=400,
            content={"message": f"set refer audio failed", "Exception": str(e)},
        )
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
            return JSONResponse(
                status_code=400, content={"message": "gpt weight path is required"}
            )
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"change gpt weight failed", "Exception": str(e)},
        )

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(
                status_code=400, content={"message": "sovits weight path is required"}
            )
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"change sovits weight failed", "Exception": str(e)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


# *********** Training Apis ***********

"""
slice_audio(
        inp=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/Input",
        opt_root=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/1_slice_audio_out",
        threshold=-34,
        min_length=4000,
        min_interval=300,
        hop_size=10,
        max_sil_kept=500,
        _max=0.9,
        alpha=0.25,
        n_parts=4,
    )
"""


@APP.post("/training/slice_audio")
async def slice_audio_endpoint(
    inp: str = None,
    opt_root: str = None,
    threshold: int = -34,
    min_length: int = 4000,
    min_interval: int = 300,
    hop_size: int = 10,
    max_sil_kept: int = 500,
    _max: float = 0.9,
    alpha: float = 0.25,
    n_parts: int = 4,
):
    try:
        if inp in [None, ""]:
            return JSONResponse(status_code=400, content={"message": "inp is required"})
        if opt_root in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "opt_root is required"}
            )
        slice_audio(
            inp=inp,
            opt_root=opt_root,
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=hop_size,
            max_sil_kept=max_sil_kept,
            _max=_max,
            alpha=alpha,
            n_parts=n_parts,
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"slice audio failed", "Exception": str(e)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


"""
    denoise(
        denoise_inp_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/1_slice_audio_out",
        denoise_opt_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/2_denoise_out",
    )
"""


@APP.post("/training/denoise")
async def denoise_endpoint(denoise_inp_dir: str = None, denoise_opt_dir: str = None):
    try:
        if denoise_inp_dir in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "denoise_inp_dir is required"}
            )
        if denoise_opt_dir in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "denoise_opt_dir is required"}
            )
        denoise(denoise_inp_dir=denoise_inp_dir, denoise_opt_dir=denoise_opt_dir)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"denoise failed", "Exception": str(e)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


"""
open_asr(
        asr_inp_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/2_denoise_out",
        asr_opt_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/3_asr_out",
        asr_model="Faster Whisper (多语种)",
        asr_model_size="large-v3",
        asr_lang="auto",
        asr_precision="int8",
    )
"""


@APP.post("/training/asr")
async def asr_endpoint(
    asr_inp_dir: str = None,
    asr_opt_dir: str = None,
    asr_model: str = "Faster Whisper (多语种)",
    asr_model_size: str = "large-v3",
    asr_lang: str = "auto",
    asr_precision: str = "int8",
):
    try:
        if asr_inp_dir in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "asr_inp_dir is required"}
            )
        if asr_opt_dir in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "asr_opt_dir is required"}
            )
        open_asr(
            asr_inp_dir=asr_inp_dir,
            asr_opt_dir=asr_opt_dir,
            asr_model=asr_model,
            asr_model_size=asr_model_size,
            asr_lang=asr_lang,
            asr_precision=asr_precision,
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"asr failed", "Exception": str(e)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


"""
    preprocess_one_step(inp_text=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/3_asr_out/2_denoise_out.list",
                        inp_wav_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/2_denoise_out",
                        exp_name=exp_name,
                        gpu_numbers1a="0-0",
                        gpu_numbers1Ba="0-0",
                        gpu_numbers1c="0-0",
                        bert_pretrained_dir="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                        ssl_pretrained_dir="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base",
                        pretrained_s2G_path="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                        opt_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/{exp_name}")
"""


@APP.post("/training/preprocess")
async def preprocess_endpoint(
    inp_text: str = None,
    inp_wav_dir: str = None,
    exp_name: str = None,
    gpu_numbers1a: str = "0-0",
    gpu_numbers1Ba: str = "0-0",
    gpu_numbers1c: str = "0-0",
    bert_pretrained_dir: str = "chinese-roberta-wwm-ext-large",
    ssl_pretrained_dir: str = "chinese-hubert-base",
    pretrained_s2G_path: str = "gsv-v2final-pretrained/s2G2333k.pth",
    opt_dir: str = None,
):
    try:

        if inp_text in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "inp_text is required"}
            )
        if inp_wav_dir in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "inp_wav_dir is required"}
            )
        if exp_name in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "exp_name is required"}
            )
        if opt_dir in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "opt_dir is required"}
            )

        preprocess_one_step(
            inp_text=inp_text,
            inp_wav_dir=inp_wav_dir,
            exp_name=exp_name,
            gpu_numbers1a=gpu_numbers1a,
            gpu_numbers1Ba=gpu_numbers1Ba,
            gpu_numbers1c=gpu_numbers1c,
            bert_pretrained_dir=pretrain_models_dir + bert_pretrained_dir,
            ssl_pretrained_dir=pretrain_models_dir + ssl_pretrained_dir,
            pretrained_s2G_path=pretrain_models_dir + pretrained_s2G_path,
            opt_dir=opt_dir,
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"preprocess failed", "Exception": str(e)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


"""
    s_batch_size = 11
    s_epoch = 8
    s_exp_name = exp_name
    s_text_low_lr_rate = 0.4
    s_if_save_latest = True
    s_if_save_every_weights = True
    s_save_every_n_epoch = 4
    s_gpunumbers = "0"
    s_pretrained_s2G = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    s_pretrained_s2D = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
    s_if_grad_ckpt = False
    s_lora_rank = 32
    s2_dir = f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/{s_exp_name}"

    start_sovits_training(s_batch_size,
                          s_epoch,
                          s_exp_name,
                          s_text_low_lr_rate,
                          s_if_save_latest,
                          s_if_save_every_weights,
                          s_save_every_n_epoch,
                          s_gpunumbers,
                          s_pretrained_s2G,
                          s_pretrained_s2D,
                          s_if_grad_ckpt,
                          s_lora_rank,
                          s2_dir)
"""


class SovitsTrainingRequest(BaseModel):
    batch_size: int
    epoch: int
    exp_name: str
    text_low_lr_rate: float
    if_save_latest: bool
    if_save_every_weights: bool
    save_every_n_epoch: int
    gpunumbers: str
    pretrained_s2G: str
    pretrained_s2D: str
    if_grad_ckpt: bool
    lora_rank: int
    opt_dir: str


@APP.post("/training/sovits")
async def sovits_training_endpoint(request: SovitsTrainingRequest):
    try:
        if request.exp_name in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "exp_name is required"}
            )
        if request.opt_dir in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "opt_dir is required"}
            )

        start_sovits_training(
            batch_size=request.batch_size,
            total_epoch=request.epoch,  # 🔁 Rename
            exp_name=request.exp_name,
            text_low_lr_rate=request.text_low_lr_rate,
            if_save_latest=request.if_save_latest,
            if_save_every_weights=request.if_save_every_weights,
            save_every_epoch=request.save_every_n_epoch,  # 🔁 Rename
            gpu_numbers1Ba=request.gpunumbers,  # 🔁 Rename
            pretrained_s2G=pretrain_models_dir + request.pretrained_s2G,
            pretrained_s2D=pretrain_models_dir + request.pretrained_s2D,
            if_grad_ckpt=request.if_grad_ckpt,
            lora_rank=request.lora_rank,
            s2_dir=request.opt_dir,  # 🔁 Rename
        )
    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"message": "sovits training failed", "Exception": str(e)},
        )


"""
    g_batch_size = 11
    g_epoch = 15
    g_exp_name = exp_name
    g_if_dpo = False
    g_if_save_latest = True
    g_if_save_every_weights = True
    g_save_every_n_epoch = 5
    g_gpu_numbers = "0"
    g_pretrained_s1 = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    s1_dir = f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/{s_exp_name}"

    start_gpt_training(g_batch_size,
                       g_epoch,
                       g_exp_name,
                       g_if_dpo,
                       g_if_save_latest,
                       g_if_save_every_weights,
                       g_save_every_n_epoch,
                       g_gpu_numbers,
                       g_pretrained_s1,
                       s1_dir)
"""


class GptTrainingRequest(BaseModel):
    batch_size: int
    epoch: int
    exp_name: str
    if_dpo: bool
    if_save_latest: bool
    if_save_every_weights: bool
    save_every_n_epoch: int
    gpu_numbers: str
    pretrained_s1: str
    opt_dir: Optional[str] = None


@APP.post("/training/gpt")
async def gpt_training_endpoint(request: GptTrainingRequest):
    try:
        if request.exp_name in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "exp_name is required"}
            )
        if request.opt_dir in [None, ""]:
            return JSONResponse(
                status_code=400, content={"message": "opt_dir is required"}
            )

        start_gpt_training(
            batch_size=request.batch_size,
            total_epoch=request.epoch,  # 🔁 Rename
            exp_name=request.exp_name,
            if_dpo=request.if_dpo,
            if_save_latest=request.if_save_latest,
            if_save_every_weights=request.if_save_every_weights,
            save_every_epoch=request.save_every_n_epoch,  # 🔁 Rename
            gpu_numbers=request.gpu_numbers,
            pretrained_s1=pretrain_models_dir + request.pretrained_s1,
            s1_dir=request.opt_dir,  # 🔁 Rename
        )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": "gpt training failed", "Exception": str(e)},
        )
    return JSONResponse(status_code=200, content={"message": "success"})


class GenerateRequest(BaseModel):
    gpt_model_path: str = None
    sovits_model_path: str = None
    ref_audio_path: str = None
    ref_text_path: str = None
    ref_language: str = None
    target_text_path: str = None
    target_language: str = None
    output_path: str = None
    output_file_name: str = None
    ref_free: bool = False


@APP.post("/generate_audio")
async def generate_audio_endpoint(request: GenerateRequest):
    try:
        data = {
            "gpt_model_path": request.gpt_model_path,
            "sovits_model_path": request.sovits_model_path,
            "ref_audio_path": request.ref_audio_path,
            "ref_text_path": request.ref_text_path,
            "ref_language": request.ref_language,
            "target_text_path": request.target_text_path,
            "target_language": request.target_language,
            "output_path": request.output_path,
            "output_file_name": request.output_file_name,
            "ref_free": request.ref_free,
        }
        print(data)
        generate_audio(
            gpt_model_path=now_dir + "\\GPT_weights_v3\\" + request.gpt_model_path,
            sovits_model_path=now_dir
            + "\\Sovits_weights\\"
            + request.sovits_model_path,
            ref_audio_path=request.ref_audio_path,
            ref_text_path=request.ref_text_path,
            ref_language=request.ref_language,
            target_text_path=request.target_text_path,
            target_language=request.target_language,
            output_path=request.output_path,
            output_file_name=request.output_file_name,
            ref_free=request.ref_free,
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": "generate audio failed", "Exception": str(e)},
        )


@APP.get("/get_model_list")
async def get_model_list():
    try:
        gpt_model_list = os.listdir(gpt_v3_model_dir)
        sovits_model_list = os.listdir(sovits_v3_model_dir)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": "get model list failed", "Exception": str(e)},
        )
    return JSONResponse(
        status_code=200,
        content={
            "gpt_model_list": gpt_model_list,
            "sovits_model_list": sovits_model_list,
        },
    )


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
