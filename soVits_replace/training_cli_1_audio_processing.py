import os
import sys

from tools import my_utils
from tools.my_utils import load_audio, check_for_existance, check_details
from config import (
    python_exec,
    infer_device,
    is_half,
    exp_root,
    webui_port_main,
    webui_port_infer_tts,
    webui_port_uvr5,
    webui_port_subfix,
    is_share,
)
from subprocess import Popen


# Audio Slicing
ps_slice = []


def slice_audio(
    inp,
    opt_root,
    threshold,
    min_length,
    min_interval,
    hop_size,
    max_sil_kept,
    _max,
    alpha,
    n_parts,
):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if os.path.exists(inp) == False:
        print("Input path does not exist")
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        print("Input path exists but cannot be used")
        return
    if ps_slice == []:
        for i_part in range(n_parts):
            cmd = (
                '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s'
                ""
                % (
                    python_exec,
                    inp,
                    opt_root,
                    threshold,
                    min_length,
                    min_interval,
                    hop_size,
                    max_sil_kept,
                    _max,
                    alpha,
                    i_part,
                    n_parts,
                )
            )
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        print("Audio Slice: Start")
        for p in ps_slice:
            p.wait()
        ps_slice = []
        print("Audio Slice: Finished")
    else:
        print("Audio Slice: Failed")


# Denoising
p_denoise = None


def denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if p_denoise == None:
        denoise_inp_dir = my_utils.clean_path(denoise_inp_dir)
        denoise_opt_dir = my_utils.clean_path(denoise_opt_dir)
        check_for_existance([denoise_inp_dir])
        cmd = '"%s" tools/cmd-denoise.py -i "%s" -o "%s" -p %s' % (
            python_exec,
            denoise_inp_dir,
            denoise_opt_dir,
            "float16" if is_half == True else "float32",
        )

        print("Audio Denoising: Start")
        print(cmd)
        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise = None
        print("Audio Denoising: Finished")
    else:
        print("Audio Denoising: Failed")


# ASR
from tools.asr.config import asr_dict

p_asr = None


def open_asr(
    asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision
):
    global p_asr
    if p_asr is None:
        asr_inp_dir = my_utils.clean_path(asr_inp_dir)
        asr_opt_dir = my_utils.clean_path(asr_opt_dir)
        check_for_existance([asr_inp_dir])
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f" -s {asr_model_size}"
        cmd += f" -l {asr_lang}"
        cmd += f" -p {asr_precision}"
        # output_file_name = os.path.basename(asr_inp_dir)
        # output_folder = asr_opt_dir or "output/asr_opt"
        # output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.list')
        print("ASR: Start")
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None
        print("ASR: Finished")
    else:
        print("ASR: Failed")


if __name__ == "__main__":
    # Slice audio
    exp_dir = "Temp_20250422_v3"
    print("Slicing audio...")
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
    # Denoise
    print("Denoising audio...")
    denoise(
        denoise_inp_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/1_slice_audio_out",
        denoise_opt_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/2_denoise_out",
    )
    # ASR
    print("ASR...")
    open_asr(
        asr_inp_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/2_denoise_out",
        asr_opt_dir=f"L:/MSCCS Project/GPT-SoVITS/Experimental/{exp_dir}/3_asr_out",
        asr_model="Faster Whisper (多语种)",
        asr_model_size="large-v3",
        asr_lang="auto",
        asr_precision="int8",
    )
