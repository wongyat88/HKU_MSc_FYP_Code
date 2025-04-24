# -*- coding: utf-8 -*-

import argparse
import os
import soundfile as sf

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import (
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)

i18n = I18nAuto()


def synthesize(
    GPT_model_path,
    SoVITS_model_path,
    ref_audio_path,
    ref_text_path,
    ref_language,
    target_text_path,
    target_language,
    output_path,
    output_file_name,
    ref_free,
):
    # Read reference text
    # with open(ref_text_path, "r", encoding="utf-8") as file:
    # ref_text = file.read()
    ref_text = ref_text_path

    # Read target text
    # with open(target_text_path, "r", encoding="utf-8") as file:
    #     target_text = file.read()
    target_text = target_text_path

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=i18n(ref_language),
        text=target_text,
        text_language=i18n(target_language),
        top_p=1,
        temperature=1,
        ref_free=ref_free,
    )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]

        # Check if 'SPEAKER' folder is existing in the output path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_wav_path = os.path.join(output_path, output_file_name)
        print(">>> Output Path:", output_wav_path)
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument("--gpt_model", required=True, help="Path to the GPT model file")
    parser.add_argument(
        "--sovits_model", required=True, help="Path to the SoVITS model file"
    )
    parser.add_argument(
        "--ref_audio", required=True, help="Path to the reference audio file"
    )
    parser.add_argument(
        "--ref_text", required=True, help="Path to the reference text file"
    )
    parser.add_argument(
        "--ref_language",
        required=True,
        choices=["中文", "英文", "日文"],
        help="Language of the reference audio",
    )
    parser.add_argument(
        "--target_text", required=True, help="Path to the target text file"
    )
    parser.add_argument(
        "--target_language",
        required=True,
        choices=["中文", "英文", "日文", "粤语", "中英混合", "日英混合", "多语种混合"],
        help="Language of the target text",
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to the output directory"
    )

    args = parser.parse_args()

    synthesize(
        args.gpt_model,
        args.sovits_model,
        args.ref_audio,
        args.ref_text,
        args.ref_language,
        args.target_text,
        args.target_language,
        args.output_path,
    )


def generate_audio(
    gpt_model_path,
    sovits_model_path,
    ref_audio_path,
    ref_text_path,
    ref_language,
    target_text_path,
    target_language,
    output_path,
    output_file_name,
    ref_free,
):
    synthesize(
        gpt_model_path,
        sovits_model_path,
        ref_audio_path,
        ref_text_path,
        ref_language,
        target_text_path,
        target_language,
        os.path.dirname(output_path),
        output_file_name,
        ref_free,
    )


if __name__ == "__main__":
    # main()
    generate_audio(
        gpt_model_path="F:\\School\\FYP2\\GPT-SoVITS-v3lora-20250228\\GPT_weights_v3\\SPEAKER_00-e15.ckpt",
        sovits_model_path="F:\\School\\FYP2\\GPT-SoVITS-v3lora-20250228\\SoVITS_weights_v3\\SPEAKER_00_e8_s64_l32.pth",
        ref_audio_path="F:\\School\\FYP2\\backend_frontend_ui\\backend\\public\\phase1\\SPEAKER\\SPEAKER_00_8.wav",
        ref_text_path="Well, we're going to do something with the border and very strong, very powerful, that'll be our first signal and our first signal to America that we're not playing games.",
        ref_language="英文",
        target_text_path="唔刻曬, 英文, 粤语",  # Ensure file is saved as UTF-8 for these characters
        target_language="粤语",  # Ensure file is saved as UTF-8 for these characters
        output_path="F:\\School\\FYP2",
        output_file_name="output.wav",
    )
