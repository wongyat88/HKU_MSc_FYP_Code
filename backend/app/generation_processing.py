import json
import os
import threading
from decouple import config
from pydub import AudioSegment
import requests

from funasr.utils.postprocess_utils import rich_transcription_postprocess

from app.audio_processing import get_asr_model


SOVITS_SERVER = config("SOVITS_SERVER")
server_url = SOVITS_SERVER + "/generate_audio"


def update_status(api_status_path, phase, is_complete, message, data=None):
    """Update the API status file for a specific phase"""
    with open(api_status_path, "r") as status_file:
        status = json.load(status_file)

    status[phase]["is_complete"] = is_complete
    status[phase]["message"] = message
    if data is not None:
        status[phase]["data"] = data

    with open(api_status_path, "w") as status_file:
        json.dump(status, status_file, indent=4)


"""
{
        "gpt_model_path": "F:\\School\\FYP2\\GPT-SoVITS-v3lora-20250228\\GPT_weights_v3\\SPEAKER_00-e15.ckpt",
        "sovits_model_path": "F:\\School\\FYP2\\GPT-SoVITS-v3lora-20250228\\SoVITS_weights_v3\\SPEAKER_00_e8_s64_l32.pth",
        "ref_audio_path": "F:\\School\\FYP2\\backend_frontend_ui\\backend\\public\\phase1\\SPEAKER\\SPEAKER_00_8.wav",
        "ref_text_path": "Well, we're going to do something with the border and very strong, very powerful, that'll be our first signal and our first signal to America that we're not playing games.",
        "ref_language": "英文",
        "target_text_path": "唔刻曬, 英文, 粤语",  # Ensure file is saved as UTF-8 for these characters
        "target_language": "粤语",  # Ensure file is saved as UTF-8 for these characters
        "output_path": "F:\\School\\FYP2",
        "ref_freeze": False,
    }
"""


def process_generation(data):
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_generation_thread,
        args=(data,),
    )
    thread.daemon = True
    thread.start()


def generate_random_string(length=10):
    """Generate a random string of fixed length"""
    import random
    import string

    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


def trim_audio(input_path, output_path, duration):
    """
    Trim the audio file to the specified duration.
    """

    # Load the audio file
    audio = AudioSegment.from_file(input_path)

    # Trim the audio to the specified duration
    trimmed_audio = audio[: duration * 1000]  # Convert seconds to milliseconds

    # Export the trimmed audio
    trimmed_audio.export(output_path, format="wav")


def _process_generation_thread(data):
    """
    Process the generation of translations.
    """

    for i in data:
        target_speaker = i["target_speaker"]
        gpt_model_path = i["gpt_model_path"]
        sovits_model_path = i["sovits_model_path"]
        ref_audio_path = i["ref_audio_path"]
        ref_text_path = i["ref_text_path"]
        ref_language = i["ref_language"]
        target_text_path = i["target_text_path"]
        target_language = i["target_language"]
        output_path = i["output_path"]
        output_file_name = i["output_file_name"]
        ref_freeze = i["ref_free"]
        api_status_path = i["api_status_path"]
        phase4_dir = i["phase4_dir"]
        ref_need_to_trim = i["ref_need_to_trim"]
        is_last_one = i["is_last_one"]

        if ref_need_to_trim is True:
            print("Trimming the reference audio ...")

            # Update the status to indicate completion
            update_status(
                api_status_path,
                "phase4",
                True,
                f"Trimming Audio for ref on {target_text_path} ...",
            )

            # Trim the reference audio
            new_audio_name = f"ref_audio_trimmed_{generate_random_string()}.wav"
            new_audio_path = os.path.join(phase4_dir, new_audio_name)
            trim_audio(ref_audio_path, new_audio_path, 9)

            ref_audio_path = new_audio_path

            # Do ASR on trim audio
            # Initialize the ASR model
            model = get_asr_model()

            try:
                # Process the segment with FunASR
                res = model.generate(
                    input=ref_audio_path,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )

                # Extract the transcription text
                text = rich_transcription_postprocess(res[0]["text"])
                ref_text_path = text
            except Exception as e:
                print(f"Error transcribing combined segment: {e}")
                text = "[Transcription error]"
                return

        # Update the status to indicate completion
        update_status(
            api_status_path,
            "phase4",
            True,
            f"Generating audio on {target_text_path} ...",
        )

        returnData = {
            "gpt_model_path": gpt_model_path,
            "sovits_model_path": sovits_model_path,
            "ref_audio_path": ref_audio_path,
            "ref_text_path": ref_text_path,
            "ref_language": ref_language,
            "target_text_path": target_text_path,
            "target_language": target_language,
            "output_path": output_path,
            "output_file_name": output_file_name,
            "ref_free": ref_freeze,
        }

        print("Sending request to server ...")
        print(returnData)

        try:
            response = requests.post(server_url, json=returnData)
            response.raise_for_status()

            if is_last_one is True:
                update_status(
                    api_status_path,
                    "phase4",
                    True,
                    f"Generation all completed successfully.",
                )
            else:
                update_status(
                    api_status_path,
                    "phase4",
                    False,
                    f"Generation completed ... Continue to next one.",
                )

        except Exception as e:
            print(f"Error: {e}")
            update_status(
                api_status_path,
                "phase4",
                False,
                f"Error: {e}",
            )
            return

    return {"output_dir": output_path}
