import json
import os
from pathlib import Path
import shutil
import threading
import traceback
from decouple import config
import librosa
import soundfile as sf
import pyrubberband as pyrb
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


def process_generation(data, phase3_dir, phase4_dir, single_update):
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_generation_thread,
        args=(
            data,
            phase3_dir,
            phase4_dir,
            single_update,
        ),
    )
    thread.daemon = True
    thread.start()


def process_respeed(id, speed, phase4_dir, API_STATUS_PATH):
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_respeed_thread,
        args=(
            id,
            speed,
            phase4_dir,
            API_STATUS_PATH,
        ),
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


def search_dict_and_add_update(data, tran_id, key, value):
    for item in data:
        if item["id"] == tran_id:
            item[key] = value
            break

    return data


def get_audio_length(audio_path):
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000  # Convert milliseconds to seconds


def calculate_speed_ratio(old_duration, new_duration):
    if new_duration == 0:
        raise ValueError("New duration cannot be zero. (calculate_speed_ratio)")

    speed_ratio = new_duration / old_duration
    return speed_ratio


def change_audio_speed(audio_path: str, speed_ratio: float):
    # * Remind that u need to install rubberband cli in your system, and add it to the PATH (Windows)
    if speed_ratio <= 0:
        raise ValueError("Speed ratio must be positive.")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Rubber Band time-stretch (keeps pitch constant)
    y_st = pyrb.time_stretch(y, sr, speed_ratio)

    # Write out
    sf.write(audio_path, y_st, sr)
    return audio_path


def _process_generation_thread(data, phase3_dir, phase4_dir, single_update):
    """
    Process the generation of translations.
    """

    # Read the translated_data.json file
    file_name = (
        "translated_data.json" if single_update is False else "final_result.json"
    )
    file_path = phase3_dir if single_update is False else phase4_dir
    transcriptions_path = os.path.join(file_path, file_name)
    with open(transcriptions_path, "r", encoding="utf-8") as translated_data:
        final_out_json = json.load(translated_data)

    for i in data:
        tran_id = i["tran_id"]
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
                False,
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
            False,
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

            # Read the audio length and add final_out_json
            new_audio_path = os.path.join(output_path, output_file_name)

            # duplicate the file with adding '_copy' to the name
            dir_path = os.path.dirname(new_audio_path)
            duplicate_audio_name = f"{output_file_name.split('.')[0]}_copy.wav"
            dst_path = os.path.join(dir_path, duplicate_audio_name)
            shutil.copy(new_audio_path, dst_path)

            audio_length = get_audio_length(new_audio_path)
            final_out_json = search_dict_and_add_update(
                final_out_json, tran_id, "translated_text", target_text_path
            )
            final_out_json = search_dict_and_add_update(
                final_out_json, tran_id, "generated_audio_duration", audio_length
            )

            # Get the original audio length and calculate the speed ratio
            for item in final_out_json:
                if item["id"] == tran_id:
                    # Must Exist, if not, it will cuz error
                    original_audio_duration = item["duration"]

            speed_ratio = calculate_speed_ratio(original_audio_duration, audio_length)
            final_out_json = search_dict_and_add_update(
                final_out_json, tran_id, "generated_audio_speed", speed_ratio
            )

            # Update the audio speed
            change_audio_speed(new_audio_path, speed_ratio)

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
            traceback.print_exc()
            update_status(
                api_status_path,
                "phase4",
                False,
                f"Error: {e}",
            )
            return

    # Save the updated final_out_json to the file
    # if single_update is True:
    #     print("Single update is true, so we will update the original json.")
    #     print(">>> final_out_json: ", final_out_json)
    #     # Get original final result json as single update must be happen after generated
    #     original_json_path = os.path.join(phase4_dir, "final_result.json")
    #     with open(original_json_path, "r", encoding="utf-8") as original_json:
    #         original_json = json.load(original_json)
    #         for item in original_json:
    #             for i in final_out_json:
    #                 if item["id"] == i["id"]:
    #                     print(">>> INSIDE the Loop: ", i)
    #                     item["generated_audio_duration"] = i["generated_audio_duration"]
    #                     item["generated_audio_speed"] = i["generated_audio_speed"]
    #                     item["translated_text"] = i["translated_text"]
    #                     break
    #         final_out_json = original_json

    output_json_dir = os.path.join(phase4_dir, "final_result.json")
    with open(output_json_dir, "w", encoding="utf-8") as final_output:
        json.dump(final_out_json, final_output, ensure_ascii=False, indent=4)

    return {"output_dir": output_path}


def _process_respeed_thread(id, speed, phase4_dir, api_status_path):
    """
    Process the generation of translations.
    """
    update_status(
        api_status_path,
        "phase4",
        False,
        f"Changing speed ...",
    )

    # Read the translated_data.json file
    result_dir = os.path.join(phase4_dir, "final_result.json")
    with open(result_dir, "r", encoding="utf-8") as translated_data:
        final_out_json = json.load(translated_data)

    tran_id = id
    speed_ratio = speed

    for i in final_out_json:
        if i["id"] == tran_id:
            # Must Exist, if not, it will cause error
            i["generated_audio_speed"] = speed_ratio

            # Construct full paths
            original_relative_path = i["file_path"]  # "SPEAKER\SPEAKER_01_14.wav"
            original_path = Path(phase4_dir) / original_relative_path

            # Get copy path (replace ".wav" with "_copy.wav")
            copy_path = original_path.with_name(original_path.stem + "_copy.wav")

            # Verify paths
            if not copy_path.exists():
                raise FileNotFoundError(f"Copy file not found: {copy_path}")
            if not original_path.parent.exists():
                raise FileNotFoundError(
                    f"SPEAKER directory not found: {original_path.parent}"
                )

            # Overwrite original with copy
            shutil.copy2(copy_path, original_path)

            # Apply speed change
            change_audio_speed(str(original_path), speed_ratio)
            break

    # Save the updated final_out_json to the file
    output_json_dir = os.path.join(phase4_dir, "final_result.json")
    with open(output_json_dir, "w", encoding="utf-8") as final_output:
        json.dump(final_out_json, final_output, ensure_ascii=False, indent=4)

    update_status(
        api_status_path,
        "phase4",
        True,
        f"Changing speed completed",
    )

    return {"output_dir": "phase4_dir"}
