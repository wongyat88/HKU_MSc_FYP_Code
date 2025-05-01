import json
import os
import subprocess
import threading
from decouple import config


SOVITS_SERVER = config("SOVITS_SERVER")
FACE_DETECTION_PATH = config("FACE_DETECTION_PATH")


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


def process_face_detection(
    api_path,
    INPUT_DIR,
    phase1_dir,
    phase5_dir,
):
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_face_detection_thread,
        args=(
            api_path,
            INPUT_DIR,
            phase1_dir,
            phase5_dir,
        ),
    )
    thread.daemon = True
    thread.start()


def _process_face_detection_thread(api_path, INPUT_DIR, phase1_dir, phase5_dir):
    """
    1. Process the video to detect faces, and output 2 images for group. For user to select which one them is which speaker.
    2. Create a json for the selected images with the group ID and the selected image for each speaker and save to phase5_dir/target.json.
    """
    update_status(
        api_path,
        "phase5",
        False,
        "Start Face detection ...",
    )

    # Load the speaker list from the phase1_dir
    with open(f"{phase1_dir}/speaker_list.json", "r") as f:
        speaker_list = json.load(f)

    speaker_number = len(speaker_list)
    input_video = f"{INPUT_DIR}/original_video.mp4"
    output_json = f"{phase5_dir}/detection_output.json"
    batch_size = 256

    """
    python detect_v2.py --vid_input_path "F:\School\FYP\face_detection\input.mp4" --json_output_path "F:\School\FYP\face_detection\output.json" --yolo_batch_size 256 --n_speakers 2
    """

    # Get the Python executable from FACE_DETECTION_PATH's venv
    python_executable = os.path.join(
        FACE_DETECTION_PATH, "venv", "Scripts", "python.exe"
    )
    detect_script = os.path.join(FACE_DETECTION_PATH, "detect_v2.py")

    # Construct the command
    cmd = [
        python_executable,
        detect_script,
        "--vid_input_path",
        input_video,
        "--json_output_path",
        output_json,
        "--yolo_batch_size",
        str(batch_size),
        "--n_speakers",
        str(speaker_number),
    ]

    # Run the command with waiting for it to finish
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Face detection completed successfully.")
        print("Output:", result.stdout)

        update_status(
            api_path,
            "phase5",
            True,
            "Face detection completed successfully.",
        )
    except subprocess.CalledProcessError as e:
        print("Error in face detection:", e.stderr)
        raise


def process_final_generation(
    api_path,
    phase4_dir,
    phase5_dir,
):
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_final_generation_thread,
        args=(
            api_path,
            phase4_dir,
            phase5_dir,
        ),
    )
    thread.daemon = True
    thread.start()


def _process_final_generation_thread(api_path, phase4_dir, phase5_dir):
    """
    1. Combine the audio with the generated audio based on the final_result.json.
        * Create a new audio file based on each SPEAKER; The audio will contain silence with that timestamp the speaker is not speaking.
    2.
    """
    pass
