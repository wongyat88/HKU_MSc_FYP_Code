import json
import os
import subprocess
import threading
from decouple import config
from pydub import AudioSegment
import numpy as np
from typing import Dict, Any, List
import cv2
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
)
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

SOVITS_SERVER = config("SOVITS_SERVER")
FACE_DETECTION_PATH = config("FACE_DETECTION_PATH")
WAV_2_LIP_PATH = config("WAV_2_LIP_PATH")


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
    input_dir,
    phase1_dir,
    phase4_dir,
    phase5_dir,
    face_detection_result_json,
):
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_final_generation_thread,
        args=(
            api_path,
            input_dir,
            phase1_dir,
            phase4_dir,
            phase5_dir,
            face_detection_result_json,
        ),
    )
    thread.daemon = True
    thread.start()


def combine_speaker_audios(segments, speakers, phase4_dir, output_dir):
    # figure out full conversation length (in ms)
    total_dur_s = max(item["end_time"] for item in segments)
    total_ms = int(total_dur_s * 1000)

    os.makedirs(output_dir, exist_ok=True)

    for spk in speakers:
        # start with a silent track of the full length
        combined = AudioSegment.silent(duration=total_ms)

        # overlay each of that speaker’s clips at the correct offset
        for item in segments:
            if item["speaker"] != spk:
                continue

            start_ms = int(item["start_time"] * 1000)
            clip_path = os.path.join(phase4_dir, item["file_path"])
            clip = AudioSegment.from_wav(clip_path)

            combined = combined.overlay(clip, position=start_ms)

        # export
        out_path = os.path.join(output_dir, f"{spk}.wav")
        combined.export(out_path, format="wav")
        print(f"→ {spk}: {out_path}")


# ---------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------


def _json_to_coord_lookup(
    detection_json: List[Dict[str, Any]],
) -> Dict[int, Dict[int, tuple]]:
    """
    Convert the raw detection_output.json structure into

        {frame_index: {cluster_group_id: (x1, y1, x2, y2), ...}, ...}

    Frame numbers in the JSON start at **1**, so we keep the same indexing.
    """
    lookup: Dict[int, Dict[int, tuple]] = {}
    for item in detection_json:
        f_idx = int(item["frame_number"])
        lookup.setdefault(f_idx, {})
        for face in item["faces"]:
            cid = int(face["prediction"])
            c = face["coordinates"]
            lookup[f_idx][cid] = (
                int(c["x1"]),
                int(c["y1"]),
                int(c["x2"]),
                int(c["y2"]),
            )
    return lookup


def _alpha_compose(
    bg: np.ndarray, fg: np.ndarray, mask: np.ndarray, top: int, left: int
) -> None:
    """
    Alpha-composite the foreground `fg` onto background `bg` at position (left, top),
    using the single-channel 8-bit mask `mask` (nonzero = keep fg).
    Operates in-place on `bg`.
    """
    # Region of interest in the background
    h, w = fg.shape[:2]
    roi = bg[top : top + h, left : left + w]

    # 1) Mask out the fg region in the ROI
    inv_mask = cv2.bitwise_not(mask)
    bg_cleared = cv2.bitwise_and(roi, roi, mask=inv_mask)

    # 2) Mask in the fg pixels
    fg_masked = cv2.bitwise_and(fg, fg, mask=mask)

    # 3) Add them together
    composite = cv2.add(bg_cleared, fg_masked)

    # 4) Write back into the background
    bg[top : top + h, left : left + w] = composite


# ---------------------------------------------------------------
#  main function
# ---------------------------------------------------------------


def merge_faces_and_audio(
    package_data: Dict[str, Dict[str, Any]],
    silent_video_path: str,
    instrumental_audio_path: str,
    output_video_path: str,
    detection_result_json: List[Dict[str, Any]],
) -> str:
    """
    Overlay every masked / lip‑synced face clip onto `silent_video_path`
    and add the mixed audio (all speakers + instrumental) in one pass.
    Returns the final video path.

    Parameters
    ----------
    package_data : dict
        {
            speaker_name: {
                "audio_path": path_to_wav,
                "video_path": path_to_masked_video,
                "cluster_group_id": int
            },
            ...
        }
    silent_video_path : str
        The original video **without** any audio track.
    instrumental_audio_path : str
        Demucs "no_vocals.wav" (background bed).
    output_video_path : str
        Where the finished movie will be written.
    detection_result_json : list[dict]
        Parsed JSON (already loaded with json.load).
    """
    # -----------------------------------------------------------
    #  preparation
    # -----------------------------------------------------------
    base_clip = VideoFileClip(silent_video_path)
    print(f"Base video: {base_clip.fps} fps, {base_clip.reader.fps} fps")
    output_fps = base_clip.reader.fps if hasattr(base_clip.reader, "fps") else 30.0
    # Ensure we have a valid FPS
    if output_fps is None:
        output_fps = 30.0  # Default fallback

    fps = output_fps
    width, height = base_clip.size
    total_frames = int(round(base_clip.duration * fps))

    coords = _json_to_coord_lookup(detection_result_json)

    # Open every masked‑face video with OpenCV for very quick random seek.
    # Keep a parallel VideoFileClip for its audio.
    speaker_caps = {}
    speaker_audio = []
    for speaker, data in package_data.items():
        cap = cv2.VideoCapture(data["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open masked video {data['video_path']}")
        speaker_caps[speaker] = {
            "cap": cap,
            "cid": int(data["cluster_group_id"]),
        }
        speaker_audio.append(AudioFileClip(data["audio_path"]))

    # -----------------------------------------------------------
    #  create silent video with faces composited in
    # -----------------------------------------------------------
    tmp_video_path = os.path.splitext(output_video_path)[0] + "_tmp_video.mp4"
    writer = FFMPEG_VideoWriter(
        tmp_video_path,
        size=(width, height),
        fps=fps,
        codec="libx264",
        # audiofile=False,
        preset="medium",
        bitrate="4000k",
    )

    for f_idx in range(total_frames):
        t = f_idx / fps
        # moviepy returns RGB, cv2 wants BGR – convert once
        frame = cv2.cvtColor(base_clip.get_frame(t), cv2.COLOR_RGB2BGR)

        # overlay every speaker that appears on this frame
        for sp, meta in speaker_caps.items():
            cid = meta["cid"]
            if (f_idx + 1) not in coords or cid not in coords[f_idx + 1]:
                continue

            x1, y1, x2, y2 = coords[f_idx + 1][cid]
            w_box, h_box = x2 - x1, y2 - y1

            cap: cv2.VideoCapture = meta["cap"]
            # Fast random seek to correct frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, face_frame = cap.read()
            if not ok:
                continue  # nothing to overlay

            # Resize masked‑face frame exactly to bounding box
            face_resized = cv2.resize(
                face_frame, (w_box, h_box), interpolation=cv2.INTER_LINEAR
            )

            # Build a binary mask – pixels that are *not* pure black
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            _alpha_compose(frame, face_resized, mask, y1, x1)

        # back to RGB for writer
        writer.write_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    writer.close()
    base_clip.close()
    for meta in speaker_caps.values():
        meta["cap"].release()

    # -----------------------------------------------------------
    #  mix audio (speakers + instrumental) and attach to video
    # -----------------------------------------------------------
    bed_audio = AudioFileClip(instrumental_audio_path)
    mixed_audio = CompositeAudioClip([bed_audio] + speaker_audio)
    mixed_audio = mixed_audio.set_duration(VideoFileClip(tmp_video_path).duration)

    (
        VideoFileClip(tmp_video_path)
        .set_audio(mixed_audio)
        .write_videofile(
            output_video_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            threads=4,
            preset="medium",
        )
    )

    # optional: tidy temp file
    os.remove(tmp_video_path)

    return output_video_path


def _process_final_generation_thread(
    api_path, input_dir, phase1_dir, phase4_dir, phase5_dir, face_detection_result_json
):
    """
    1. Combine the audio with the generated audio based on the final_result.json.
        * Create a new audio file based on each SPEAKER; The audio will contain silence with that timestamp the speaker is not speaking.
    2. Create masked video based on each speaker.
    3. Pass the masked video to wav2lip, return a lip synced video for each speaker with new audio.
    4. Remove all target voices from the original video via bytedance/music_source_separation
    5. Combine the processed masked video together on top of the original video
    6. Done and return the final video path.
    """

    update_status(
        api_path,
        "phase6",
        False,
        "Start combining generated audio ...",
    )

    # Load the final result JSON file
    with open(f"{phase4_dir}/final_result.json", "r", encoding="utf-8") as f:
        final_result = json.load(f)

    # Load the speaker list from the phase1_dir
    with open(f"{phase1_dir}/speaker_list.json", "r", encoding="utf-8") as f:
        speaker_list = json.load(f)

    # Loop the speaker_list to combine the audio from phase4_dir and the generated audio from phase5_dir
    combine_speaker_audios(final_result, speaker_list, phase4_dir, phase5_dir)

    # Create masked video based on each speaker
    update_status(
        api_path,
        "phase6",
        False,
        f"Start creating masked video for speakers ...",
    )

    cluster_mapping = {}
    index = 0
    for i in range(len(speaker_list)):
        cluster_mapping[str(index)] = [str(index)]
        index += 1

    print(f"Cluster mapping: {cluster_mapping}")

    # Create a masked video for each speaker
    python_executable = os.path.join(
        FACE_DETECTION_PATH, "venv", "Scripts", "python.exe"
    )
    mask_script = os.path.join(FACE_DETECTION_PATH, "mask_video_v2.py")

    cluster_json = json.dumps(cluster_mapping)
    mask_cmd = [
        python_executable,
        mask_script,
        "--frame_face_info_json",
        os.path.join(phase5_dir, "detection_output.json"),
        "--output_dir",
        phase5_dir,
        "--cluster_mapping",
        f"{cluster_json}",
    ]

    # * Uncomment the following line to run the command
    # subprocess.run(mask_cmd, check=True)

    """
    Key will be the speaker, value will be cluster group id
    face_detection_result_json = { "SPEAKER_01": 0, "SPEAKER_00": 1 }
    """
    package_data = {}
    # Call wav2lip to generate the lip synced video for each speaker
    for i in range(len(speaker_list)):
        # Get the speaker name and cluster group id from the face_detection_result_json
        speaker_name = speaker_list[i]

        update_status(
            api_path,
            "phase6",
            False,
            f"Start lip sync on {speaker_name} ...",
        )

        print(face_detection_result_json)

        cluster_group_id = face_detection_result_json[speaker_name]

        # Get the audio path for the speaker
        audio_path = os.path.join(phase5_dir, f"{speaker_name}.wav")

        # Get the video path for the speaker
        video_path = os.path.join(phase5_dir, f"masked_vid_{cluster_group_id}.mp4")

        package_data[speaker_name] = {
            "audio_path": audio_path,
            "video_path": video_path,
            "cluster_group_id": cluster_group_id,
        }

        # Call wav2lip to generate the lip synced video for each speaker
        """
        python inference.py --checkpoint_path wav2lip_gan.pth --face "F:\School\FYP2\backend_frontend_ui\backend\public\phase5\masked_vid_1.mp4" --audio "F:\School\FYP2\backend_frontend_ui\backend\public\phase5\SPEAKER_01.wav" --outfile "F:\School\FYP2\backend_frontend_ui\backend\public\phase5\SPEAKER_01.mp4"
        """
        python_wav_executable = os.path.join(
            WAV_2_LIP_PATH, "venv", "Scripts", "python.exe"
        )
        inference_script = os.path.join(WAV_2_LIP_PATH, "inference.py")
        wav2lip_model = os.path.join(WAV_2_LIP_PATH, "wav2lip.pth")

        wav2lip_cmd = [
            python_wav_executable,
            inference_script,
            "--checkpoint_path",
            wav2lip_model,
            "--face",
            video_path,
            "--audio",
            audio_path,
            "--outfile",
            os.path.join(phase5_dir, f"{speaker_name}.mp4"),
        ]

        # * Uncomment the following line to run the command
        # subprocess.run(
        #     wav2lip_cmd,
        #     check=True,
        #     cwd=WAV_2_LIP_PATH,
        # )

    # Remove vocals from original wav file
    update_status(
        api_path,
        "phase6",
        False,
        "Start Removing vocals from original audio ...",
    )
    main_audio_path = os.path.join(phase1_dir, "main_audio.wav")
    """
    demucs -d cuda test.mp3 --two-stems=vocals -o ./output_folder --filename "{stem}.{ext}"

    # Use the `no_vocals.wav`, remind that there will be a sub-folder also created `htdemucs`
    """
    demucs_cmd = [
        "demucs",
        "-d",
        "cuda",
        main_audio_path,
        "--two-stems=vocals",
        "-o",
        phase5_dir,
        "--filename",
        "{stem}.{ext}",
    ]

    # * Uncomment the following line to run the command
    # subprocess.run(demucs_cmd, check=True)

    # Get the path to the no_vocals.wav file
    no_vocals_path = os.path.join(phase5_dir, "htdemucs", "no_vocals.wav")

    # Get the path to the original video
    original_video_path = os.path.join(input_dir, "original_video.mp4")

    # Create a original video without sound
    original_video_no_sound_path = os.path.join(
        phase5_dir, "original_video_no_sound.mp4"
    )
    original_video = VideoFileClip(original_video_path)
    original_video = original_video.set_audio(None)

    print(f"Original video: {original_video.fps} fps, {original_video.reader.fps} fps")
    output_fps = (
        original_video.reader.fps if hasattr(original_video.reader, "fps") else 30.0
    )
    # Ensure we have a valid FPS
    if output_fps is None:
        output_fps = 30.0  # Default fallback

    original_video.write_videofile(
        original_video_no_sound_path,
        codec="libx264",
        audio_codec="aac",
        fps=output_fps,
    )
    original_video.close()

    # Get the path of the detection result JSON file
    detection_result_json_path = os.path.join(phase5_dir, "detection_output.json")
    with open(detection_result_json_path, "r", encoding="utf-8") as f:
        detection_result_json = json.load(f)

    # Get the path to the output video
    output_video_path = os.path.join(phase5_dir, "final_video.mp4")

    # Extract the processed masked video for each speaker and merge to the original video
    update_status(
        api_path,
        "phase6",
        False,
        "Start Merging processed masked video ...",
    )

    merge_faces_and_audio(
        package_data,
        original_video_no_sound_path,
        no_vocals_path,
        output_video_path,
        detection_result_json,
    )

    update_status(
        api_path,
        "phase6",
        True,
        "Generating Completed",
    )
