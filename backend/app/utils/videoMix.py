import json
from tqdm import tqdm

# Frame extraction
import subprocess
import os

# Frame mixing
import cv2
import numpy as np


def update_status(api_status_path, phase, is_complete, message, data=None):
    if not os.path.exists(api_status_path):
        pass
    """Update the API status file for a specific phase"""
    with open(api_status_path, "r") as status_file:
        status = json.load(status_file)

    status[phase]["is_complete"] = is_complete
    status[phase]["message"] = message
    if data is not None:
        status[phase]["data"] = data

    with open(api_status_path, "w") as status_file:
        json.dump(status, status_file, indent=4)


def extract_frames_from_video(video_path, frame_store_path, method="ffmpeg"):
    os.makedirs(frame_store_path, exist_ok=True)
    if method == "ffmpeg":  # ffmpeg (fast)
        frame_store_format = f"{frame_store_path}/frame_%04d.jpg"
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-q:v", "2", frame_store_format, "-y"]
        )
    print(
        f"Frame extraction finished. Total frames: {len(os.listdir(frame_store_path))}"
    )


def mix_video_and_masks(
    original_video_frames_path, masks_video_frames_paths, output_frame_save_path
):
    os.makedirs(output_frame_save_path, exist_ok=True)
    list_of_frames = os.listdir(original_video_frames_path)
    for frame_name in tqdm(list_of_frames):
        # frame_name = "frame_0001.jpg"
        orig_frame = cv2.imread(f"{original_video_frames_path}/{frame_name}")
        orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
        for mask_frame_path in masks_video_frames_paths:
            if os.path.exists(f"{mask_frame_path}/{frame_name}"):
                mask_frame = cv2.imread(f"{mask_frame_path}/{frame_name}")
                mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB)
                overlay_mask = np.all(mask_frame < 30, axis=-1)
                overlay_mask = ~overlay_mask
                orig_frame[overlay_mask] = mask_frame[overlay_mask]
        orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_frame_save_path}/{frame_name}", orig_frame)


def ffmpeg_mix_n_audios(
    video_paths: list[str], output_path: str, volumes: list[float] = None
):
    if volumes is None:
        volumes = [1.0] * len(video_paths)
    inputs = []
    filters = []
    # Build input and filter chains
    for i, (path, vol) in enumerate(zip(video_paths, volumes)):
        inputs.extend(["-i", path])
        filters.append(f"[{i}:a]volume={vol}[a{i}]")
    # Join all audio streams
    filter_str = (
        ";".join(filters)
        + f';{"".join(f"[a{i}]" for i in range(len(video_paths)))}amix=inputs={len(video_paths)}:duration=longest'
    )
    command = [
        "ffmpeg",
        *inputs,
        "-filter_complex",
        filter_str,
        "-c:a",
        "libmp3lame",  # MP3 codec
        "-q:a",
        "2",  # Quality (0-9, 0=best)
        "-y",  # Overwrite without asking
        output_path,
    ]
    subprocess.run(command)


def ffmpeg_create_video(frames_dir, output_path, fps=30, pattern="frame_%04d.jpg"):
    command = [
        "ffmpeg",
        "-y",  # Overwrite without asking
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frames_dir, pattern),
        "-c:v",
        "libx264",  # H.264 codec
        "-pix_fmt",
        "yuv420p",  # Standard pixel format
        "-crf",
        "28",  # Quality (0-51, lower is better)
        "-y",  # Overwrite without asking
        output_path,
    ]
    subprocess.run(command)
    print(f"Video created at {output_path}")


def mix_video_audio(video_path, audio_path, output_path, volume=1.0):
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-filter_complex",
        f"[1:a]volume={volume}[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-y",  # Overwrite without asking
        output_path,
    ]
    subprocess.run(cmd)


def mix_video(original_video, mask_videos, master_save_dir, api_path, other_sound_path):
    # original_video = "L:/MSCCS Project/HKU_MSc_FYP/video_mixing/phase5/original_video_no_sound.mp4"
    # mask_videos = ["L:/MSCCS Project/HKU_MSc_FYP/video_mixing/phase5/SPEAKER_00.mp4",
    #               "L:/MSCCS Project/HKU_MSc_FYP/video_mixing/phase5/SPEAKER_01.mp4"]
    # master_save_dir = "L:/MSCCS Project/HKU_MSc_FYP/video_mixing/temp"
    # other_sound_path = "xxxx.wav"

    # Extract frames
    print("Extract frames from original video...")
    update_status(
        api_path,
        "phase6",
        False,
        "Start Extract frames from original video...",
    )
    original_video_frames_path = f"{master_save_dir}/original_video"
    extract_frames_from_video(original_video, original_video_frames_path)

    print("Extract frames from masked videos...")
    masks_video_frames_paths = []
    update_status(
        api_path,
        "phase6",
        False,
        "Start Extract frames from masked videos ...",
    )
    for mask_video in mask_videos:
        mask_name = mask_video.split("\\")[-1].replace(".mp4", "")
        masks_video_frames_paths.append(f"{master_save_dir}/{mask_name}")
        print(f">>>> {master_save_dir}/{mask_name}")
        extract_frames_from_video(mask_video, f"{master_save_dir}/{mask_name}")

    # Overlay frames
    print("Overlay frames...")
    update_status(
        api_path,
        "phase6",
        False,
        "Overlay frames...",
    )
    output_frame_save_path = f"{master_save_dir}/mixed_output"
    mix_video_and_masks(
        original_video_frames_path, masks_video_frames_paths, output_frame_save_path
    )

    # Create video from frames
    update_status(
        api_path,
        "phase6",
        False,
        "Create video from overlayed frames...",
    )
    print("Create video from overlayed frames...")
    output_video_path = f"{master_save_dir}/mixed_video.mp4"
    ffmpeg_create_video(output_frame_save_path, output_video_path)

    # Audio blending
    update_status(
        api_path,
        "phase6",
        False,
        "Blend audio...",
    )
    print("Blend audio...")
    output_audio_path = f"{master_save_dir}/mixed_audio_raw.mp3"
    ffmpeg_mix_n_audios(mask_videos, output_audio_path)

    # The original_video has audio, so we also need to mix it with the blended audio
    final_output_audio_path = f"{master_save_dir}/mixed_audio.mp3"
    # Combine the original video audio with the blended audio
    # other_sound_path is the path to the other sound file(wav) that need to mix with the mixed_audio_raw.mp3
    command = [
        "ffmpeg",
        "-y",  # Overwrite the output file without asking
        "-i",
        output_audio_path,  # Input first audio file
        "-i",
        other_sound_path,  # Input second audio file
        "-filter_complex",
        "[0][1]amix=inputs=2",  # Mix both inputs
        "-ac",
        "2",  # Set the output audio channels (stereo)
        "-ar",
        "44100",  # Set the sample rate to 44.1 kHz
        final_output_audio_path,  # Output path for the mixed audio
    ]

    # Run the command
    subprocess.run(command, check=True)

    # Final output
    update_status(
        api_path,
        "phase6",
        False,
        "Produce final video...",
    )
    print("Produce final video...")
    final_path = f"{master_save_dir}/final_video.mp4"
    mix_video_audio(output_video_path, final_output_audio_path, final_path)

    print(f"Done! Video saved at {final_path}")


if __name__ == "__main__":
    original_video = "L:/MSCCS Project/HKU_MSc_FYP/video_mixing/phase5/original_video_no_sound.mp4"  # Original video
    mask_videos = [
        "L:/MSCCS Project/HKU_MSc_FYP/video_mixing/phase5/SPEAKER_00.mp4",
        "L:/MSCCS Project/HKU_MSc_FYP/video_mixing/phase5/SPEAKER_01.mp4",
    ]  # A list contains the directories of all lip-synced video (should contain translated audio as well)
    master_save_dir = "L:/MSCCS Project/HKU_MSc_FYP/video_mixing/temp"  # An empty folder for storing intermediate output
    mix_video(original_video, mask_videos, master_save_dir, "")
