import os
import json
import requests
import torch
import subprocess
import threading
from pathlib import Path
from decouple import config
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment
from pydub.silence import split_on_silence
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from app.utils.tools import recreate_folder

# Path to the SenseVoiceSmall model
# Get the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent
# Path to the SenseVoiceSmall model relative to the script directory
MODEL_DIR = SCRIPT_DIR / "models" / "SenseVoiceSmall"

SOVITS_SERVER = config("SOVITS_SERVER")


# Initialize FunASR model - this will be lazy loaded when needed
def get_asr_model():
    return AutoModel(
        model=MODEL_DIR,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )


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


def extract_audio(video_file, output_audio_file):
    """Convert MP4 video to WAV audio using FFmpeg"""
    try:
        # Use subprocess to call ffmpeg directly
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_file,
                "-y",  # Overwrite output file if it exists
                "-vn",  # Disable video
                "-acodec",
                "pcm_s16le",  # Output codec
                "-ar",
                "16000",  # Sample rate
                "-ac",
                "1",  # Mono audio
                output_audio_file,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from video: {e}")
        return False


def run_diarization(audio_file):
    """Perform speaker diarization using pyannote-audio"""
    try:
        # Load the pre-trained model
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        # Use GPU if available
        print(
            ">>> run_diarization Using GPU for processing..."
            if torch.cuda.is_available()
            else ">>> run_diarization Using CPU for processing..."
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        # Apply the diarization model to the audio
        with ProgressHook() as hook:
            diarization = pipeline(audio_file, hook=hook)

        return diarization
    except Exception as e:
        print(f"Error in speaker diarization: {e}")
        return None


def separate_audio_by_speaker(audio_file, diarization, output_folder):
    """Extract and save audio segments by speaker"""
    # Load the audio file
    audio = AudioSegment.from_wav(audio_file)

    # Dictionary to organize transcriptions by speaker
    speakers_data = []

    # Create a folder name 'SPEAKER', if exist, clear it
    speaker_dir = "SPEAKER"
    speaker_folder = os.path.join(output_folder, "SPEAKER")
    recreate_folder(speaker_folder)

    # Loop through each speaker's segments
    speaker_set = set()
    id = 1
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start * 1000  # Convert to milliseconds
        end_time = turn.end * 1000  # Convert to milliseconds

        # Extract the segment for the current speaker
        speaker_audio = audio[start_time:end_time]

        # Generate filename for the segment
        # speaker_filename = f"{speaker_dir}_{turn.start:.1f}-{turn.end:.1f}.wav"
        # ! Use id to name the file as too many '.' exist in the file name may cause issue
        speaker_filename = f"{speaker}_{id}.wav"
        file_path = os.path.join(speaker_folder, speaker_filename)

        # Save the audio segment to file
        # ! No need to split it to small pieces
        speaker_audio.export(file_path, format="wav")

        # Store segment information
        segment_info = {
            "id": id,
            "speaker": speaker,
            "start_time": turn.start,
            "end_time": turn.end,
            "duration": turn.end - turn.start,
            "file_path": os.path.join(speaker_dir, speaker_filename),
        }
        speakers_data.append(segment_info)
        speaker_set.add(speaker)
        id += 1

    # Save the speaker list to a json file
    speaker_list_path = os.path.join(output_folder, "speaker_list.json")
    with open(speaker_list_path, "w") as f:
        json.dump(list(speaker_set), f, indent=4)

    return speakers_data


def transcribe_audio_segments(speakers_data, output_folder):
    """Transcribe audio segments using FunASR"""
    # Initialize the ASR model
    model = get_asr_model()

    # Dictionary to store transcriptions
    transcriptions = []

    # Process each speaker's segments
    for data in speakers_data:
        # Get the full path to the audio segment
        segment_path = os.path.join(output_folder, data["file_path"])

        try:
            # Process the segment with FunASR
            res = model.generate(
                input=segment_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )

            # Extract the transcription text
            text = rich_transcription_postprocess(res[0]["text"])

            # Add transcription to the segment data
            segment_with_text = data.copy()
            segment_with_text["text"] = text
            transcriptions.append(segment_with_text)

        except Exception as e:
            print(f"Error transcribing segment {segment_path}: {e}")
            segment_with_text = data.copy()
            segment_with_text["text"] = "[Transcription error]"
            transcriptions.append(segment_with_text)

    return transcriptions


def process_video(video_path, output_folder, api_status_path):
    """Main function to process a video file in a separate thread"""
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_video_thread, args=(video_path, output_folder, api_status_path)
    )
    thread.daemon = True
    thread.start()


def _process_video_thread(video_path, output_folder, api_status_path):
    """Internal function that runs in a separate thread to process the video"""
    try:
        # Make sure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Set status to processing
        update_status(api_status_path, "phase1", False, "Converting video to audio...")

        # Step 1: Convert MP4 to WAV
        audio_file = os.path.join(output_folder, "main_audio.wav")
        if not extract_audio(video_path, audio_file):
            update_status(
                api_status_path, "phase1", False, "Error converting video to audio"
            )
            return

        # * Should remove bg from the audio first
        # Remove vocals from original wav file
        update_status(
            api_status_path,
            "phase1",
            False,
            "Start Removing background sound from original audio ...",
        )
        """
        demucs -d cuda test.mp3 --two-stems=vocals -o ./output_folder --filename "{stem}.{ext}"

        # Use the `no_vocals.wav`, remind that there will be a sub-folder also created `htdemucs`
        """
        demucs_cmd = [
            "demucs",
            "-d",
            "cuda",
            audio_file,
            "--two-stems=vocals",
            "-o",
            output_folder,
            "--filename",
            "{stem}.{ext}",
        ]

        # * Uncomment the following line to run the command
        subprocess.run(demucs_cmd, check=True)

        # Get the path to the no_vocals.wav file
        audio_file = os.path.join(output_folder, "htdemucs", "vocals.wav")

        # Step 2: Perform speaker diarization
        update_status(
            api_status_path, "phase1", False, "Performing speaker diarization..."
        )
        diarization = run_diarization(audio_file)
        if diarization is None:
            update_status(
                api_status_path, "phase1", False, "Error in speaker diarization"
            )
            return

        # Step 3: Separate audio by speaker
        update_status(
            api_status_path, "phase1", False, "Separating audio by speaker..."
        )
        speakers_data = separate_audio_by_speaker(
            audio_file, diarization, output_folder
        )

        transcriptions_path = os.path.join(output_folder, "transcriptions.json")
        with open(transcriptions_path, "w") as f:
            json.dump(speakers_data, f, indent=4)

        # Step 4: Transcribe each audio segment
        update_status(
            api_status_path, "phase1", False, "Transcribing audio segments..."
        )

        transcriptions = transcribe_audio_segments(speakers_data, output_folder)

        # Step 5: Save transcriptions to JSON
        transcriptions_path = os.path.join(output_folder, "transcriptions.json")
        with open(transcriptions_path, "w") as f:
            json.dump(transcriptions, f, indent=4)

        # Update status to complete
        update_status(
            api_status_path, "phase1", True, "Processing complete", transcriptions
        )

    except Exception as e:
        print(f"Error processing video: {e}")
        update_status(api_status_path, "phase1", False, f"Error: {str(e)}")


def find_audio_segments_by_ids(transcriptions_data, segment_ids):
    """Find audio segments by their IDs in the list of transcriptions"""
    segments_to_combine = []
    segment_speakers = []

    # Create a dictionary to store segments by their ID for faster lookup
    segment_dict = {}
    for segment in transcriptions_data:
        segment_dict[segment["id"]] = {
            "segment": segment,
            "speaker": segment["speaker"],
        }

    # Add segments in the order specified by segment_ids
    for segment_id in segment_ids:
        if segment_id in segment_dict:
            segments_to_combine.append(segment_dict[segment_id])
            segment_speakers.append(segment_dict[segment_id]["speaker"])

    return segments_to_combine, segment_speakers


def combine_audio_segments(segments_to_combine, output_folder):
    """Combine multiple audio segments into a single audio file"""
    if not segments_to_combine:
        return None, None, None

    # Use the first segment's speaker as the target speaker
    first_segment = segments_to_combine[0]
    target_speaker = first_segment["speaker"]

    # Create a combined audio segment
    combined_audio = AudioSegment.empty()

    # Track the start and end times
    start_time = float("inf")
    end_time = 0
    total_duration = 0

    # Combine the audio segments in the order they were provided
    for item in segments_to_combine:
        segment = item["segment"]
        segment_path = os.path.join(output_folder, segment["file_path"])

        # Check if file exists
        if not os.path.exists(segment_path):
            print(f"Warning: Audio file not found: {segment_path}")
            continue

        # Load the audio segment
        audio = AudioSegment.from_wav(segment_path)

        # Add to the combined audio
        combined_audio += audio

        # Update timing information
        if segment["start_time"] < start_time:
            start_time = segment["start_time"]
        if segment["end_time"] > end_time:
            end_time = segment["end_time"]

        # Add to total duration
        total_duration += segment["duration"]

    if combined_audio.duration_seconds == 0:
        return None, None, None

    return (
        combined_audio,
        target_speaker,
        {"start_time": start_time, "end_time": end_time, "duration": total_duration},
    )


def transcribe_combined_audio(
    audio_segment, speaker, timing_info, output_folder, next_id
):
    """Transcribe the combined audio segment"""
    # Create filename based on start and end times
    filename = (
        # f"{speaker}_{timing_info['start_time']:.1f}-{timing_info['end_time']:.1f}.wav"
        f"{speaker}_{next_id}.wav"
    )

    # Ensure speaker folder exists
    speaker_folder = os.path.join(output_folder, "SPEAKER")
    # if not os.path.exists(speaker_folder):
    #     os.makedirs(speaker_folder)

    # Save combined audio to file
    file_path = os.path.join(speaker_folder, filename)
    audio_segment.export(file_path, format="wav")

    # Initialize the ASR model
    model = get_asr_model()

    try:
        # Process the segment with FunASR
        res = model.generate(
            input=file_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )

        # Extract the transcription text
        text = rich_transcription_postprocess(res[0]["text"])
    except Exception as e:
        print(f"Error transcribing combined segment: {e}")
        text = "[Transcription error]"

    # Create segment data
    segment_data = {
        "id": next_id,
        "speaker": speaker,
        "start_time": timing_info["start_time"],
        "end_time": timing_info["end_time"],
        "duration": timing_info["duration"],
        "file_path": os.path.join("SPEAKER", filename),
        "text": text,
    }

    return segment_data


def process_audio_combination(segment_ids, output_folder, api_status_path):
    """Process the audio combination in a separate thread"""
    # Start a thread to process the combination asynchronously
    thread = threading.Thread(
        target=_process_audio_combination_thread,
        args=(segment_ids, output_folder, api_status_path),
    )
    thread.daemon = True
    thread.start()
    return thread


def _process_audio_combination_thread(segment_ids, output_folder, api_status_path):
    """Internal function that runs in a separate thread to combine audio segments"""
    try:
        # Update status to processing
        update_status(
            api_status_path, "combine", False, "Starting audio combination..."
        )

        # Load transcriptions
        transcriptions_path = os.path.join(output_folder, "transcriptions.json")
        with open(transcriptions_path, "r") as f:
            transcriptions = json.load(f)

        # Find segments to combine
        segments_to_combine, segment_speakers = find_audio_segments_by_ids(
            transcriptions, segment_ids
        )

        if not segments_to_combine:
            update_status(
                api_status_path, "combine", False, "No matching segments found"
            )
            return None

        update_status(api_status_path, "combine", False, "Combining audio segments...")

        # Combine audio segments
        combined_audio, target_speaker, timing_info = combine_audio_segments(
            segments_to_combine, output_folder
        )

        if combined_audio is None:
            update_status(
                api_status_path, "combine", False, "Failed to combine audio segments"
            )
            return None

        # Find the next available ID
        next_id = 1
        for segment in transcriptions:
            next_id = max(next_id, segment["id"] + 1)

        # Transcribe the combined audio
        update_status(
            api_status_path, "combine", False, "Transcribing combined audio..."
        )
        new_segment = transcribe_combined_audio(
            combined_audio, target_speaker, timing_info, output_folder, next_id
        )

        if new_segment is None:
            update_status(
                api_status_path, "combine", False, "Failed to transcribe combined audio"
            )
            return None

        # Add the new segment to the transcriptions data and
        # remove the original segments that were combined
        updated_transcriptions = []

        # Add all segments that weren't combined
        for segment in transcriptions:
            if segment["id"] not in segment_ids:
                updated_transcriptions.append(segment)

        # Add the new combined segment
        updated_transcriptions.append(new_segment)

        # Sort segments by start time to maintain chronological order
        updated_transcriptions.sort(key=lambda x: x["start_time"])

        # Remove the original audio files
        for item in segments_to_combine:
            segment = item["segment"]
            segment_path = os.path.join(output_folder, segment["file_path"])
            if os.path.exists(segment_path):
                os.remove(segment_path)

        # Save the updated transcriptions
        with open(transcriptions_path, "w") as f:
            json.dump(updated_transcriptions, f, indent=4)

        # Update status to complete
        update_status(
            api_status_path,
            "combine",
            True,
            "Audio combination complete",
            {"new_segment": new_segment, "removed_segments": segment_ids},
        )

        return new_segment

    except Exception as e:
        print(f"Error combining audio segments: {e}")
        update_status(api_status_path, "combine", False, f"Error: {str(e)}")
        return None


def process_delete_audio(segment_ids, output_folder, api_status_path):
    """Delete audio segments based on their IDs"""
    try:
        # Load transcriptions
        transcriptions_path = os.path.join(output_folder, "transcriptions.json")
        with open(transcriptions_path, "r") as f:
            transcriptions = json.load(f)

        # Find segments to delete
        deleted_segments = []
        updated_transcriptions = []

        # Create ID set for faster lookup
        segment_id_set = set(segment_ids)

        # Keep segments that aren't being deleted
        for segment in transcriptions:
            if segment["id"] not in segment_id_set:
                updated_transcriptions.append(segment)
            else:
                # Add to deleted segments list
                deleted_segments.append(segment)

                # Remove the audio file
                segment_path = os.path.join(output_folder, segment["file_path"])
                if os.path.exists(segment_path):
                    os.remove(segment_path)

        # Save the updated transcriptions
        with open(transcriptions_path, "w") as f:
            json.dump(updated_transcriptions, f, indent=4)

        return deleted_segments

    except Exception as e:
        print(f"Error deleting audio segments: {e}")
        update_status(api_status_path, "delete", False, f"Error: {str(e)}")
        return None


def process_save_segments(data, output_folder):
    """data is a json file, save as in work_dir/phase2/final_input.json"""
    try:
        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the segments data to a JSON file
        segments_path = os.path.join(output_folder, "final_input.json")

        # Create file if it doesn't exist (the open() with "w" mode will create the file,
        # but we'll ensure the directory exists)
        os.makedirs(os.path.dirname(segments_path), exist_ok=True)

        with open(segments_path, "w") as f:
            json.dump(data, f, indent=4)

        return {"message": "Segments saved successfully", "path": segments_path}

    except Exception as e:
        print(f"Error saving segments: {e}")
        raise Exception(f"Error saving segments: {e}")


def process_preprogessing(phase1_path, data_json_path, api_status_path):
    """Main function to process a video file in a separate thread"""
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_data_preprogessing,
        args=(phase1_path, data_json_path, api_status_path),
    )
    thread.daemon = True
    thread.start()


def _process_data_preprogessing(output_folder, data_json_path, api_status_path):
    # * New feature: Create Copy of the SPEAKER and prepare training data for SoVits
    update_status(
        api_status_path,
        "phase2",
        False,
        "Creating Dataset for SoVits... Slicing audio...",
    )
    """
    ! New Idea:
    - Copy all the content on "SPEAKER" to new folder name "SPEAKER_DATASET"
    - Combine them based on speaker name via the json data
    """

    # Set a default speaker list if needed (example: two speakers)
    speaker_list = {"SPEAKER_00", "SPEAKER_01"}

    with open(data_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Avoid crashing, copy all the content on "SPEAKER" to new folder name "SPEAKER_DATASET"
    speaker_dataset_folder = os.path.join(output_folder, "SPEAKER_DATASET")
    recreate_folder(speaker_dataset_folder)
    # Copy all files from SPEAKER to SPEAKER_DATASET
    speaker_folder = os.path.join(output_folder, "SPEAKER")
    if os.path.exists(speaker_folder):
        for file in os.listdir(speaker_folder):
            source_path = os.path.join(speaker_folder, file)
            dest_path = os.path.join(speaker_dataset_folder, file)
            if os.path.isfile(source_path):
                # Copy the file to the new folder
                with open(source_path, "rb") as src_file:
                    with open(dest_path, "wb") as dst_file:
                        dst_file.write(src_file.read())
    else:
        update_status(
            api_status_path,
            "phase2",
            False,
            "No SPEAKER folder found in the output directory.",
        )
        return

    # Create folder for each speaker, and move the audio to the corresponding folder
    for segment in segments:
        segment["file_path"] = segment["file_path"].replace("SPEAKER\\", "")
        speaker_name = segment["speaker"]
        speaker_folder = os.path.join(speaker_dataset_folder, speaker_name)
        if not os.path.exists(speaker_folder):
            os.makedirs(speaker_folder)

        # Move the audio file to the corresponding speaker folder
        source_path = os.path.join(speaker_dataset_folder, segment["file_path"])
        dest_path = os.path.join(speaker_folder, f"{segment['id']}.wav")
        if os.path.exists(source_path):
            os.rename(source_path, dest_path)

    # Merge all audio into one wav on each speaker folder
    speaker_list = set()
    for speaker in os.listdir(speaker_dataset_folder):
        speaker_folder = os.path.join(speaker_dataset_folder, speaker)
        speaker_list.add(speaker)
        if os.path.isdir(speaker_folder):
            # Create a list to hold the audio segments
            audio_segments = []
            for file in os.listdir(speaker_folder):
                if file.endswith(".wav"):
                    file_path = os.path.join(speaker_folder, file)
                    audio_segments.append(AudioSegment.from_wav(file_path))

            # Combine all audio segments into one
            if audio_segments:
                # combined_audio = sum(audio_segments)
                silence = AudioSegment.silent(duration=500)  # 500ms of silence
                combined_audio = audio_segments[0]

                for segment in audio_segments[1:]:
                    combined_audio += silence + segment
                combined_audio.export(
                    os.path.join(speaker_folder, f"{speaker}.wav"), format="wav"
                )

                # Remove individual audio files after merging
                for file in os.listdir(speaker_folder):
                    if file.endswith(".wav") and file != f"{speaker}.wav":
                        os.remove(os.path.join(speaker_folder, file))

    # * Now, you should have 1 wav file for each speaker in the SPEAKER_DATASET folder

    for speaker in speaker_list:
        # Step 1: Pass merged audio to SoVits for slicing
        """
        1. Will set min length to process is 4 seconds, only audio less then 4s will pass
        But the audio must be longer than 4s as already merged
        2. Need to update the JSON data after SoVits splitted the audio

        slice_audio(inp="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/Input",
                opt_root="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/1_slice_audio_out",
                threshold=-34,
                min_length=4000,
                min_interval=300,
                hop_size=10,
                max_sil_kept=500,
                _max=0.9,
                alpha=0.25,
                n_parts=4)
        """
        speaker_sp_output_folder = os.path.join(
            output_folder, "SPEAKER_DATASET", speaker
        )
        server_url = SOVITS_SERVER + "/training/slice_audio"
        process_folder = os.path.join(speaker_sp_output_folder, "SPEAKER_PROCESSED")
        recreate_folder(process_folder)
        try:
            returnData = {
                "inp": speaker_sp_output_folder,
                "opt_root": process_folder,
                "threshold": -34,
                "min_length": 4000,
                "min_interval": 300,
                "hop_size": 10,
                "max_sil_kept": 500,
                "_max": 0.9,
                "alpha": 0.25,
                "n_parts": 4,
            }
            response = requests.post(
                server_url,
                # Change 'data' to 'json' to send a JSON payload
                params=returnData,
            )
            response.raise_for_status()
        except Exception as e:
            update_status(api_status_path, "phase2", False, f"Error on slice: {str(e)}")
            return

        # Step 3.2: Do denoise on the processed audio
        """
            # Denoise
            denoise(denoise_inp_dir="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/1_slice_audio_out",
                    denoise_opt_dir="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/2_denoise_out")
        """
        update_status(
            api_status_path,
            "phase2",
            False,
            "Creating Dataset for SoVits... Denoise audio...",
        )
        server_url = SOVITS_SERVER + "/training/denoise"
        try:
            response = requests.post(
                server_url,
                params={
                    "denoise_inp_dir": process_folder,
                    "denoise_opt_dir": speaker_sp_output_folder
                    + "/SPEAKER_PROCESSED_DENOISE",
                },
            )
            response.raise_for_status()
        except Exception as e:
            update_status(
                api_status_path, "phase2", False, f"Error on denoise: {str(e)}"
            )
            return

        # Step 3.4: Do ASR on the processed audio (Grouped by speaker)
        """
        open_asr(asr_inp_dir="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/2_denoise_out",
            asr_opt_dir="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/3_asr_out",
            asr_model="Faster Whisper (多语种)",
            asr_model_size="large-v3",
            asr_lang="auto",
            asr_precision="int8")
        """
        update_status(
            api_status_path,
            "phase2",
            False,
            "Creating Dataset for SoVits... Preprocessing audio on ASR... " + speaker,
        )
        server_url = SOVITS_SERVER + "/training/asr"
        try:
            response = requests.post(
                server_url,
                params={
                    "asr_inp_dir": f"{speaker_sp_output_folder}\\SPEAKER_PROCESSED_DENOISE",
                    "asr_opt_dir": f"{speaker_sp_output_folder}\\SPEAKER_PROCESSED_DENOISE_ASR",
                    "asr_model": "Faster Whisper (多语种)",
                    "asr_model_size": "large-v3",
                    "asr_lang": "auto",
                    "asr_precision": "int8",
                },
            )
            response.raise_for_status()
        except Exception as e:
            update_status(api_status_path, "phase2", False, f"Error on ASR: {str(e)}")
            return

        # Step 3.5: Preprocess the ASR output to .list for training
        """
        preprocess_one_step(inp_text="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/3_asr_out/2_denoise_out.list",
                    inp_wav_dir="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/2_denoise_out",
                    exp_name="20250409_Test",
                    gpu_numbers1a="0-0",
                    gpu_numbers1Ba="0-0",
                    gpu_numbers1c="0-0",
                    bert_pretrained_dir="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                    ssl_pretrained_dir="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base",
                    pretrained_s2G_path="L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                    opt_dir="L:/MSCCS Project/GPT-SoVITS/Experimental/Temp/20250409_Test")
        """
        update_status(
            api_status_path,
            "phase2",
            False,
            "Creating Dataset for SoVits... Preprocessing audio to list..." + speaker,
        )
        server_url = SOVITS_SERVER + "/training/preprocess"
        try:
            response = requests.post(
                server_url,
                params={
                    # ! The list name is same as the output folder name
                    "inp_text": f"{speaker_sp_output_folder}\\SPEAKER_PROCESSED_DENOISE_ASR\\SPEAKER_PROCESSED_DENOISE.list",
                    "inp_wav_dir": f"{speaker_sp_output_folder}\\SPEAKER_PROCESSED_DENOISE",
                    "exp_name": "SPEAKER",
                    "gpu_numbers1a": "0-0",
                    "gpu_numbers1Ba": "0-0",
                    "gpu_numbers1c": "0-0",
                    "bert_pretrained_dir": "chinese-roberta-wwm-ext-large",
                    "ssl_pretrained_dir": "chinese-hubert-base",
                    "pretrained_s2G_path": "gsv-v2final-pretrained/s2G2333k.pth",
                    "opt_dir": f"{speaker_sp_output_folder}\\SPEAKER_PROCESSED_DENOISE_ASR_PREPROCESS",
                },
            )
            response.raise_for_status()

            update_status(
                api_status_path,
                "phase2",
                False,
                f"Completed preprocessing for speaker: {speaker}",
            )
        except Exception as e:
            update_status(
                api_status_path,
                "phase2",
                False,
                f"Error on preprocess: {str(e)}",
            )
            return

    # Step 4: Completed
    update_status(
        api_status_path,
        "phase2",
        True,
        f"Completed preprocessing for all speaker",
    )
