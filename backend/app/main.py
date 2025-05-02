import os
import json
import shutil
import logging
import traceback
from fastapi import Body, FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from decouple import config
from typing import Optional
import aiofiles
from fastapi.staticfiles import StaticFiles

from app.translation_processing import process_translation
from app.training_processing import process_training
from app.generation_processing import process_generation, process_respeed
from app.final_video_generate_processing import (
    process_face_detection,
    process_final_generation,
)
from app.utils.updateApiStatus import updateApiStatus
import requests

from app.utils.tools import recreate_folder

from .audio_processing import (
    process_audio_combination,
    process_delete_audio,
    process_save_segments,
    process_video,
)

SOVITS_SERVER = config("SOVITS_SERVER")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Processing API",
    description="API for processing videos with speaker diarization and transcription",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize work directory paths
WORK_DIR = os.path.join(os.getcwd(), "public")
INPUT_DIR = os.path.join(WORK_DIR, "input")
PHASE1_DIR = os.path.join(WORK_DIR, "phase1")
PHASE2_DIR = os.path.join(WORK_DIR, "phase2")
PHASE3_DIR = os.path.join(WORK_DIR, "phase3")
PHASE4_DIR = os.path.join(WORK_DIR, "phase4")
PHASE5_DIR = os.path.join(WORK_DIR, "phase5")
API_STATUS_PATH = os.path.join(WORK_DIR, "api_status.json")

# Mount the public directory for static files
app.mount("/public", StaticFiles(directory=WORK_DIR), name="public")


@app.get("/")
async def root():
    return {"message": "Video Processing API is running"}


@app.post("/upload-video")
async def upload_video(
    file: Optional[UploadFile] = File(None),
    input_language: str = Form(...),
    output_language: str = Form(...),
):
    try:
        # Check if file was actually provided
        if file is None:
            logger.error("No file was provided in request")
            raise HTTPException(status_code=400, detail="No file was provided")

        logger.info(
            f"Received file: {file.filename}, content_type: {file.content_type}"
        )

        # Save the intput language and output language to json on input directory
        dataToSave = {
            "input_language": input_language,
            "output_language": output_language,
        }
        language_setting_path = os.path.join(INPUT_DIR, "language.json")
        with open(language_setting_path, "w") as f:
            json.dump(dataToSave, f)

        # Validate file is an MP4
        if not file.filename.lower().endswith(".mp4"):
            logger.error(f"Invalid file format: {file.filename}")
            raise HTTPException(status_code=400, detail="Only MP4 files are accepted")

        # Save the video file
        input_video_path = os.path.join(INPUT_DIR, "original_video.mp4")

        # If a video already exists, remove it
        if os.path.exists(input_video_path):
            os.remove(input_video_path)

        # Clean the phase1 directory
        if os.path.exists(PHASE1_DIR):
            shutil.rmtree(PHASE1_DIR)
            os.makedirs(PHASE1_DIR)

        # Reset API status
        updateApiStatus(
            "phase1",
            {"is_complete": False, "message": "Processing video...", "data": {}},
        )

        # Save the uploaded file
        async with aiofiles.open(input_video_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # Start processing in the background
        process_video(input_video_path, PHASE1_DIR, API_STATUS_PATH)

        return {"message": "Video uploaded successfully. Processing started."}

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing upload: {str(e)}"
        )


@app.get("/status/{phase}")
async def get_status(phase: str):
    if not os.path.exists(API_STATUS_PATH):
        return JSONResponse(status_code=404, content={"error": "Status file not found"})

    with open(API_STATUS_PATH, "r") as status_file:
        status = json.load(status_file)

    if phase not in status:
        return JSONResponse(
            status_code=400, content={"error": f"Invalid phase: {phase}"}
        )

    return status[phase]


@app.get("/phase2/result")
async def get_transcriptions():
    return get_transcriptions_json()


def get_transcriptions_json():
    transcriptions_path = os.path.join(PHASE1_DIR, "transcriptions.json")

    if not os.path.exists(transcriptions_path):
        return {}

    with open(transcriptions_path, "r", encoding="utf-8") as transcriptions_file:
        transcriptions = json.load(transcriptions_file)

    return transcriptions


def get_translated_json():
    transcriptions_path = os.path.join(PHASE3_DIR, "translated_data.json")

    if not os.path.exists(transcriptions_path):
        return {}

    with open(transcriptions_path, "r", encoding="utf-8") as transcriptions_file:
        transcriptions = json.load(transcriptions_file)

    return transcriptions


@app.post("/phase2/combine-audio")
async def combine_audio(data: dict = Body(...)):
    try:
        if data is None:
            logger.error("No combine data provided")
            raise HTTPException(status_code=400, detail="No combine data provided")

        # Get the list of segments to combine from the request body
        combine_list = data.get("combine", [])

        # Reset API status
        updateApiStatus(
            "combine",
            {"is_complete": False, "message": "Processing Audio...", "data": {}},
        )

        # Check if the combine list is empty
        if not combine_list:
            logger.error("Combine list is empty")
            raise HTTPException(status_code=400, detail="Combine list is empty")

        process_audio_combination(combine_list, PHASE1_DIR, API_STATUS_PATH)

        return {"message": "Audio combined processing started."}

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error combining audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error combining audio: {str(e)}")


@app.post("/phase2/delete-audio")
async def delete_audio(data: dict = Body(...)):
    try:
        if data is None:
            logger.error("No delete data provided")
            raise HTTPException(status_code=400, detail="No delete data provided")

        # Get the list of segments to delete from the request body
        delete_list = data.get("delete", [])

        # Check if the delete list is empty
        if not delete_list:
            logger.error("Delete list is empty")
            raise HTTPException(status_code=400, detail="Delete list is empty")

        process_delete_audio(delete_list, PHASE1_DIR, API_STATUS_PATH)

        return {"message": "Audio deletion processing started."}

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error deleting audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting audio: {str(e)}")


@app.post("/phase2/save")
async def save_phase2(data: dict = Body(...)):
    try:
        if data is None:
            logger.error("No save data provided")
            raise HTTPException(status_code=400, detail="No save data provided")

        # Get the list of segments to save from the request body
        save_list = data.get("data", {})

        # Check if the save list is empty
        if not save_list:
            logger.error("Save list is empty")
            raise HTTPException(status_code=400, detail="Save list is empty")

        # Process saving the segments (this function should be implemented in your audio_processing module)
        process_save_segments(save_list, PHASE2_DIR)

        # Process translation
        get_language_setting_path = INPUT_DIR + "/language.json"
        with open(get_language_setting_path, "r") as f:
            language_setting = json.load(f)
        input_language = language_setting["input_language"]
        output_language = language_setting["output_language"]

        get_pharse2_result_path = PHASE2_DIR + "/final_input.json"
        with open(get_pharse2_result_path, "r") as f:
            json_data = json.load(f)

        # Reset API status
        updateApiStatus(
            "phase2",
            {"is_complete": False, "message": "Processing Translation ...", "data": {}},
        )

        output_json_path = os.path.join(PHASE2_DIR, "final_output.json")
        process_translation(
            input_language,
            output_language,
            json_data,
            API_STATUS_PATH,
            output_json_path,
        )

        return {"message": "Translation Processing started."}

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error saving audio segments: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error saving audio segments: {str(e)}"
        )


@app.get("/phase2/final-result")
async def get_final_result():
    try:
        final_result_path = os.path.join(PHASE2_DIR, "final_output.json")

        if not os.path.exists(final_result_path):
            return JSONResponse(
                status_code=404, content={"error": "Final result not found"}
            )

        with open(final_result_path, "r", encoding="utf-8") as final_result_file:
            final_result = json.load(final_result_file)

        return final_result

    except Exception as e:
        logger.error(f"Error retrieving final result: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving final result: {str(e)}"
        )


@app.post("/phase3/training")
async def training(data: dict = Body(...)):
    # Frontend will return the translation data in the request body, save it as 'train_data.json'

    # Reset API status
    updateApiStatus(
        "phase3",
        {"is_complete": False, "message": "Processing Training...", "data": {}},
    )

    try:
        if data is None:
            logger.error("No translated data provided")
            raise HTTPException(status_code=400, detail="No translated data provided")

        # Get the training data from the request body
        training_data = data.get("data", {})

        # Check if the training data is empty
        if not training_data:
            logger.error("Translated data is empty")
            raise HTTPException(status_code=400, detail="Translated data is empty")

        # Save the training data to a JSON file
        training_data_path = os.path.join(PHASE3_DIR, "translated_data.json")
        with open(training_data_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=4)

        # Start to train the model with preprocessed SoVits Dataset
        process_training(
            PHASE1_DIR,
            PHASE3_DIR,
            API_STATUS_PATH,
        )

        return {"message": "Training data saved successfully."}

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error saving training data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error saving training data: {str(e)}"
        )


@app.get("/phase4/model-list")
async def get_model_list():
    try:
        response = requests.get(SOVITS_SERVER + "/get_model_list")
        response.raise_for_status()

        # Get model list from the json on phase3 directory
        model_name_path = os.path.join(PHASE3_DIR, "model_name.json")
        if not os.path.exists(model_name_path):
            return JSONResponse(
                status_code=404, content={"error": "Model list not found"}
            )

        with open(model_name_path, "r") as model_name_file:
            model_name = json.load(model_name_file)
            return {
                "name": model_name,
                "model_list": response.json(),
            }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"error": f"Error retrieving model list: {str(e)}"}
        )


@app.post("/phase4/generate")
async def generate(
    # target_speaker: str = None,
    selectedModels: dict = None,
    # ref_audio_path: str = None,  # This is the `file_path` : "SPEAKER\\SPEAKER_01_14.wav"
    # target_text_path: str = None,
    # target_language: str = None,
    # output_path: str = None,
    ref_freeze: bool = False,
):
    """
    selected_models = {
    "SPEAKER_00":
        { "sovits": "SPEAKER_00_sovits_17454777141_e8_s64_l32.pth", "gpt": "SPEAKER_00_gpt_17454783252-e15.ckpt" },
    "SPEAKER_01":
        { "sovits": "SPEAKER_01_sovits_17454771691_e8_s80_l32.pth", "gpt": "SPEAKER_01_gpt_17454782742-e15.ckpt" }
    }
    """

    main_data_list = []

    selected_models = selectedModels["selectedModels"]
    # print("selected_models: ", selected_models)

    trans_json = get_translated_json()
    is_last_one = False
    # Auto loop the json to generate all the audios
    for idx, data in enumerate(trans_json):

        # Check if this is the last one
        is_last_one = idx == len(trans_json) - 1

        real_ref_audio_path = ""
        real_ref_audio_text = ""

        # * Ref Audio need to be 3 to 10 seconds long
        # * If the ref audio is too long, trim to 9 seconds and save it as temp_xxx.wav and do asr on it for ref_text
        # * If the ref audio is too short, get `target_speaker` and random select a ref audio from the same speaker that is more than 3 seconds long

        get_random_ref_audio = False
        ref_need_to_trim = False

        # Search the ref audio in json from phase1 directory
        ref_audio_path = data["file_path"]
        target_speaker = data["speaker"]

        if data["duration"] >= 3:
            real_ref_audio_path = PHASE1_DIR + "\\" + ref_audio_path
            real_ref_audio_text = data["text"]
            if data["duration"] >= 10:
                ref_need_to_trim = True
        else:
            get_random_ref_audio = True

        # if real_ref_audio_path is "" and real_ref_audio_text is "", mean not found the ref
        if real_ref_audio_path == "" and real_ref_audio_text == "":
            get_random_ref_audio = True

        if get_random_ref_audio is True:
            for i in trans_json:
                if i["speaker"] == target_speaker and (i["duration"] >= 3):
                    real_ref_audio_path = PHASE1_DIR + "\\" + i["file_path"]
                    real_ref_audio_text = i["text"]

                    if i["duration"] >= 10:
                        ref_need_to_trim = True
                    break

        # Read the language json from input directory
        get_language_setting_path = INPUT_DIR + "/language.json"
        with open(get_language_setting_path, "r") as f:
            language_setting = json.load(f)

        # ref language "中文", "英文", "日文"
        MAPPING_INPUT_LANGUAGES = {
            "Cantonese": "粤语",
            "Mandarin": "中文",
            "English": "英文",
            "Japanese": "日文",
            "Korean": "韩文",
        }

        # Output language ["中文", "英文", "日文", "粤语", "中英混合", "日英混合", "多语种混合"]
        MAPPING_OUTPUT_LANGUAGES = {
            "Cantonese": "粤英混合",
            "Mandarin": "中文",
            "English": "英文",
            "Japanese": "日文",
            "Korean": "韩文",
        }

        log_data = {
            "tran_id": data["id"],
            "target_speaker": target_speaker,
            "gpt_model_path": selected_models[target_speaker]["gpt"],
            "sovits_model_path": selected_models[target_speaker]["sovits"],
            "ref_audio_path": real_ref_audio_path,
            "ref_text_path": real_ref_audio_text,
            "ref_language": MAPPING_INPUT_LANGUAGES[language_setting["input_language"]],
            "target_text_path": data["translated_text"],
            "target_language": MAPPING_OUTPUT_LANGUAGES[
                language_setting["output_language"]
            ],
            "output_path": PHASE4_DIR + "\\SPEAKER\\",
            "output_file_name": data["file_path"].replace("SPEAKER\\", ""),
            "ref_free": ref_freeze,
            "api_status_path": API_STATUS_PATH,
            "phase4_dir": PHASE4_DIR,
            "ref_need_to_trim": ref_need_to_trim,
            "is_last_one": is_last_one,
        }
        main_data_list.append(log_data)

    # Clean the folder first
    recreate_folder(PHASE4_DIR)

    process_generation(main_data_list, PHASE3_DIR, PHASE4_DIR, False)

    return {"message": "Generation processing started."}


@app.get("/phase4/result")
async def get_generation_result():
    try:
        final_json_path = os.path.join(PHASE4_DIR, "final_result.json")
        if not os.path.exists(final_json_path):
            return []

        with open(final_json_path, "r", encoding="utf-8") as final_result_file:
            final_result = json.load(final_result_file)
            return final_result

    except Exception as e:
        logger.error(f"Error retrieving final result: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving final result: {str(e)}"
        )


"""
{
    "id": 14,
    "speed": 1.7792818198909568,
    "translated_text": "選舉總統唐納德·特朗普，歡迎再次見面新聞。"
    "selectedModels" : {}
}
"""


@app.post("/phase4/re-generate")
async def re_generate(
    data: dict = None,
):
    # Check if all data included
    if data is None:
        logger.error("Missing data for re-generate")
        raise HTTPException(status_code=500, detail="Missing data for re-generate")

    id = data["id"]
    speed = data["speed"]
    translated_text = data["translated_text"]
    selected_models = data["selectedModels"]

    # Check if the id is valid
    # Get the final result json
    final_json_path = os.path.join(PHASE4_DIR, "final_result.json")
    if not os.path.exists(final_json_path):
        logger.error("Final result json not found")
        raise HTTPException(status_code=404, detail="Final result json not found")

    with open(final_json_path, "r", encoding="utf-8") as final_result_file:
        final_result = json.load(final_result_file)

        for q in final_result:
            if q["id"] == id:

                # Check if text update or update update

                if translated_text != q["translated_text"]:

                    real_ref_audio_path = ""
                    real_ref_audio_text = ""

                    # * Ref Audio need to be 3 to 10 seconds long
                    # * If the ref audio is too long, trim to 9 seconds and save it as temp_xxx.wav and do asr on it for ref_text
                    # * If the ref audio is too short, get `target_speaker` and random select a ref audio from the same speaker that is more than 3 seconds long

                    get_random_ref_audio = False
                    ref_need_to_trim = False

                    # Search the ref audio in json from phase1 directory
                    ref_audio_path = q["file_path"]
                    target_speaker = q["speaker"]

                    if q["duration"] >= 3:
                        real_ref_audio_path = PHASE1_DIR + "\\" + ref_audio_path
                        real_ref_audio_text = q["text"]
                        if q["duration"] >= 10:
                            ref_need_to_trim = True
                    else:
                        get_random_ref_audio = True

                    # if real_ref_audio_path is "" and real_ref_audio_text is "", mean not found the ref
                    if real_ref_audio_path == "" and real_ref_audio_text == "":
                        get_random_ref_audio = True

                    if get_random_ref_audio is True:
                        for i in final_result:
                            if i["speaker"] == target_speaker and (i["duration"] >= 3):
                                real_ref_audio_path = PHASE1_DIR + "\\" + i["file_path"]
                                real_ref_audio_text = i["text"]

                                if i["duration"] >= 10:
                                    ref_need_to_trim = True
                                break

                    # Read the language json from input directory
                    get_language_setting_path = INPUT_DIR + "/language.json"
                    with open(get_language_setting_path, "r") as f:
                        language_setting = json.load(f)

                    # ref language "中文", "英文", "日文"
                    MAPPING_INPUT_LANGUAGES = {
                        "Cantonese": "粤语",
                        "Mandarin": "中文",
                        "English": "英文",
                        "Japanese": "日文",
                        "Korean": "韩文",
                    }

                    # Output language ["中文", "英文", "日文", "粤语", "中英混合", "日英混合", "多语种混合"]
                    MAPPING_OUTPUT_LANGUAGES = {
                        "Cantonese": "粤英混合",
                        "Mandarin": "中文",
                        "English": "英文",
                        "Japanese": "日文",
                        "Korean": "韩文",
                    }

                    log_data = {
                        "tran_id": q["id"],
                        "target_speaker": target_speaker,
                        "gpt_model_path": selected_models[target_speaker]["gpt"],
                        "sovits_model_path": selected_models[target_speaker]["sovits"],
                        "ref_audio_path": real_ref_audio_path,
                        "ref_text_path": real_ref_audio_text,
                        "ref_language": MAPPING_INPUT_LANGUAGES[
                            language_setting["input_language"]
                        ],
                        "target_text_path": translated_text,
                        "target_language": MAPPING_OUTPUT_LANGUAGES[
                            language_setting["output_language"]
                        ],
                        "output_path": PHASE4_DIR + "\\SPEAKER\\",
                        "output_file_name": q["file_path"].replace("SPEAKER\\", ""),
                        "ref_free": False,
                        "api_status_path": API_STATUS_PATH,
                        "phase4_dir": PHASE4_DIR,
                        "ref_need_to_trim": ref_need_to_trim,
                        "is_last_one": True,
                    }

                    print(">>> Before generate", log_data)

                    # Update the speed and id
                    process_generation([log_data], PHASE3_DIR, PHASE4_DIR, True)
                else:
                    process_respeed(id, speed, PHASE4_DIR, API_STATUS_PATH)
                break

    return {"message": "Re-generation processing started."}


@app.get("/phase5/start_face_detection")
async def start_detection_result():
    try:
        # Check if 'images' folder exists in phase5 directory
        phase5_images_dir = os.path.join(PHASE5_DIR, "images")
        if os.path.exists(phase5_images_dir):
            return JSONResponse(
                status_code=200, content={"message": "Face detection already started."}
            )

        process_face_detection(
            API_STATUS_PATH,
            INPUT_DIR,
            PHASE1_DIR,
            PHASE5_DIR,
        )

        return {"message": "Face detection processing started."}

    except Exception as e:
        logger.error(f"Error retrieving final result: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving final result: {str(e)}"
        )


@app.get("/phase5/get_face_detection")
async def get_detection_result():
    try:
        # Open the phase5 directory and the folder of 'images'
        phase5_dir = os.path.join(PHASE5_DIR, "images")
        if not os.path.exists(phase5_dir):
            return JSONResponse(
                status_code=404, content={"error": "Face detection result not found"}
            )

        # Get the list of all files in the directory
        file_list = os.listdir(phase5_dir)

        # loop the files and split by '_' and group by the first part, return a json with the file name and the group name
        grouped_files = {}
        for file in file_list:
            group_name = file.split("_")[0]
            if group_name not in grouped_files:
                grouped_files[group_name] = []
            grouped_files[group_name].append(file)

        # Get the speaker list from the phase1 directory
        speaker_list_path = os.path.join(PHASE1_DIR, "speaker_list.json")
        if not os.path.exists(speaker_list_path):
            return JSONResponse(
                status_code=404, content={"error": "Speaker list not found"}
            )

        with open(speaker_list_path, "r") as f:
            speaker_names = json.load(f)
            # Convert speaker names list to list of dictionaries
            speaker_list = [{"name": name} for name in speaker_names]

        # random get an audio that is more than 3 seconds long for each speaker from the phase 4 /final result json
        final_result_path = os.path.join(PHASE4_DIR, "final_result.json")
        if not os.path.exists(final_result_path):
            return JSONResponse(
                status_code=404, content={"error": "Final result json not found"}
            )

        with open(final_result_path, "r", encoding="utf-8") as final_result_file:
            final_result = json.load(final_result_file)
            speaker_audio = {}

            # Get the audio path for each speaker
            for speaker_info in speaker_list:
                speaker_name = speaker_info["name"]
                found_audio = False
                for q in final_result:
                    if q["speaker"] == speaker_name and (q["duration"] >= 3):
                        speaker_audio[speaker_name] = q["file_path"]
                        found_audio = True
                        break
                # If no suitable audio >= 3s found, assign the first one found for that speaker
                if not found_audio:
                    for q in final_result:
                        if q["speaker"] == speaker_name:
                            speaker_audio[speaker_name] = q["file_path"]
                            break

        return {
            "image_group": grouped_files,
            "speakers": speaker_list,
            "speaker_audio": speaker_audio,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error retrieving final result: {str(e)}"
        )


@app.post("/phase5/generate_final_video")
async def mask_video(data: dict = Body(...)):
    try:
        if data is None:
            logger.error("No mask data provided")
            raise HTTPException(status_code=400, detail="No mask data provided")

        # Process saving the segments (this function should be implemented in your audio_processing module)
        process_final_generation(
            API_STATUS_PATH, INPUT_DIR, PHASE1_DIR, PHASE4_DIR, PHASE5_DIR, data["data"]
        )

        return {"message": "Final Video generating process started."}

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error saving audio segments: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error saving audio segments: {str(e)}"
        )
