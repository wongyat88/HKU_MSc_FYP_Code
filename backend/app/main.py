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
from app.utils.updateApiStatus import updateApiStatus
import requests

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
    transcriptions_path = os.path.join(PHASE1_DIR, "transcriptions.json")

    if not os.path.exists(transcriptions_path):
        return {}

    with open(transcriptions_path, "r") as transcriptions_file:
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


# @app.post("/phase3/generate")
# async def training(data: dict = Body(...)):
@app.get("/phase3/generate")
async def training():
    server_url = SOVITS_SERVER + "/generate_audio"
    try:
        response = requests.post(server_url)
        response.raise_for_status()

        print("Response from SoVits Training: " + response.text)

        return {"message": "Generate successfully."}

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error saving training data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error saving training data: {str(e)}"
        )
