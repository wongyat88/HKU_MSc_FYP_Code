import datetime
import os
import threading
import json
from decouple import config
import requests

SOVITS_SERVER = config("SOVITS_SERVER")


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


def process_training(input_path, phase3_dir, api_status_path):
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_training_thread,
        args=(input_path, phase3_dir, api_status_path),
    )
    thread.daemon = True
    thread.start()


def _process_training_thread(input_path, phase3_dir, api_status_path):
    # Read the speaker_list.json to get the list of speakers
    speaker_list_path = os.path.join(input_path, "speaker_list.json")
    if not os.path.exists(speaker_list_path):
        print(f"File {speaker_list_path} does not exist.")
        return

    with open(speaker_list_path, "r") as f:
        speaker_list = json.load(f)

    print(f"Speaker list: {speaker_list}")

    model_name = {}

    """
    # SOVITS Training Parameters
    s_batch_size = 11
    s_epoch = 8
    s_exp_name = "20250402_Test04"
    s_text_low_lr_rate = 0.4
    s_if_save_latest = True
    s_if_save_every_weights = True
    s_save_every_n_epoch = 4
    s_gpunumbers = "0"
    s_pretrained_s2G = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    s_pretrained_s2D = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
    s_if_grad_ckpt = False
    s_lora_rank = 32
    s2_dir = "L:/MSCCS Project/GPT-SoVITS/Experimental/Steps/4_Preprocessing/20250402_Test04"

    start_sovits_training(s_batch_size,
                        s_epoch,
                        s_exp_name,
                        s_text_low_lr_rate,
                        s_if_save_latest,
                        s_if_save_every_weights,
                        s_save_every_n_epoch,
                        s_gpunumbers,
                        s_pretrained_s2G,
                        s_pretrained_s2D,
                        s_if_grad_ckpt,
                        s_lora_rank,
                        s2_dir)
    """
    for speaker in speaker_list:
        model_name[speaker] = {"sovits": None, "gpt": None}

        # Get now Datetime and convert to int then to str
        now = datetime.datetime.now()
        now_str = str(int(now.timestamp()))

        model_name[speaker]["sovits"] = speaker + "_sovits_" + now_str + "1"
        print(f"Model name for {speaker}: {model_name[speaker]['sovits']}")

        # Loop all speaker to train the SoVits models
        server_url = SOVITS_SERVER + "/training/sovits"
        try:
            update_status(
                api_status_path,
                "phase3",
                False,
                f"Start Training SoVits Model for {speaker} ...",
            )

            folder_path = os.path.join(
                input_path,
                "SPEAKER_DATASET",
                speaker,
                "SPEAKER_PROCESSED_DENOISE_ASR_PREPROCESS",
            )

            returnData = {
                "batch_size": 1,  # 1/2/5/8/12/14
                "epoch": 4,
                "exp_name": model_name[speaker]["sovits"],
                "text_low_lr_rate": 0.4,
                "if_save_latest": True,
                "if_save_every_weights": True,
                "save_every_n_epoch": 2,
                "gpunumbers": "0",
                "pretrained_s2G": "gsv-v2final-pretrained/s2G2333k.pth",
                "pretrained_s2D": "gsv-v2final-pretrained/s2G2333k.pth",
                "if_grad_ckpt": False,
                "lora_rank": 32,
                # "opt_dir": os.path.join(folder_path, speaker),
                "opt_dir": folder_path,
            }

            # Send the request to the server
            response = requests.post(
                server_url,
                json=returnData,
            )
            response.raise_for_status()

            print("Response from SoVits Training: " + response.text)

            # # Check if the request was successful
            # if response.status_code != 200:
            #     update_status(
            #         api_status_path,
            #         "phase3",
            #         False,
            #         f"Training SoVits Model for {speaker} failed: {response.text}",
            #     )
        except Exception as e:
            print(f"Error: {e}")
            update_status(
                api_status_path,
                "phase3",
                False,
                f"Error: {e}",
            )
            return

    for speaker in speaker_list:
        # Get now Datetime and convert to int then to str
        now = datetime.datetime.now()
        now_str = str(int(now.timestamp()))

        model_name[speaker]["gpt"] = speaker + "_gpt_" + now_str + "2"

        folder_path = os.path.join(
            input_path,
            "SPEAKER_DATASET",
            speaker,
            "SPEAKER_PROCESSED_DENOISE_ASR_PREPROCESS",
        )

        # Loop all speaker to train the GPT models
        server_url = SOVITS_SERVER + "/training/gpt"
        """
        # GPT Training Parameters
        g_batch_size = 11
        g_epoch = 15
        g_exp_name = "20250402_Test04"
        g_if_dpo = False
        g_if_save_latest = True
        g_if_save_every_weights = True
        g_save_every_n_epoch = 5
        g_gpu_numbers = "0"
        g_pretrained_s1 = "L:/MSCCS Project/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        s1_dir = "L:/MSCCS Project/GPT-SoVITS/Experimental/Steps/4_Preprocessing/20250402_Test04"

        start_gpt_training(g_batch_size,
                        g_epoch,
                        g_exp_name,
                        g_if_dpo,
                        g_if_save_latest,
                        g_if_save_every_weights,
                        g_save_every_n_epoch,
                        g_gpu_numbers,
                        g_pretrained_s1,
                        s1_dir)
        """
        try:
            update_status(
                api_status_path,
                "phase3",
                False,
                f"Start Training GPT Model for {speaker} ...",
            )

            # Send the request to the server
            response = requests.post(
                server_url,
                json={
                    "batch_size": 4,
                    "epoch": 10,
                    "exp_name": model_name[speaker]["gpt"],
                    "if_dpo": False,
                    "if_save_latest": True,
                    "if_save_every_weights": True,
                    "save_every_n_epoch": 5,
                    "gpu_numbers": "0",
                    "pretrained_s1": "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                    # "opt_dir": os.path.join(folder_path, speaker),
                    "opt_dir": folder_path,
                },
            )
            response.raise_for_status()
            print("Response from GPT Training: " + response.text)

            # # Check if the request was successful
            # if response.status_code != 200:
            #     update_status(
            #         api_status_path,
            #         "phase3",
            #         False,
            #         f"Training GPT Model for {speaker} failed: {response.text}",
            #     )
        except Exception as e:
            print(f"Error: {e}")
            update_status(
                api_status_path,
                "phase3",
                False,
                f"Error: {e}",
            )
            return

    # Save the model_name in json file
    model_name_path = os.path.join(phase3_dir, "model_name.json")
    with open(model_name_path, "w") as f:
        json.dump(model_name, f, indent=4)

    # Update the API status to indicate that the training is complete
    update_status(
        api_status_path,
        "phase3",
        True,
        "Training SoVits and GPT models completed successfully.",
    )
