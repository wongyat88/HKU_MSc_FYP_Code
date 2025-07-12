## MSc Computer Science Final Year Project

## Project Number : msp24051

### Integrated Solution for Speaker-Specific Dubbing and Translation with Semantic Audio

### Project Installation Guide

1. This project requires to run on Windows 10 / 11 OS
2. Download Git for windows to clone extra repository
3. Download Conda to create an Python environment on version 3.10.9 for windows
4. Download Node.js for version 22 LTS on Windows
5. Prepare a GPU for running the project. NVIDA GPU is reqired and CUDA version 11.8 is required for the project to run.
6. Activate the conda environment and CD to the backend folder and run `pip install -r requirements.txt` to install the backend dependencies.
7. Then install the Torch and Torchvision with the command `pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118` to install the GPU version of PyTorch.
    - If you have a different CUDA version, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for the correct command.
8. CD to the frontend folder and run `npm install` to install the frontend dependencies.
9. CD to backend folder and run `python test_gpu.py` to check if the GPU is working properly.
10. Go to `https://github.com/RVC-Boss/GPT-SoVITS/releases/tag/20250228v3` to download the GPT SoVits 20250228 v3 Repo for the project. It is a zip file, extract it to the backend folder.
11. Unzip the `wav2lip.zip` file to the backend folder. Then go to `https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view` to download the models files and place it in the `wav2lip` root folder.
12. Go to the folder `soVits_replace` and copy all the files and paste it to the donwloaded `GPT-SoVits` folder, replace all the existing folder.
13. Apply an account on Hugging Face and login using the command `huggingface-cli login` in the terminal. You need to create a token with `FINEGRAINED` access and save it.
14. Go to `https://huggingface.co/pyannote/speaker-diarization` with the same account and request for the access to the model. IF NOT DO THAT, THE DIARIZATION WILL NOT WORK.
15. Go to `https://huggingface.co/FunAudioLLM/SenseVoiceSmall` with the same account and request for the access to the model. Also, download the model and place it in the `backend/app/models` folder.
16. Go the backend folder, open and update the `.env` file with the content, here are some example values:

```
# .env
SOVITS_SERVER = "http://127.0.0.1:9880"
FACE_DETECTION_PATH = "F:\School\FYP2\backend_frontend_ui\face_detection"
WAV_2_LIP_PATH = "F:\School\FYP\lipSync\Wav2Lip"
LLM_API_URL = ""
LLM_API_KEY = ""
```

16. Go to the backend folder and go to the `face_detection` folder, run `pip install -r requirements.txt` to install the dependencies for face detection.
17. Go to the backend folder and go to the `wav2lip` folder, run `pip install -r requirements.txt` to install the dependencies for wav2lip.
18. Install the rubberband-4.0.0-gpl-executable-windows from the given folder. Add the `rubberband-4.0.0-gpl-executable-windows` folder to the PATH environment variable.

19. Go to the backend folder and run `python server.py` to start the backend server.
20. Go to the frontend folder and run `npm run dev` to start the frontend server.
21. Go to the GPT SoVits folder and run `runtime\python.exe api_v2.py` to start the GPT SoVits server.
22. Open your browser and go to `http://localhost:3001/` to access the frontend.

### Hugging Face Details

-   Hugging Face login.
    `pip install transformers` # This package include hugging face
    `huggingface-cli login` # Login with your token. Create your token at `https://huggingface.co/settings/tokens` , create a `FINEGRAINED` Token, u need to tick `Read access to contents of all repos under your personal namespace`, `Read access to contents of all public gated repos you can access` & `Read access to all collections under your personal namespace`; then give it a name and save. Remember to save the token as u cannot copy or view it again after u close the dialog.
