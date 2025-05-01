1. Create A venv first, `pip install -r requirements.txt`
2. Hugging Face login.
   `pip install transformers` # This package include hugging face
   `huggingface-cli login` # Login with your token. Create your token at `https://huggingface.co/settings/tokens` , create a `FINEGRAINED` Token, u need to tick `Read access to contents of all repos under your personal namespace`, `Read access to contents of all public gated repos you can access` & `Read access to all collections under your personal namespace`; then give it a name and save. Remember to save the token as u cannot copy or view it again after u close the dialog.

3. Download the models, details in models folder
4. Install the rubberband to your system path
5. Install GPT SoVits v3 from the release page, and copy and paste the `soVits` folder file to your GPT SoVits installed root folder
6. Follow the instruction to install venv in folder `face_detection` and `wav2lip`

### Run GPT SoVits server `runtime\python.exe api_v2.py`

### Run Backend `py server.py`

### Run Frontend `npm run dev`

Hosted on `http://localhost:3001/`

### Check list

-   Video remove human voice but remain background sounding
-   We need 3 different backend => 1. Backend for UI, 2. GPT SoVits, 3. wav2Lip
-   We need GPU support on the backend for SoVits and wav2Lip, but UI need to? (pyannote + funASR, but FunASR seem not support GPU???)
-   Speaker may say: arrr, emmm, ermmm. Need to think how to support.
-   GPT SoVits have training process api?
-   wav2Lip: PyTorch version lower, not suppport Higher CUDA version
-   wav2Lip: two or more target? (Yolo8 -> Cut Video -> modify video -> Combine back to the original one)

## What have done

1. Input a video extract to audio
2. Use pyannote => get the speaker diarization
3. FunASR => get the text
4. Combine the data into a new JSON file.

## Coming need to do:

0. Frontend: Choice Speaker (??), choice language <Harcode (Cantonese)>
1. GPT SoVits training API ??
2. English to Cantonese (Small 100 + ??) <implement Backend>
3. GPT SoVits inference API (Checked is exist) <wait for step 1>
