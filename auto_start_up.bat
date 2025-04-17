@echo off


cd /d F:\School\FYP\backend_frontend_ui\venv\Scripts
call activate.bat
cd /d F:\School\FYP\backend_frontend_ui\backend
start cmd /k "code ."
start cmd /k "python server.py"


cd /d F:\School\FYP\backend_frontend_ui\backend\app\utils\GPTSoVITS_v3lora_20250228
start cmd /k "code ."
start cmd /k "runtime\python.exe api_v2.py"


cd /d F:\School\FYP\backend_frontend_ui\frontend\my-app
start cmd /k "code ."
start cmd /k "npm run dev"
