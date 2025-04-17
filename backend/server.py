import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    # Check if app folders exist
    work_dir = os.path.join(os.getcwd(), "public")
    input_dir = os.path.join(work_dir, "input")
    phase1_dir = os.path.join(work_dir, "phase1")
    
    # Create directories if they don't exist
    for directory in [work_dir, input_dir, phase1_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    # Run the FastAPI server
    uvicorn.run(
        "app.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )