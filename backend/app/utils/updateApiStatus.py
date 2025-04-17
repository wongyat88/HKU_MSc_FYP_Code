import json
import os
import traceback



# Initialize work directory paths
WORK_DIR = os.path.join(os.getcwd(), "public")
INPUT_DIR = os.path.join(WORK_DIR, "input")
PHASE1_DIR = os.path.join(WORK_DIR, "phase1")
API_STATUS_PATH = os.path.join(WORK_DIR, "api_status.json")

DEFAULT_STATUS = {
    "is_complete": False,
    "message": "Not started",
    "data": {}
}


def updateApiStatus(key, content):
    """
    Update the API status file with the given key and content.
    
    Args:
        key (str): The key to update in the status file.
        content (dict): The content to update in the status file.
    """
    try:
        # Read the existing status file
        with open(API_STATUS_PATH, "r") as status_file:
            status = json.load(status_file)
        
        # Update the status with the new content
        status[key] = content
        
        # Write the updated status back to the file
        with open(API_STATUS_PATH, "w") as status_file:
            json.dump(status, status_file, indent=4)
    
    except Exception as e:
        traceback.print_exc()
        print(f"Error updating API status: {str(e)}")