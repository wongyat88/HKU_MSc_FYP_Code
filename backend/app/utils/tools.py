import os
import shutil


def recreate_folder(folder_path):
    """Recreate the folder: If it exists, remove and recreate it."""
    # Check if the folder exists
    if os.path.exists(folder_path):
        # If exists, remove the folder and its contents
        shutil.rmtree(folder_path)

    # Recreate the folder
    os.makedirs(folder_path)
