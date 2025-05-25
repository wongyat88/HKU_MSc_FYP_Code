import os
import shutil
from openai import OpenAI
from decouple import config
import re

API_URL = config("LLM_API_URL")
API_KEY = config("LLM_API_KEY")


def recreate_folder(folder_path):
    """Recreate the folder: If it exists, remove and recreate it."""
    # Check if the folder exists
    if os.path.exists(folder_path):
        # If exists, remove the folder and its contents
        shutil.rmtree(folder_path)

    # Recreate the folder
    os.makedirs(folder_path)


def call_llm_api(prompt, model=None):

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_URL,  # https://xxx.com/v1
    )

    response = client.chat.completions.create(
        model="deepseek-v3",  # deepseek-v3 / grok-3
        messages=[
            {
                "role": "system",
                "content": "You are a large language model, please follow the user's instructions carefully.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    response_data = response.choices[0].message.content

    print(response_data)

    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response_data, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return response_data.strip()
