import json
import threading
from transformers import M2M100ForConditionalGeneration
from app.models.small100.tokenization_small100 import SMALL100Tokenizer
from app.models.bart_translation_zh_yue.translation_pipeline import TranslationPipeline

# tokenizer.tgt_lang = "zh"  # Change this for different target language


# encoded_zh = tokenizer("What do you want to eat tonight ?", return_tensors="pt")
# generated_tokens = model.generate(**encoded_zh)
# zh_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(zh_text)

# pipe = TranslationPipeline(device=0)
# print(pipe(zh_text))

MAP_LANGUAGES = {
    "Cantonese": "zh",
    "Mandarin": "zh",
    "English": "en",
    "Korean": "ko",
    "Japanese": "ja",
}

model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")


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


def do_translation(text, src_lang, tgt_lang):
    """
    Translate text from source language to target language.
    """
    # Set the tokenizer's source and target languages
    tokenizer.src_lang = MAP_LANGUAGES[src_lang]
    tokenizer.tgt_lang = MAP_LANGUAGES[tgt_lang]

    # Encode the input text
    encoded_text = tokenizer(text, return_tensors="pt")

    # Generate translation
    generated_tokens = model.generate(**encoded_text)

    # Decode the generated tokens to get the translated text
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    if tgt_lang == "Cantonese":
        pipe = TranslationPipeline(device=0)
        result = pipe(translated_text[0])

        return result[0]["translation_text"] if result else ""

    return translated_text[0] if translated_text else ""


def process_translation(
    src_lang, tgt_lang, json_data, api_status_path, output_json_path
):
    # Start a thread to process the video asynchronously
    thread = threading.Thread(
        target=_process_translation_thread,
        args=(src_lang, tgt_lang, json_data, api_status_path, output_json_path),
    )
    thread.daemon = True
    thread.start()


def save_result(processed_data, output_json_path):
    # Save the processed data to a JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)


def _process_translation_thread(
    src_lang, tgt_lang, json_data, api_status_path, output_json_path
):
    update_status(
        api_status_path,
        "phase2",
        False,
        f"Loading Models for translation ...",
    )

    if json_data:
        # If json_data is provided, extract the text to be translated
        for data in json_data:
            if data.get("text"):
                data["translated_text"] = do_translation(
                    data["text"], src_lang, tgt_lang
                )
                update_status(
                    api_status_path,
                    "phase2",
                    False,
                    f"Translation in progress ... -> {data}",
                )

        save_result(json_data, output_json_path)

        update_status(
            api_status_path,
            "phase2",
            True,
            f"Translation and saving is completed",
        )

        return json_data

    save_result(json_data, output_json_path)

    update_status(
        api_status_path,
        "phase2",
        True,
        f"Translation and saving is completed",
    )

    return {}
