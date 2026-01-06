import argparse
import base64
import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from openai import OpenAI
from PIL import Image

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

# This is the column where we store the model's result (similar to the Gemini approach).
MODEL_COLUMN_LLAMA = "llama-3.2-90b-latest"

SYSTEM_PROMPT = (
    "You are an agent walking through an environment in Blender. "
    "You will receive a series of images, each taken after taking an action in the environment (either moving straight or turning 15 degrees left/right). "
    "You will also receive a question that you must answer correctly after seeing all images. "
    "You will see objects with a shape and a color. The possible shapes include cuboid, cone, sphere. "
    "The possible colors include red, green, blue, yellow, purple, brown, black, orange. "
    "Please answer the question based on the set of images."
    "Answer as concisely as possible, usually only a single word. If you're asked about a true/false question, "
    "answer with 'yes' or 'no' only. If it's a question where you're asked to compare the number of objects, respond only with whichever object there are more of, or equal, if there are the same number of objects. "
    "If you're asked to count objects, answer only with the number (as a number, not in English) of objects you see. "
    "If you're asked whether you saw something before, after, or at the same time as another object, "
    "answer only with 'before', 'after', or 'same time' only. If the first time you see an object is in an image before another object, it comes before (and the other comes after). If two objects appear in the same frame together for their first viewing, it's same time."
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

QUESTION_CSVS = [
    "comparison_questions_in_frame.csv",
    "comparison_questions_out_of_frame.csv",
    "left_right_questions.csv",
    "number_questions.csv",
    "order_preserving_questions.csv",
]


def resize_and_encode_image(image_path, max_file_size_mb=5, quality=90):
    """
    Shrink an image (if needed) and convert it to base64-encoded JPG.
    Returns the string or None if we encountered an error.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB mode if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize image to 960x640 for better consistency
            img = img.resize((960, 640), Image.LANCZOS)

            buffer = BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=quality)

            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

    except Exception as e:
        logging.warning(f"Failed to process image {image_path}: {e}")
        return None


def get_custom_id(run_dir: str, csv_name: str, row_index: int, model_name: str) -> str:
    """
    Sanitize the run_dir, CSV filename, and model name then append __row_{row_index}.
    This ensures uniqueness across multiple run directories and model usage.
    """
    safe_run_dir = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(run_dir))
    safe_run_dir = safe_run_dir[:50]

    safe_csv_name = re.sub(r"[^a-zA-Z0-9_-]", "_", csv_name)
    safe_csv_name = safe_csv_name[:50]

    safe_model_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    safe_model_name = safe_model_name[:50]

    result = f"{safe_run_dir}__{safe_csv_name}__{safe_model_name}__row_{row_index}"
    return result[:64]


def call_llama_api(
    model_name,
    base64_images,
    question,
    annotations_text_block,
    extra_logging=False,
    max_retries=3,
    initial_delay=1,
):
    """
    Calls the LLaMA model on OpenRouter with the base64-encoded images + question + system prompt.
    Returns the response text or an error message.
    Implements exponential backoff retry logic with reset on success.
    """
    delay = initial_delay
    last_exception = None

    # Create the client once (outside the loop)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    # Prepare the message content with system prompt followed by images, then the question
    def build_messages():
        # "System" role instruction:
        system_msg = {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }
        # "User" role includes annotations, images, and question
        user_content = []
        if annotations_text_block:
            user_content.append({"type": "text", "text": annotations_text_block})

        for b64_img in base64_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                }
            )

        if question:
            user_content.append({"type": "text", "text": question})

        user_msg = {"role": "user", "content": user_content}
        return [system_msg, user_msg]

    for attempt in range(max_retries):
        try:
            messages = build_messages()

            if extra_logging:
                logging.info(f"Request for model={model_name}, question={question}")

            completion = client.chat.completions.create(
                model="meta-llama/llama-3.2-90b-vision-instruct",
                messages=messages,
            )

            # The call succeeded; reset if multiple attempts made
            if attempt > 0:
                logging.info(f"Successfully recovered after {attempt + 1} attempts")

            return completion.choices[0].message.content.strip()

        except Exception as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                logging.warning(
                    f"Error on attempt {attempt + 1}/{max_retries} for model={model_name}: {exc}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2

    logging.error(
        f"Final error calling LLaMA via OpenRouter for model={model_name}: {last_exception}"
    )
    return f"ERROR: {last_exception}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Parallel questions to LLaMA (via OpenRouter) for images, logging debug info."
    )
    parser.add_argument(
        "--base_dir",
        default="src/trajectories",
        help="Base directory with subfolders. Each should have an images/ subdir.",
    )
    parser.add_argument(
        "--logging",
        default="false",
        help="Set to 'true' to enable extra logging of question and messages.",
    )

    args = parser.parse_args()
    extra_logging = args.logging.strip().lower() == "true"

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        logging.error(f"{base_dir} does not exist.")
        sys.exit(1)

    # Collect all run directories
    run_dirs = sorted(
        [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    )

    # We'll store an in-memory map of run_dir -> csv_name -> row data
    csv_data_tracker = {}

    def load_annotations(images_dir):
        annotations_file = os.path.join(images_dir, "annotations.csv")
        annotations_list = []
        if os.path.isfile(annotations_file):
            logging.info(f"Found annotations.csv in {images_dir}, loading...")
            with open(annotations_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ann_text = row.get("annotation", "").strip()
                    if ann_text:
                        annotations_list.append(ann_text)
        else:
            logging.info(f"No annotations.csv in {images_dir}, skipping.")
        return annotations_list

    all_requests = []

    for run_dir in run_dirs:
        images_dir = os.path.join(run_dir, "images")
        if not os.path.isdir(images_dir):
            logging.info(f"No images/ folder found in {run_dir}, skipping.")
            continue

        logging.info(f"\nGathering data from run directory: {run_dir}")

        annotations_list = load_annotations(images_dir)
        if annotations_list:
            annotations_text_block = "Annotations:\n" + "\n".join(annotations_list)
        else:
            annotations_text_block = "Annotations:\n(None)"

        image_files = [
            fn
            for fn in sorted(os.listdir(images_dir))
            if os.path.splitext(fn)[1].lower() in VALID_EXTENSIONS
        ]
        base64_images = []
        for fn in image_files:
            path = os.path.join(images_dir, fn)
            b64 = resize_and_encode_image(path, max_file_size_mb=5)
            if b64:
                base64_images.append(b64)
                if extra_logging:
                    logging.info(f"Loaded and encoded image: {fn}")

        csv_data_tracker[run_dir] = {}

        for csv_name in QUESTION_CSVS:
            csv_path = os.path.join(images_dir, csv_name)
            if not os.path.isfile(csv_path):
                continue

            with open(csv_path, "r", encoding="utf-8", newline="") as f_in:
                reader = csv.DictReader(f_in)
                if not reader.fieldnames:
                    continue
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            # Ensure column for our LLaMA model
            if MODEL_COLUMN_LLAMA not in fieldnames:
                fieldnames.append(MODEL_COLUMN_LLAMA)

            csv_data_tracker[run_dir][csv_name] = {
                "fieldnames": fieldnames,
                "rows": rows,
            }

            for i, row in enumerate(rows):
                question = row.get("question", "").strip()
                if not question:
                    continue

                # If cell is empty or starts with ERROR, queue for re-run
                existing_llama = row.get(MODEL_COLUMN_LLAMA, "")
                if not existing_llama or existing_llama.startswith("ERROR:"):
                    all_requests.append(
                        {
                            "run_dir": run_dir,
                            "csv_name": csv_name,
                            "row_index": i,
                            "model_name": MODEL_COLUMN_LLAMA,
                            "question": question,
                            "base64_images": base64_images,
                            "annotations_text_block": annotations_text_block,
                        }
                    )

    if not all_requests:
        logging.info("No new requests to process. Exiting.")
        return

    max_workers = 3  # changed to 3
    logging.info(
        f"Processing {len(all_requests)} requests in parallel, up to {max_workers} at a time..."
    )
    results_map = {}  # (run_dir, csv_name, row_index, model_name) -> answer

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_request = {}
        for req in all_requests:
            future = executor.submit(
                call_llama_api,
                req["model_name"],
                req["base64_images"],
                req["question"],
                req["annotations_text_block"],
                extra_logging,
            )
            future_to_request[future] = req

        for future in as_completed(future_to_request):
            req = future_to_request[future]
            try:
                answer = future.result()
            except Exception as exc:
                logging.error(f"Request failed: {exc}")
                answer = f"ERROR: {exc}"

            run_dir = req["run_dir"]
            csv_name = req["csv_name"]
            row_index = req["row_index"]
            model_name = req["model_name"]
            results_map[(run_dir, csv_name, row_index, model_name)] = answer

    # Update CSVs with results
    for run_dir, csv_dict in csv_data_tracker.items():
        for csv_name, data_obj in csv_dict.items():
            fieldnames = data_obj["fieldnames"]
            rows = data_obj["rows"]

            for i, row in enumerate(rows):
                key = (run_dir, csv_name, i, MODEL_COLUMN_LLAMA)
                if key in results_map:
                    row[MODEL_COLUMN_LLAMA] = results_map[key]

            csv_path = os.path.join(run_dir, "images", csv_name)
            with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logging.info(f"Updated {csv_path} with LLaMA results.")

    logging.info("All requests completed and CSVs updated. Done.")


if __name__ == "__main__":
    main()

