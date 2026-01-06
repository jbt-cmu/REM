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

import google.generativeai as genai
from PIL import Image

# We will write two new model columns, one for each Gemini model:
MODEL_COLUMN_PRO = "gemini-2.0-flash"

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

# The name(s) of CSV files that might contain questions:
QUESTION_CSVS = [
    "comparison_questions_in_frame.csv",
    "comparison_questions_out_of_frame.csv",
    "left_right_questions.csv",
    "number_questions.csv",
    "order_preserving_questions.csv",
]


def resize_and_encode_image(image_path, max_file_size_mb=5, quality=85):
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
            file_size = buffer.tell()
            max_bytes = max_file_size_mb * 1024 * 1024

            while file_size > max_bytes and quality > 10:
                quality = int(quality * 0.9)
                buffer.seek(0)
                buffer.truncate(0)
                img.save(buffer, format="JPEG", optimize=True, quality=quality)
                file_size = buffer.tell()

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


def call_gemini_api(
    model_name,
    base64_images,
    question,
    annotations_text_block,
    extra_logging=False,
    max_retries=3,
    initial_delay=1,
):
    """
    Calls the specified Gemini model with the base64-encoded images and question + system prompt.
    Returns the response text or an error message.
    Implements exponential backoff retry logic with reset on success.
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            # Instead of calling genai.models.get_model, create a GenerativeModel directly:
            model = genai.GenerativeModel(model_name=model_name)

            content_parts = [
                SYSTEM_PROMPT,  # system-like instructions
                annotations_text_block,  # (Annotations block)
            ]
            # Add images as inline data
            for b64_img in base64_images:
                content_parts.append({"mime_type": "image/jpeg", "data": b64_img})
            # Finally the user's actual question
            content_parts.append(question)

            if extra_logging:
                logging.info(f"Request for model={model_name} question={question}")

            # Make the call to Gemini
            response = model.generate_content(content_parts)

            # If we get here, the call succeeded, so reset the delay for the next call
            if attempt > 0:
                logging.info(f"Successfully recovered after {attempt + 1} attempts")
            return response.text.strip()

        except Exception as exc:
            last_exception = exc
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                logging.warning(
                    f"Error on attempt {attempt + 1}/{max_retries} for model={model_name}: {exc}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff

    # If we get here, we've exhausted all retries
    logging.error(
        f"Final error calling Gemini API for model={model_name}: {last_exception}"
    )
    return f"ERROR: {last_exception}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Parallel questions to Gemini for images, logging debug info."
    )
    parser.add_argument(
        "--base_dir",
        default="src/trajectories",
        help="Base directory with subfolders (e.g., 0001-run). Each should have an images/ subdir.",
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

    # We'll store an in-memory map of run_dir -> csv_name -> row data for final writing
    csv_data_tracker = {}
    # {
    #   run_dir_path: {
    #       csv_file_name: {
    #           "fieldnames": [...],
    #           "rows": [...],
    #       },
    #       ...
    #   },
    #   ...
    # }

    # Helper to load additional annotations from images/annotations.csv
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

    # We will build a list of requests to run in parallel
    # Each request is a dictionary with everything needed to call Gemini
    # e.g. {
    #   "run_dir": str,
    #   "csv_name": str,
    #   "row_index": int,
    #   "model_name": str,
    #   "question": str,
    #   "base64_images": [...],
    #   "annotations_text_block": str
    # }
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

        # Collect images
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

        # Prepare an entry in csv_data_tracker for this run_dir
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

            # Ensure columns for our two Gemini models
            if MODEL_COLUMN_PRO not in fieldnames:
                fieldnames.append(MODEL_COLUMN_PRO)

            csv_data_tracker[run_dir][csv_name] = {
                "fieldnames": fieldnames,
                "rows": rows,
            }

            # Build parallel requests (two per row if not already answered)
            for i, row in enumerate(rows):
                question = row.get("question", "").strip()
                if not question:
                    continue

                # If the cell is empty OR starts with 'ERROR:', queue for re-run:
                existing_pro = row.get(MODEL_COLUMN_PRO, "")
                if not existing_pro or existing_pro.startswith("ERROR:"):
                    all_requests.append(
                        {
                            "run_dir": run_dir,
                            "csv_name": csv_name,
                            "row_index": i,
                            "model_name": MODEL_COLUMN_PRO,
                            "question": question,
                            "base64_images": base64_images,
                            "annotations_text_block": annotations_text_block,
                        }
                    )

    if not all_requests:
        logging.info("No new requests to process. Exiting.")
        return

    max_workers = 4
    # We will now process all_requests in parallel with a max of 2 workers
    logging.info(
        f"Processing {len(all_requests)} requests in parallel, up to {max_workers} at a time..."
    )
    results_map = {}  # (run_dir, csv_name, row_index, model_name) -> answer

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_request = {}
        for req in all_requests:
            future = executor.submit(
                call_gemini_api,
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

    # Now update CSV files with results
    for run_dir, csv_dict in csv_data_tracker.items():
        for csv_name, data_obj in csv_dict.items():
            fieldnames = data_obj["fieldnames"]
            rows = data_obj["rows"]

            for i, row in enumerate(rows):
                # If we have results for either model, fill them in
                key = (run_dir, csv_name, i, MODEL_COLUMN_PRO)
                if key in results_map:
                    row[MODEL_COLUMN_PRO] = results_map[key]

            # Now write out the updated CSV
            csv_path = os.path.join(run_dir, "images", csv_name)
            with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            logging.info(f"Updated {csv_path} with Gemini results.")

    logging.info("All requests completed and CSVs updated. Done.")


if __name__ == "__main__":
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    main()

