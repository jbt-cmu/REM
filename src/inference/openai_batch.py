import argparse
import base64
import csv
import json
import logging
import os
import re
import sys
import time
from io import BytesIO

from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=api_key)
from PIL import Image

MODEL_NAME = "gpt-4o"
MODEL_COLUMN = "gpt-4o"

SYSTEM_PROMPT = (
    "You are an agent walking through an environment in Blender. "
    "You will receive a series of images, each taken after taking an action in the environment (either moving straight or turning 15 degrees left/right). "
    "You will also receive a question that you must answer correctly after seeing all images. "
    "You will see objects with a shape and a color. The possible shapes include cuboid, cone, sphere. "
    "The possible colors include red, green, blue, yellow, purple, brown, black, orange. "
    "Please answer the question based on the set of images."
    "Answer as concisely as possible, usually only a single word. If you're asked about a true/false question, "
    "answer with 'yes' or 'no' only. If it's a question where you're asked to compare the number of objects, respond only with whichever object there are more of, or equal, if there are the same number of objects"
    "If you're asked to count objects, answer only with the number (as a number, not in english) of objects you see."
    "If you're asked whether you saw something before, after, or at the same time as another object, "
    "answer only with 'before', 'after', or 'same time' only. If the first time you see an object is in an image before another object, it comes before (and the other comes after). If two objects appear in the same frame together for their first viewing, its same time"
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

# We define the question CSV files we want to process in each run directory:
QUESTION_CSVS = [
    "comparison_questions_in_frame.csv",
    "comparison_questions_out_of_frame.csv",
    "left_right_questions.csv",
    "number_questions.csv",
    "order_preserving_questions.csv",
]

global ALL_BATCHES
ALL_BATCHES = (
    []
)  # each entry: { "batch_id": str, "requests_subset": [...], "processed": False }


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

            # Resize image to 960x640 for better compatibility
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


def get_custom_id(run_dir: str, csv_name: str, row_index: int) -> str:
    """
    Sanitize the run_dir and CSV filename, then append __row_{row_index}.
    This ensures uniqueness across multiple run directories.
    """
    safe_run_dir = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.basename(run_dir))
    safe_run_dir = safe_run_dir[:50]  # up to 20 so we can add suffix

    safe_csv_name = re.sub(r"[^a-zA-Z0-9_-]", "_", csv_name)
    safe_csv_name = safe_csv_name[:50]  # up to 20 so we can add suffix

    result = f"{safe_run_dir}__{safe_csv_name}__row_{row_index}"
    return result[:64]  # final guard if total >64


def create_batch(requests_subset, base_dir):
    """
    Write requests_subset to JSONL, upload to OpenAI, create a batch.
    Return the new batch_id so we can track it later.
    """
    if not requests_subset:
        return None

    requests_jsonl_path = os.path.join(base_dir, "batch_input.jsonl")
    logging.info(f"Creating JSONL file for {len(requests_subset)} requests.")

    with open(requests_jsonl_path, "w", encoding="utf-8") as f_out:
        for item in requests_subset:
            request_line = {
                "custom_id": item["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": item["messages"],
                    "max_tokens": 512,
                    "temperature": 0.0,
                },
            }
            f_out.write(json.dumps(request_line) + "\n")

    # Upload to OpenAI
    logging.info("Uploading JSONL file...")
    with open(requests_jsonl_path, "rb") as f_in:
        uploaded_file = client.files.create(file=f_in, purpose="batch")
    input_file_id = uploaded_file.id

    # Create the batch
    logging.info("Creating new batch in OpenAI...")
    batch_obj = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch_obj.id
    logging.info(f"Created batch {batch_id}, status: {batch_obj.status}")

    # Cleanup local JSONL (no longer needed after we start the batch)
    if os.path.exists(requests_jsonl_path):
        os.remove(requests_jsonl_path)

    return batch_id


def poll_all_batches(csv_data_tracker):
    """
    Loop over ALL_BATCHES, keep polling each batch that isn't processed yet.
    As soon as a batch is completed, parse & write results to CSV,
    mark it as processed, and move on. Stop when all are processed or have final states.
    """
    while True:
        all_finished = True
        for batch_info in ALL_BATCHES:
            if batch_info["processed"]:
                continue  # we've already written results for this batch

            batch_id = batch_info["batch_id"]
            batch_check = client.batches.retrieve(batch_id)
            status = batch_check.status
            logging.info(f"Batch {batch_id} => status={status}")

            if status in ["completed", "failed", "expired", "cancelled"]:
                # This batch is done - parse results if it's completed or partially completed
                if status == "completed":
                    parse_and_write_results(
                        batch_info["requests_subset"], batch_check, csv_data_tracker
                    )
                else:
                    logging.warning(
                        f"Batch {batch_id} ended with status '{status}'. No results to parse."
                    )

                batch_info["processed"] = True
            else:
                all_finished = False

        if all_finished:
            break  # All done, exit poll loop

        time.sleep(10)  # Wait a bit before checking statuses again


def parse_and_write_results(requests_subset, batch_check, csv_data_tracker):
    """
    Once a batch is completed, fetch output lines, match them to requests_subset by custom_id,
    and store the answers in csv_data_tracker. Then write updated CSVs to disk.
    """
    output_file_id = batch_check.output_file_id
    error_file_id = batch_check.error_file_id

    results_dict = {}

    # Retrieve output
    try:
        output_content = client.files.content(output_file_id).read()
        output_lines = output_content.decode("utf-8").splitlines()
        for line in output_lines:
            entry = json.loads(line)
            cid = entry["custom_id"]
            response_body = entry["response"]["body"]
            if response_body and "choices" in response_body:
                final_text = response_body["choices"][0]["message"]["content"].strip()
                results_dict[cid] = final_text
    except Exception as exc:
        logging.error(f"Error fetching/parsing output file: {exc}")

    # Retrieve error file if needed
    if error_file_id:
        try:
            error_content = client.files.content(error_file_id).read()
            error_lines = error_content.decode("utf-8").splitlines()
            for line in error_lines:
                entry = json.loads(line)
                cid = entry.get("custom_id", "")
                err = entry.get("error", {})
                if cid:
                    results_dict[cid] = f"ERROR: {err.get('message', 'unknown')}"
        except Exception as e:
            logging.warning(f"Failed to parse error file: {e}")

    # Write results into csv_data_tracker
    for item in requests_subset:
        cid = item["custom_id"]
        run_dir_path = item["run_dir"]
        csv_name = item["csv_name"]
        row_index = item["row_index"]
        answer = results_dict.get(cid, "")
        row = csv_data_tracker[run_dir_path][csv_name]["rows"][row_index]
        row[MODEL_COLUMN] = answer

    # Now flush updated CSVs to disk
    touched = set()
    for item in requests_subset:
        run_dir_path = item["run_dir"]
        csv_name = item["csv_name"]
        if (run_dir_path, csv_name) in touched:
            continue
        touched.add((run_dir_path, csv_name))

        data = csv_data_tracker[run_dir_path][csv_name]
        fieldnames = data["fieldnames"]
        rows = data["rows"]
        csv_path = os.path.join(run_dir_path, "images", csv_name)

        with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logging.info(f"Updated {csv_path} with {MODEL_COLUMN} results.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Batch questions to OpenAI GPT for images, logging debug info."
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

    # Convert the string argument to a boolean
    extra_logging = args.logging.strip().lower() == "true"

    global base_dir
    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        logging.error(f"{base_dir} does not exist.")
        sys.exit(1)

    # Collect all run directories in a single list
    run_dirs = sorted(
        [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    )

    global current_batch_size
    current_batch_size = 0

    requests_to_process = (
        []
    )  # each item includes "run_dir", "csv_name", "row_index", etc.

    # We'll store an in-memory map of run_dir -> csv_name -> row data for final writing
    csv_data_tracker = {}
    # Structure:
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

    # Keep track of approximate JSON size in bytes to cap at ~200 MB
    max_batch_bytes = 180 * 1024 * 1024  # 200 MB

    # Prepare a helper to read annotations.csv if it exists
    def load_annotations(images_dir):
        annotations_file = os.path.join(images_dir, "annotations.csv")
        annotations_list = []
        if os.path.isfile(annotations_file):
            logging.info(f"Found annotations.csv in {images_dir}, loading...")
            with open(annotations_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # For example, if there's a column "annotation"
                    ann_text = row.get("annotation", "").strip()
                    if ann_text:
                        annotations_list.append(ann_text)
        else:
            logging.info(f"No annotations.csv in {images_dir}, skipping.")
        return annotations_list

    # Build the requests and store CSV data
    for run_dir in run_dirs:
        images_dir = os.path.join(run_dir, "images")
        if not os.path.isdir(images_dir):
            logging.info(f"No images/ folder found in {run_dir}, skipping.")
            continue

        logging.info(f"\nGathering data from run directory: {run_dir}")

        # Load all annotations for this directory (from images/annotations.csv if present)
        annotations_list = load_annotations(images_dir)
        # Combine them into a single block of text
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

        # Prepare an entry in csv_data_tracker
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

            # Ensure result column
            if MODEL_COLUMN not in fieldnames:
                fieldnames.append(MODEL_COLUMN)

            csv_data_tracker[run_dir][csv_name] = {
                "fieldnames": fieldnames,
                "rows": rows,
            }

            # Build requests for each question
            for i, row in enumerate(rows):
                question = row.get("question", "").strip()
                if not question:
                    continue

                # Insert the combined annotation block after the question
                # but before sending the images
                user_content_parts = [
                    {"type": "text", "text": question},
                    {"type": "text", "text": annotations_text_block},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        }
                        for img_b64 in base64_images
                    ],
                ]
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content_parts},
                ]

                # NEW LOGGING, conditional upon --logging=true
                if extra_logging:
                    logging.info(f"Constructing prompt for question {i}: {question}")
                    # Uncomment below if you also want to see image filenames
                    # logging.info(f"Image files for prompt: {image_files}")
                    logging.info(
                        "Full request messages:\n%s", json.dumps(messages, indent=2)
                    )

                custom_id = get_custom_id(run_dir, csv_name, i)

                # Estimate line size to ensure we don't exceed 200 MB
                new_line_bytes = (
                    len(
                        json.dumps(
                            {"custom_id": custom_id, "body": {"messages": messages}}
                        ).encode("utf-8")
                    )
                    + 1
                )

                # If adding another request would exceed 50,000 or 200 MB, flush the current batch first
                if (len(requests_to_process) >= 50000) or (
                    current_batch_size + new_line_bytes > max_batch_bytes
                ):
                    logging.info(
                        "Reached either 50,000-query limit or ~200 MB size. Sending current batch..."
                    )
                    batch_id = create_batch(requests_to_process, base_dir)

                    ALL_BATCHES.append(
                        {
                            "batch_id": batch_id,
                            "requests_subset": list(requests_to_process),  # copy
                            "processed": False,
                        }
                    )

                    requests_to_process.clear()
                    current_batch_size = 0

                requests_to_process.append(
                    {
                        "run_dir": run_dir,
                        "csv_name": csv_name,
                        "row_index": i,
                        "custom_id": custom_id,
                        "messages": messages,
                    }
                )
                current_batch_size += new_line_bytes

    # If any leftover requests remain, process them
    if requests_to_process:
        batch_id = create_batch(requests_to_process, base_dir)
        ALL_BATCHES.append(
            {
                "batch_id": batch_id,
                "requests_subset": list(requests_to_process),
                "processed": False,
            }
        )
        requests_to_process.clear()

    logging.info("All runs processed, done.")

    # We created all batches in a loop above, now start the poll loop:
    poll_all_batches(csv_data_tracker)

    logging.info("All batches are finished, done.")


if __name__ == "__main__":
    main()

