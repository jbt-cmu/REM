import argparse
import base64
import csv
import logging
import os
import re
import sys
import time

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from PIL import Image

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
client = Anthropic(api_key=api_key)

MODEL_NAME = "claude-3-5-sonnet-latest"
MODEL_COLUMN = "claude-3.5-sonnet-latest"

SYSTEM_PROMPT = (
    "You are an agent walking through an environment in Blender. "
    "You have a sequence of images taken at each step, possibly containing objects. "
    "You will receive a question. Please answer using only the details in the images, "
    "and give concise, direct answers following these guidelines."
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


def resize_and_encode_image(image_path, max_file_size_mb=5, quality=85):
    """
    Shrink an image (if needed) and convert it to base64-encoded JPG.
    Returns the string or None if we encountered an error.
    """
    if not os.path.exists(image_path):
        return None

    try:
        from io import BytesIO

        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Example: limit dimension to 1024x1024
            img.thumbnail((1024, 1024), Image.LANCZOS)

            buffer = BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=quality)
            file_size = buffer.tell()
            max_bytes = max_file_size_mb * 1024 * 1024

            while file_size > max_bytes and quality > 10:
                quality = int(quality * 0.8)
                buffer.seek(0)
                buffer.truncate(0)
                img.save(buffer, format="JPEG", optimize=True, quality=quality)
                file_size = buffer.tell()

            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

    except Exception as e:
        logging.warning(f"Failed to process image {image_path}: {e}")
        return None


def get_custom_id(csv_name: str, row_index: int) -> str:
    """
    Sanitize the CSV filename consistently, then append __row_{row_index}.
    This ensures the string matches in both the creation phase and the retrieval phase.
    """
    safe_csv_name = re.sub(r"[^a-zA-Z0-9_-]", "_", csv_name)
    safe_csv_name = safe_csv_name[:40]  # up to 40 so we can add suffix
    result = f"{safe_csv_name}__row_{row_index}"
    return result[:64]  # final guard if total >64


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Batch questions to Anthropic with images, logging debug info."
    )
    parser.add_argument(
        "--base_dir",
        default="src/trajectories",
        help="Base directory with subfolders (e.g., 0001-run). Each should have an images/ subdir.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        logging.error(f"{base_dir} does not exist.")
        sys.exit(1)

    # Find all run directories with an "images" subfolder
    run_dirs = sorted(
        [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
    )

    for i in range(0, len(run_dirs), 20):
        chunk_of_run_dirs = run_dirs[i : i + 20]
        for run_dir in chunk_of_run_dirs:
            images_dir = os.path.join(run_dir, "images")
            if not os.path.isdir(images_dir):
                logging.info(f"No images/ folder found in {run_dir}, skipping.")
                continue

            logging.info(f"\nProcessing run directory: {run_dir}")

            # Gather image files, convert to base64
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

            logging.info(f"Total images included: {len(base64_images)}")

            # We'll accumulate a list of "Request" objects for this run_dir.
            batch_requests = []

            # For each known CSV, read the questions, build requests
            for csv_name in QUESTION_CSVS:
                csv_path = os.path.join(images_dir, csv_name)
                if not os.path.isfile(csv_path):
                    continue  # skip if CSV is not present

                with open(csv_path, "r", encoding="utf-8", newline="") as f_in:
                    reader = csv.DictReader(f_in)
                    if not reader.fieldnames:
                        continue
                    rows = list(reader)
                    fieldnames = list(reader.fieldnames)

                # Ensure result column
                if MODEL_COLUMN not in fieldnames:
                    fieldnames.append(MODEL_COLUMN)

                # Build a request for each question row if question is non-empty
                for i, row in enumerate(rows):
                    question = row.get("question", "").strip()
                    if not question:
                        row[MODEL_COLUMN] = ""
                        logging.debug(
                            f"Row {i} in '{csv_name}' has no question => skipping."
                        )
                        continue

                    # Merge question text + all images in single user role
                    user_contents = [{"type": "text", "text": question}]
                    for img_b64 in base64_images:
                        user_contents.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            }
                        )

                    custom_id = get_custom_id(csv_name, i)
                    req_obj = Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=MODEL_NAME,
                            max_tokens=512,
                            temperature=0.0,
                            system=SYSTEM_PROMPT,
                            messages=[
                                {
                                    "role": "user",
                                    "content": user_contents,
                                }
                            ],
                        ),
                    )
                    batch_requests.append(req_obj)
                    logging.debug(
                        f"Request built for row {i} in '{csv_name}' => custom_id={custom_id}"
                    )

                # We'll rewrite the CSV after the batch is done, so keep all rows in memory.

            # If no requests, skip
            if not batch_requests:
                logging.info(f"No questions found for {run_dir}; skipping.")
                continue

            logging.info(f"Building a batch with {len(batch_requests)} requests.")
            # Submit the batch
            batch_obj = client.messages.batches.create(requests=batch_requests)
            batch_id = batch_obj.id
            logging.info(
                f"Created batch {batch_id}, in {batch_obj.processing_status} status."
            )

            # Poll for completion
            while True:
                batch_check = client.messages.batches.retrieve(batch_id)
                status = batch_check.processing_status
                counts = batch_check.request_counts
                logging.info(f"Batch {batch_id} => {status}, {counts}")
                # Exit loop if final
                if status in ["completed", "errored", "canceled", "expired", "ended"]:
                    break
                time.sleep(2)

            # Retrieve final results
            results_dict = {}
            for result_item in client.messages.batches.results(batch_id):
                cid = result_item.custom_id
                # result_item.result is an object with .type, .message, etc.
                if result_item.result.type != "succeeded":
                    # Log each errored request with details so you can see what went wrong
                    logging.error(
                        f"Request {cid} failed. "
                        f"type={result_item.result.type}, "
                        f"error={result_item.result.error}"
                    )
                    # Store a short placeholder in the CSV
                    results_dict[cid] = f"ERROR: {result_item.result.type or 'unknown'}"
                else:
                    msg_obj = result_item.result.message
                    # msg_obj.content is a list of { "type": "text", "text": "..."}
                    final_text = "".join(
                        chunk.text for chunk in msg_obj.content if chunk.type == "text"
                    ).strip()
                    logging.info(
                        f"[DEBUG] Response for custom_id={cid} => {final_text!r}"
                    )
                    results_dict[cid] = final_text

            # Now write answers back to each CSV
            for csv_name in QUESTION_CSVS:
                csv_path = os.path.join(images_dir, csv_name)
                if not os.path.isfile(csv_path):
                    continue

                with open(csv_path, "r", encoding="utf-8", newline="") as f_in:
                    reader = csv.DictReader(f_in)
                    rows = list(reader)
                    fieldnames = list(reader.fieldnames)

                if MODEL_COLUMN not in fieldnames:
                    fieldnames.append(MODEL_COLUMN)

                for i, row in enumerate(rows):
                    cid = get_custom_id(csv_name, i)
                    answer = results_dict.get(cid, "")
                    row[MODEL_COLUMN] = answer
                    logging.debug(
                        f"Writing answer for row {i} in '{csv_name}' => "
                        f"custom_id={cid}, answer={answer!r}"
                    )

                # Write CSV
                with open(csv_path, "w", encoding="utf-8", newline="") as f_out:
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                # Optional extra step: read it back to confirm
                with open(csv_path, "r", encoding="utf-8") as confirm_f:
                    lines = confirm_f.read().splitlines()
                logging.debug(
                    f"Just-wrote CSV contents for '{csv_path}':\n"
                    + "\n".join(lines[: min(10, len(lines))])
                )

                logging.info(f"Updated {csv_path} with {MODEL_COLUMN} results.")

            logging.info(f"Done with run folder: {run_dir}")

        logging.info("All runs processed, done.")


if __name__ == "__main__":
    main()

