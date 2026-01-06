import argparse
import csv
import json
import logging
import os
import re
import string
from typing import Optional, Set, Tuple


def compute_average_occlusion(annotations_csv_path):
    if not os.path.isfile(annotations_csv_path):
        return 0.0, 0

    total_percentages = 0.0
    line_count = 0
    objects_seen = set()

    with open(annotations_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vis_str = row.get("visible_objects", "")
            matches = re.findall(r"([\w_]+)\s*\(([\d\.]+)%\)", vis_str)
            if matches:
                line_count += 1
                for obj_name, percent_str in matches:
                    if obj_name.lower() == "groundplane":
                        continue
                    objects_seen.add(obj_name)
                    try:
                        total_percentages += float(percent_str)
                    except ValueError:
                        pass

    if line_count == 0:
        return 0.0, len(objects_seen)
    else:
        avg = total_percentages / line_count
        return avg, len(objects_seen)


def compute_max_objects_per_frame(annotations_csv_path):
    """
    Computes the maximum number of objects visible in any single frame.

    Args:
        annotations_csv_path: Path to the annotations CSV file

    Returns:
        int: Maximum number of objects in a single frame
    """
    if not os.path.isfile(annotations_csv_path):
        return 0

    max_objects = 0

    with open(annotations_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vis_str = row.get("visible_objects", "")
            matches = re.findall(r"([\w_]+)\s*\(([\d\.]+)%\)", vis_str)

            # Count objects in this frame (excluding GroundPlane)
            frame_objects = sum(
                1 for obj_name, _ in matches if obj_name.lower() != "groundplane"
            )

            # Update max if this frame has more objects
            if frame_objects > max_objects:
                max_objects = frame_objects

    return max_objects


def compute_max_objects_of_type_per_frame(annotations_csv_path, question_text):
    """
    For number questions, computes the maximum number of specific objects mentioned in the question
    that appear in the same frame.
    
    Args:
        annotations_csv_path: Path to the annotations CSV file
        question_text: The question text to extract the object type from
        
    Returns:
        int: Maximum number of matching objects in a single frame
    """
    if not os.path.isfile(annotations_csv_path):
        return 0
        
    # Extract potential colors and shapes from the question
    q_lower = question_text.lower()
    colors = ["brown", "yellow", "red", "green", "blue", "purple", "black", "orange"]
    shapes = ["sphere", "cone", "cuboid"]
    
    mentioned_colors = [color for color in colors if color in q_lower]
    mentioned_shapes = [shape for shape in shapes if shape in q_lower]
    
    # If no colors or shapes are mentioned, return 0
    if not mentioned_colors and not mentioned_shapes:
        return 0
        
    max_objects_in_any_frame = 0
    
    with open(annotations_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vis_str = row.get("visible_objects", "")
            matches = re.findall(r"([\w_]+)\s*\(([\d\.]+)%\)", vis_str)
            
            # Count objects in this frame that match any mentioned color or shape
            frame_objects = 0
            
            for obj_name, _ in matches:
                obj_lower = obj_name.lower()
                
                # Skip GroundPlane
                if obj_lower == "groundplane":
                    continue
                    
                # Check if object matches any mentioned color or shape
                matches_color = not mentioned_colors or any(color in obj_lower for color in mentioned_colors)
                matches_shape = not mentioned_shapes or any(shape in obj_lower for shape in mentioned_shapes)
                
                if matches_color and matches_shape:
                    frame_objects += 1
            
            # Update max if this frame has more matching objects
            if frame_objects > max_objects_in_any_frame:
                max_objects_in_any_frame = frame_objects
    
    return max_objects_in_any_frame


def compute_duplicates_info(annotations_csv_path):
    """
    Computes duplicate information from the annotations CSV over the entire trajectory.

    For each row, we parse the visible_objects field (using the same regex as elsewhere) and,
    for each object (ignoring "GroundPlane"), we split the object name using the last underscore.
    For example, "black_sphere_5" becomes:
       - canonical name: "black_sphere"
       - suffix: "5"

    For each canonical object we build a set of unique suffixes encountered. The duplicate count
    for that object is defined as (number of unique suffixes - 1), so that the first instance is not
    counted as a duplicate.

    The function returns:
       - total_number_of_duplicates_seen: sum_{for each canonical object} (unique_count - 1)
       - max_number_of_duplicates_of_same_type: maximum over canonical objects of (unique_count - 1)
    """
    if not os.path.isfile(annotations_csv_path):
        return 0, 0

    unique_instances = {}  # key: canonical object, value: set of suffixes
    with open(annotations_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vis_str = row.get("visible_objects", "")
            # Use the same regex to capture the object names and their percentages.
            matches = re.findall(r"([\w_]+)\s*\(([\d\.]+)%\)", vis_str)
            for obj_name, _ in matches:
                if obj_name.lower() == "groundplane":
                    continue
                # Split the object name on the last underscore.
                canonical, sep, suffix = obj_name.rpartition("_")
                # If no underscore is found, treat the entire name as canonical.
                if not sep:
                    canonical = obj_name
                    suffix = ""
                # Initialize the set if needed.
                if canonical not in unique_instances:
                    unique_instances[canonical] = set()
                # Add the suffix (even if it is empty)
                unique_instances[canonical].add(suffix)

    total_duplicates_seen = 0
    max_duplicates_of_same_type = 0
    for canonical, suffixes in unique_instances.items():
        # The duplicate count for a canonical object is (number of unique suffixes - 1)
        duplicates = max(len(suffixes) - 1, 0)
        total_duplicates_seen += duplicates
        if duplicates > max_duplicates_of_same_type:
            max_duplicates_of_same_type = duplicates

    return total_duplicates_seen, max_duplicates_of_same_type


def get_scene_info(scene_config_path, images_path):
    if not os.path.isfile(scene_config_path):
        return {
            "num_objects_in_scene": 0,
            "trajectory_length": 0,
            "number_of_duplicates": 0,
            "total_number_of_duplicates_seen": 0,
            "max_number_of_duplicates_of_same_type": 0,
        }

    with open(scene_config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scene_settings = data.get("scene_settings", {})
    objects = data.get("objects", [])
    duplicated = scene_settings.get("duplicated_objects", [])

    filtered_objects = [
        obj for obj in objects if obj.get("name", "").lower() != "groundplane"
    ]
    num_objects_in_scene = len(filtered_objects)

    valid_exts = (".png", ".jpg", ".jpeg")
    image_files = [
        fname for fname in os.listdir(images_path) if fname.lower().endswith(valid_exts)
    ]
    trajectory_length = len(image_files)

    num_duplicates = 0
    total_number_of_duplicates_seen = 0
    max_number_of_duplicates_of_same_type = 0
    for d in duplicated:
        dup_count = d.get("duplicate_count", 0)
        total_count = d.get("total_count", dup_count + 1)
        num_duplicates += dup_count
        total_number_of_duplicates_seen += total_count
        if total_count > max_number_of_duplicates_of_same_type:
            max_number_of_duplicates_of_same_type = total_count

    return {
        "num_objects_in_scene": num_objects_in_scene,
        "trajectory_length": trajectory_length,
        "number_of_duplicates": num_duplicates,
        "total_number_of_duplicates_seen": total_number_of_duplicates_seen,
        "max_number_of_duplicates_of_same_type": max_number_of_duplicates_of_same_type,
    }


def load_data_dict(data_dict_path):
    dd = {}
    if not os.path.isfile(data_dict_path):
        return dd

    with open(data_dict_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("type", "").strip()
            k = row.get("key", "").strip()
            v_str = row.get("value", "").strip()
            dd[(t, k)] = v_str
    return dd


def guess_question_type(csv_filename):
    if csv_filename == "comparison_questions_in_frame.csv":
        return "comparison_in_frame"
    elif csv_filename == "comparison_questions_out_of_frame.csv":
        return "comparison_out_of_frame"
    elif csv_filename == "left_right_questions.csv":
        return "left_right"
    elif csv_filename == "number_questions.csv":
        return "number"
    elif csv_filename == "order_preserving_questions.csv":
        return "order_preserving"

    lower = csv_filename.lower()
    if "comparison" in lower:
        return "comparison"
    elif "number" in lower:
        return "number"
    elif "left_right" in lower:
        return "left_right"
    elif "order_preserving" in lower:
        return "order_preserving"
    else:
        return "unknown"


def parse_number_question(question_text, data_dict):
    q_lower = question_text.lower()
    colors = ["brown", "yellow", "red", "green", "blue", "purple", "black", "orange"]
    shapes = ["sphere", "cone", "cuboid"]

    for color in colors:
        for shape in shapes:
            combo = f"{color} {shape}"
            if combo in q_lower:
                data_dict_key = f"{color}_{shape}"
                if ("color_shape_count", data_dict_key) in data_dict:
                    return data_dict[("color_shape_count", data_dict_key)]
    for shape in shapes:
        if shape in q_lower:
            if ("shape_count", shape) in data_dict:
                return data_dict[("shape_count", shape)]
    for color in colors:
        if color in q_lower:
            if ("color_count", color) in data_dict:
                return data_dict[("color_count", color)]
    return ""


MODEL_COLUMNS = [
    "gpt-4o",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "nova-lite-v1",
    "llama-3.2-11b-vision-instruct",
    "gemini-1.5-flash-latest-TEXT-NEW-PROMPT",
]

SPECIAL_ANSWERS = {
    "yes": {"yes", "yep", "true", "y", "yeah"},
    "no": {"no", "n", "nope", "false"},
    "equal": {"equal", "same", "tie"},
    "before": {"before"},
    "after": {"after"},
    "same time": {"sametime", "same time", "simultaneous"},
}


def normalize_text(txt: str) -> str:
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = " ".join(txt.split())
    return txt


def tokenize_str(txt: str) -> Set[str]:
    return set(normalize_text(txt).split())


def parse_more_question(question: str):
    q = question.lower()
    if "are there more " not in q or " or " not in q:
        return None

    parts = q.split("are there more ", 1)
    remainder = parts[1] if len(parts) > 1 else ""
    subparts = remainder.split(" or ", 1)
    if len(subparts) < 2:
        return None

    left = subparts[0].strip().rstrip("?.")
    right = subparts[1].strip().rstrip("?.")
    if left == right:
        return ("__IDENTICAL__", "__IDENTICAL__")
    return (left, right)


def check_are_there_more(
    ground_truth: str, x_option: str, y_option: str, llama_answer: str
) -> str:
    known_colors = {
        "brown",
        "yellow",
        "red",
        "green",
        "blue",
        "purple",
        "black",
        "orange",
    }
    known_objects = {"cuboid", "cone", "sphere"}

    def singularize(token: str) -> str:
        token_singular = token.rstrip("s")
        if token_singular in known_objects:
            return token_singular
        return token

    def parse_option(option: str) -> Tuple[Optional[str], Optional[str], bool]:
        normalized = normalize_text(option)
        tokens = normalized.split()
        color = None
        object_type = None
        for token in tokens:
            if token in known_colors and color is None:
                color = token
            candidate = singularize(token)
            if candidate in known_objects and object_type is None:
                object_type = candidate
        color_only = False
        if object_type is None and any(tok in tokens for tok in ["object", "objects"]):
            color_only = True
        return (color, object_type, color_only)

    def is_subset(
        option_tuple: Tuple[Optional[str], Optional[str], bool],
        answer_tuple: Tuple[Optional[str], Optional[str], bool],
    ) -> bool:
        o_color, o_obj, o_color_only = option_tuple
        a_color, a_obj, a_color_only = answer_tuple
        if a_color is not None and a_color != o_color:
            return False
        if a_obj is not None and a_obj != o_obj:
            return False
        return True

    def equals_option(
        opt1: Tuple[Optional[str], Optional[str], bool],
        opt2: Tuple[Optional[str], Optional[str], bool],
    ) -> bool:
        return opt1 == opt2

    answer_norm = normalize_text(llama_answer)
    gt_norm = normalize_text(ground_truth)

    if answer_norm == gt_norm:
        return "yes"

    if gt_norm in SPECIAL_ANSWERS.get("equal", set()):
        return "yes" if answer_norm in SPECIAL_ANSWERS["equal"] else "no"

    parsed_x = parse_option(x_option)
    parsed_y = parse_option(y_option)
    parsed_answer = parse_option(llama_answer)

    if gt_norm == normalize_text(x_option):
        correct_option = parsed_x
        wrong_option = parsed_y
    elif gt_norm == normalize_text(y_option):
        correct_option = parsed_y
        wrong_option = parsed_x
    else:
        return "HUMAN_REVIEW"

    if equals_option(parsed_answer, correct_option):
        return "yes"

    if is_subset(correct_option, parsed_answer) and not is_subset(
        wrong_option, parsed_answer
    ):
        return "yes"

    if is_subset(wrong_option, parsed_answer) and not equals_option(
        parsed_answer, correct_option
    ):
        return "no"

    return "HUMAN_REVIEW"


def check_semantic_correctness(
    question: str, ground_truth: str, llama_answer: str
) -> str:
    gt_clean = ground_truth.strip().rstrip(".")
    ans_clean = llama_answer.strip().rstrip(string.punctuation).lower()

    if gt_clean.isdigit():
        try:
            m = re.search(r"\d+", ans_clean)
            if m:
                ans_num = int(m.group())
            else:
                return "HUMAN_REVIEW"
        except Exception:
            return "HUMAN_REVIEW"

        if ans_num == int(gt_clean):
            return "yes"
        else:
            return "no"

    if gt_clean.lower() in SPECIAL_ANSWERS and gt_clean.lower() in {
        "before",
        "after",
        "same time",
    }:
        synonyms = SPECIAL_ANSWERS.get(gt_clean.lower(), set())
        if any(word in ans_clean for word in synonyms):
            return "yes"
        else:
            return "no"

    parsed = parse_more_question(question)
    if parsed is not None:
        x_opt, y_opt = parsed
        return check_are_there_more(ground_truth, x_opt, y_opt, llama_answer)

    gt_norm = normalize_text(ground_truth)
    ans_norm = normalize_text(llama_answer)

    if gt_norm in SPECIAL_ANSWERS:
        synonyms = SPECIAL_ANSWERS[gt_norm]
        if ans_norm in synonyms:
            return "yes"
        for other_key, other_syns in SPECIAL_ANSWERS.items():
            if other_key != gt_norm and ans_norm in other_syns:
                return "no"
        return "HUMAN_REVIEW"

    gt_tokens = tokenize_str(ground_truth)
    ans_tokens = tokenize_str(llama_answer)

    if gt_tokens.issubset(ans_tokens):
        return "yes"
    return "no"


def get_comparison_counts(question: str, data_dict) -> tuple[str, str]:
    """Extract counts for the two items being compared in a comparison question."""
    parsed = parse_more_question(question)
    if not parsed:
        return ("", "")

    x_opt, y_opt = parsed
    if x_opt == "__IDENTICAL__":
        return ("", "")

    def get_count(item: str) -> str:
        # Check for color_shape combinations first
        words = item.split()
        if len(words) >= 2:
            color = words[0]
            shape = words[1].rstrip("s")  # Handle plural forms
            key = f"{color}_{shape}"
            if ("color_shape_count", key) in data_dict:
                return data_dict[("color_shape_count", key)]

        # Check for single color
        color = words[0]
        if ("color_count", color) in data_dict:
            return data_dict[("color_count", color)]

        # Check for single shape
        shape = words[-1].rstrip("s")
        if ("shape_count", shape) in data_dict:
            return data_dict[("shape_count", shape)]

        return ""

    return (get_count(x_opt), get_count(y_opt))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Aggregate questions and responses from multiple models into one master file."
    )
    parser.add_argument(
        "--base_dir",
        default="src/trajectories",
        help="Base directory containing the run folders (e.g., round1/, round2/, etc.)",
    )

    args = parser.parse_args()
    base_dir = args.base_dir

    if not os.path.isdir(base_dir):
        logging.error(f"Base directory {base_dir} does not exist.")
        return

    output_csv_path = os.path.join(base_dir, "all_questions_aggregated_final.csv")

    model_answer_fields = []
    model_correctness_fields = []
    for model_col in MODEL_COLUMNS:
        model_answer_fields.append(model_col)
        correctness_col_name = f"answered_correctly_{model_col}"
        model_correctness_fields.append(correctness_col_name)

    fieldnames = [
        "run_directory",
        "csv_name",
        "row_index",
        "question",
        "ground_truth",
        *model_answer_fields,
        *model_correctness_fields,
        "average_occlusion",
        "number_of_objects_it_sees",
        "number_of_objects_in_scene",
        "len_of_trajectory",
        "question_type",
        "number_of_duplicates",
        "total_number_of_duplicates_seen",
        "max_number_of_duplicates_of_same_type",
        "number_of_categories",
        "number_of_colors",
        "number_of_shapes",
        "comp_1",
        "comp_2",
        "max_number_in_single_frame",
    ]

    all_rows = []

    for dirpath, dirnames, filenames in os.walk(base_dir):
        images_path = os.path.join(dirpath, "images")
        if not os.path.isdir(images_path):
            continue

        logging.info(f"Processing directory: {dirpath}")

        scene_config_path = os.path.join(dirpath, "scene_config.json")
        data_dict_path = os.path.join(images_path, "data_dict.csv")
        annotations_csv_path = os.path.join(images_path, "annotations.csv")

        scene_info = get_scene_info(scene_config_path, images_path)
        data_dict = load_data_dict(data_dict_path)
        # Compute additional metrics from the data_dict
        num_categories = sum(
            1 for (dtype, key) in data_dict.keys() if dtype == "color_shape_count"
        )
        num_colors = sum(
            1 for (dtype, key) in data_dict.keys() if dtype == "color_count"
        )
        num_shapes = sum(
            1 for (dtype, key) in data_dict.keys() if dtype == "shape_count"
        )
        avg_occlusion, num_objs_seen = compute_average_occlusion(annotations_csv_path)
        total_duplicates_seen, max_duplicates_of_same_type = compute_duplicates_info(
            annotations_csv_path
        )

        for fn in os.listdir(images_path):
            if not fn.lower().endswith(".csv"):
                continue
            if fn in ["annotations.csv", "data_dict.csv"]:
                continue

            csv_path = os.path.join(images_path, fn)
            question_type = guess_question_type(fn)

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                fieldnames_in_this_csv = reader.fieldnames
                rows_in_this_csv = list(reader)

            for i, row_data in enumerate(rows_in_this_csv):
                question = row_data.get("question", "").strip()
                if not question:
                    continue

                ground_truth_for_all = ""
                if "answer" in fieldnames_in_this_csv:
                    ground_truth_for_all = row_data["answer"].strip()

                final_ground_truth = ground_truth_for_all

                model_answers = {}
                model_correctness = {}
                for model_col in MODEL_COLUMNS:
                    response_text = row_data.get(model_col, "").strip()
                    if response_text:
                        correctness_val = check_semantic_correctness(
                            question, final_ground_truth, response_text
                        )
                    else:
                        correctness_val = "HUMAN_REVIEW"
                    model_answers[model_col] = response_text
                    model_correctness[f"answered_correctly_{model_col}"] = (
                        correctness_val
                    )

                # Get comparison counts if this is a comparison question
                comp_1, comp_2 = "", ""
                if "comparison" in question_type.lower():
                    comp_1, comp_2 = get_comparison_counts(question, data_dict)
                
                # Set max_number_in_single_frame based on question type
                max_number_in_frame = 0
                if question_type == "number":
                    max_number_in_frame = compute_max_objects_of_type_per_frame(
                        annotations_csv_path, question
                    )

                out_row = {
                    "run_directory": dirpath,
                    "csv_name": fn,
                    "row_index": i,
                    "question": question,
                    "ground_truth": ground_truth_for_all,
                    **model_answers,
                    **model_correctness,
                    "average_occlusion": avg_occlusion,
                    "number_of_objects_it_sees": num_objs_seen,
                    "number_of_objects_in_scene": scene_info["num_objects_in_scene"],
                    "len_of_trajectory": scene_info["trajectory_length"],
                    "question_type": question_type,
                    "number_of_duplicates": scene_info["number_of_duplicates"],
                    "total_number_of_duplicates_seen": total_duplicates_seen,
                    "max_number_of_duplicates_of_same_type": max_duplicates_of_same_type,
                    "number_of_categories": num_categories,
                    "number_of_colors": num_colors,
                    "number_of_shapes": num_shapes,
                    "comp_1": comp_1,
                    "comp_2": comp_2,
                    "max_number_in_single_frame": max_number_in_frame,
                }

                all_rows.append(out_row)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logging.info(f"Aggregated results written to: {output_csv_path}")


if __name__ == "__main__":
    main()

