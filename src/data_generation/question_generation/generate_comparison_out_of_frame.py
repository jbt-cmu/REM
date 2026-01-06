import csv
import glob
import os
import random

from src.data_generation.question_generation.common_utils import (
    read_annotations,
    COLORS,
    SHAPES,
    pick_unseen_color,
    pick_unseen_shape,
    pick_unseen_color_shape,
    pick_seen_color,
    pick_seen_shape,
    pick_seen_color_shape,
    comparison_answer,
    get_color_frames,
    get_shape_frames,
    get_color_shape_frames,
    get_color_count_val,
    get_shape_count_val,
    get_color_shape_count_val,
)

NUM_COMPARISON_QUESTIONS = 8
P_COMP_UNSEEN = 0.0

# For pluralizing shapes in our labels
SHAPE_PLURAL_MAP = {
    "cone": "cones",
    "sphere": "spheres",
    "cuboid": "cuboids",
}


def pick_item_by_type(item_type, color_count, shape_count, color_shape_count):
    """
    Return (item_type, item_value), e.g. ("color", "red"), ("shape", "cone"),
    or ("color_shape", ("red", "cone")).

    Incorporates 'unseen' logic with probability P_COMP_UNSEEN.
    """
    if item_type == "color":
        if random.random() < P_COMP_UNSEEN:
            c_unseen, _ = pick_unseen_color(color_count)
            if c_unseen is not None:
                return ("color", c_unseen)
        return ("color", pick_seen_color(color_count))

    elif item_type == "shape":
        if random.random() < P_COMP_UNSEEN:
            s_unseen, _ = pick_unseen_shape(shape_count)
            if s_unseen is not None:
                return ("shape", s_unseen)
        return ("shape", pick_seen_shape(shape_count))

    elif item_type == "color_shape":
        if random.random() < P_COMP_UNSEEN:
            (combo_unseen, _count) = pick_unseen_color_shape(color_shape_count)
            if combo_unseen is not None:
                return ("color_shape", combo_unseen)
        return ("color_shape", pick_seen_color_shape(color_shape_count))

    else:
        raise ValueError(f"Unknown item type: {item_type}")


def get_count_by_type(item_type, item_value, data):
    """Fetch the count for a color, shape, or color+shape from the annotation data."""
    if item_type == "color":
        return get_color_count_val(data, item_value)
    elif item_type == "shape":
        return get_shape_count_val(data, item_value)
    elif item_type == "color_shape":
        c, s = item_value
        return get_color_shape_count_val(data, c, s)
    else:
        raise ValueError(f"Unknown item type: {item_type}")


def get_frames_by_type(item_type, item_value, data):
    """Fetch the frames in which a color, shape, or color+shape appears."""
    if item_type == "color":
        return get_color_frames(data, item_value)
    elif item_type == "shape":
        return get_shape_frames(data, item_value)
    elif item_type == "color_shape":
        c, s = item_value
        return get_color_shape_frames(data, c, s)
    else:
        raise ValueError(f"Unknown item type: {item_type}")


def label_item(item_type, item_value):
    """
    Convert the chosen item into a human-readable label:
      - color -> "red objects"
      - shape -> "cones" (pluralized via SHAPE_PLURAL_MAP)
      - color_shape -> "red cones"
    """
    if item_type == "color":
        return f"{item_value} objects"

    elif item_type == "shape":
        shape_plural = SHAPE_PLURAL_MAP.get(item_value, f"{item_value}s")
        return shape_plural

    elif item_type == "color_shape":
        c, s = item_value
        shape_plural = SHAPE_PLURAL_MAP.get(s, f"{s}s")
        return f"{c} {shape_plural}"

    else:
        raise ValueError(f"Unknown item type: {item_type}")


def main():
    print("[INFO] Generating out-of-frame comparison questions.")
    annotation_paths = glob.glob("src/trajectories/*-run/images/annotations.csv")

    for csv_path in annotation_paths:
        data = read_annotations(csv_path)

        comparison_out_of_frame = []

        # Extract counts from annotation data
        color_count = data["color_count"]
        shape_count = data["shape_count"]
        color_shape_count = data["color_shape_count"]

        attempts = 0
        # Keep generating until we have enough out-of-frame questions
        # or we exceed some max attempts to avoid infinite loops
        used_questions = set()


        # [ADDED] ------------------------------------------------------------------------------------
        # Try multiple times to generate a question from duplicates in color_shape_count
        duplicates = [(cs, cnt) for cs, cnt in color_shape_count.items() if cnt >= 2]
        max_duplicate_attempts = 20
        duplicate_attempts = 0
        found_duplicate_question = False

        while duplicate_attempts < max_duplicate_attempts and not found_duplicate_question:
            duplicate_attempts += 1
            if not duplicates:
                break  # no duplicates to try

            # If there's more than one duplicate entry, pick two at random from that list
            if len(duplicates) > 1:
                (cs_val_1, count_1), (cs_val_2, count_2) = random.sample(duplicates, 2)
            else:
                # Only one duplicate pair exists. Compare it to another random color_shape item if possible.
                (cs_val_1, count_1) = duplicates[0]
                # get some other color_shape item
                possible_others = [(cs, c) for cs, c in color_shape_count.items() if cs != cs_val_1]
                if not possible_others:
                    break
                (cs_val_2, count_2) = random.choice(possible_others)

            label1 = label_item("color_shape", cs_val_1)
            label2 = label_item("color_shape", cs_val_2)
            question = f"Are there more {label1} or {label2}?"
            if question in used_questions:
                continue  # skip if we've tried this exact question

            frames1 = get_frames_by_type("color_shape", cs_val_1, data)
            frames2 = get_frames_by_type("color_shape", cs_val_2, data)

            if not (frames1 & frames2):  # No in-frame overlap
                answer = comparison_answer(count_1, count_2, label1, label2)
                comparison_out_of_frame.append((question, answer))
                used_questions.add(question)
                found_duplicate_question = True
                break
        # [ADDED] ------------------------------------------------------------------------------------



        while len(comparison_out_of_frame) < NUM_COMPARISON_QUESTIONS and attempts < 1000:
            attempts += 1

            # 1) Pick two item types randomly
            type1 = random.choice(["color", "shape", "color_shape"])
            type2 = random.choice(["color", "shape", "color_shape"])

            # 2) Pick actual items
            item_type_1, item_val_1 = pick_item_by_type(
                type1, color_count, shape_count, color_shape_count
            )
            item_type_2, item_val_2 = pick_item_by_type(
                type2, color_count, shape_count, color_shape_count
            )

            # 3) Avoid identical picks if both are same type+value
            tries = 0
            while (item_type_1 == item_type_2 and item_val_1 == item_val_2) and tries < 10:
                item_type_2, item_val_2 = pick_item_by_type(
                    type2, color_count, shape_count, color_shape_count
                )
                tries += 1

            # 4) Get their counts
            count1 = get_count_by_type(item_type_1, item_val_1, data)
            count2 = get_count_by_type(item_type_2, item_val_2, data)

            # 5) Build labels and create a question
            label1 = label_item(item_type_1, item_val_1)
            label2 = label_item(item_type_2, item_val_2)
            question = f"Are there more {label1} or {label2}?"

            if question in used_questions:
                continue  # Skip this one, already used
            answer = comparison_answer(count1, count2, label1, label2)

            # 6) Get frames to ensure the items do NOT appear together
            frames1 = get_frames_by_type(item_type_1, item_val_1, data)
            frames2 = get_frames_by_type(item_type_2, item_val_2, data)

            # 7) We only keep the question if there's NO overlap of frames
            if not (frames1 & frames2):
                comparison_out_of_frame.append((question, answer))
                used_questions.add(question)

        # Write out the CSV
        out_csv = os.path.join(
            os.path.dirname(csv_path), "comparison_questions_out_of_frame.csv"
        )
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            for q, a in comparison_out_of_frame:
                writer.writerow([q, a])

        print(
            f"[INFO] Wrote {len(comparison_out_of_frame)} out-of-frame comparisons to {out_csv}"
        )


if __name__ == "__main__":
    main()

