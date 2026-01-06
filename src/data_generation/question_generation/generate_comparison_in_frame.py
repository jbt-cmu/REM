import os
import glob
import csv
import random
from src.data_generation.question_generation.common_utils import (
    read_annotations,
    parse_object_name,
    COLORS,
    SHAPES,
)

NUM_COMPARISON_QUESTIONS = 5
P_COMP_UNSEEN = 0.0  # Must be < 1 for in-frame questions to work

# Map each shape to its plural form
SHAPE_PLURAL_MAP = {
    "cone": "cones",
    "sphere": "spheres",
    "cuboid": "cuboids",
    # Add more if you have other shapes
}

def pick_unseen_color(color_count):
    unseen_candidates = [c for c in COLORS if c not in color_count]
    print(f"[DEBUG] unseen_candidates for color: {unseen_candidates}")

    if unseen_candidates:
        c = random.choice(unseen_candidates)
        return (c, 0)
    return (None, None)

def pick_unseen_shape(shape_count):
    unseen_candidates = [s for s in SHAPES if s not in shape_count]
    if unseen_candidates:
        s = random.choice(unseen_candidates)
        return (s, 0)
    return (None, None)

def pick_unseen_color_shape(color_shape_count):
    unseen_combos = []
    for c in COLORS:
        for s in SHAPES:
            if (c, s) not in color_shape_count:
                unseen_combos.append((c, s))
    if unseen_combos:
        print(f"[DEBUG] unseen_candidates for color_shape: {unseen_combos}")
        return (random.choice(unseen_combos), 0)
    return (None, None)

def pick_seen_color(color_count):
    if color_count:
        c, _ = random.choice(list(color_count.items()))
        return c
    return "red"

def pick_seen_shape(shape_count):
    if shape_count:
        s, _ = random.choice(list(shape_count.items()))
        return s
    return "sphere"

def pick_seen_color_shape(color_shape_count):
    if color_shape_count:
        (c, s), _ = random.choice(list(color_shape_count.items()))
        return (c, s)
    return ("red", "sphere")

def comparison_answer(count_x, count_y, label_x, label_y):
    if count_x > count_y:
        return label_x
    elif count_y > count_x:
        return label_y
    else:
        return "equal"

#
# -- Helper Functions for Cross-Type Comparison --
#
def pick_item_by_type(item_type, color_count, shape_count, color_shape_count):
    """
    Return (picked_item_type, item_value).
    E.g.:
      ("color", "red")
      ("shape", "cone")
      ("color_shape", ("red", "sphere"))

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

def get_count_by_type(item_type, item_value, color_count, shape_count, color_shape_count):
    if item_type == "color":
        return color_count.get(item_value, 0)
    elif item_type == "shape":
        return shape_count.get(item_value, 0)
    elif item_type == "color_shape":
        c, s = item_value
        return color_shape_count.get((c, s), 0)
    else:
        raise ValueError(f"Unknown item type: {item_type}")

def get_frames_by_type(item_type, item_value, color_frames, shape_frames, color_shape_frames):
    if item_type == "color":
        return color_frames.get(item_value, set())
    elif item_type == "shape":
        return shape_frames.get(item_value, set())
    elif item_type == "color_shape":
        c, s = item_value
        return color_shape_frames.get((c, s), set())
    else:
        raise ValueError(f"Unknown item type: {item_type}")


def label_item(item_type, item_value):
    """
    Convert the chosen item into a human-readable label for the question.
    
    - color -> "red objects"
    - shape -> "cones"
    - color_shape -> "red cones"
    """
    if item_type == "color":
        # e.g., "red objects"
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
    annotation_paths = glob.glob("src/trajectories/*-run/images/annotations.csv")
    for csv_path in annotation_paths:
        data = read_annotations(csv_path)
        comparison_in_frame = []

        # Data dictionaries from annotation
        color_count = data["color_count"]
        shape_count = data["shape_count"]
        color_shape_count = data["color_shape_count"]
        color_frames = data["color_frames"]
        shape_frames = data["shape_frames"]
        color_shape_frames = data["color_shape_frames"]

        # We'll try up to 1000 attempts to find 15 valid in-frame comparisons
        attempts = 0
        used_questions = set()

        # [ADDED] ------------------------------------------------------------------------------------
        # Try multiple times to generate a question from duplicates in color_shape_count
        duplicates = [(cs, cnt) for cs, cnt in color_shape_count.items() if cnt >= 2]
        max_duplicate_attempts = 1000
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

            frames1 = get_frames_by_type("color_shape", cs_val_1, color_frames, shape_frames, color_shape_frames)
            frames2 = get_frames_by_type("color_shape", cs_val_2, color_frames, shape_frames, color_shape_frames)
            if frames1 & frames2:  # in-frame overlap
                answer = comparison_answer(count_1, count_2, label1, label2)
                comparison_in_frame.append((question, answer))
                used_questions.add(question)
                found_duplicate_question = True
                break
        # [ADDED] ------------------------------------------------------------------------------------


        while len(comparison_in_frame) < NUM_COMPARISON_QUESTIONS and attempts < 1000:
            attempts += 1

            # 1) Pick two item types randomly
            type1 = random.choice(["color", "shape", "color_shape"])
            type2 = random.choice(["color", "shape", "color_shape"])

            # 2) Pick actual items
            item_type_1, item_val_1 = pick_item_by_type(type1, color_count, shape_count, color_shape_count)
            item_type_2, item_val_2 = pick_item_by_type(type2, color_count, shape_count, color_shape_count)

            # 3) Avoid duplicates if both are same type+value
            tries = 0
            while (item_type_1 == item_type_2 and item_val_1 == item_val_2) and tries < 10:
                item_type_2, item_val_2 = pick_item_by_type(type2, color_count, shape_count, color_shape_count)
                tries += 1

            # 4) Compute counts
            count1 = get_count_by_type(item_type_1, item_val_1, color_count, shape_count, color_shape_count)
            count2 = get_count_by_type(item_type_2, item_val_2, color_count, shape_count, color_shape_count)

            # 5) Build question and answer
            label1 = label_item(item_type_1, item_val_1)
            label2 = label_item(item_type_2, item_val_2)
            question = f"Are there more {label1} or {label2}?"
            if question in used_questions:
                continue  # Skip this one, already used
            answer = comparison_answer(count1, count2, label1, label2)

            # 6) Check in-frame overlap
            frames1 = get_frames_by_type(item_type_1, item_val_1, color_frames, shape_frames, color_shape_frames)
            frames2 = get_frames_by_type(item_type_2, item_val_2, color_frames, shape_frames, color_shape_frames)

            # 7) If there's overlap, add to list
            if frames1 & frames2:
                comparison_in_frame.append((question, answer))
                used_questions.add(question)

        print("[INFO] Generating in-frame comparison questions.")
        out_csv = os.path.join(
            os.path.dirname(csv_path), "comparison_questions_in_frame.csv"
        )
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            for q, a in comparison_in_frame:
                writer.writerow([q, a])
        print(f"Wrote {len(comparison_in_frame)} in-frame comparisons to {out_csv}")


if __name__ == "__main__":
    main()

