import csv
import random
from collections import defaultdict
from src.data_generation.question_generation.common_utils import (
    parse_visible_objects,
    build_left_right_relations,
    shorten_name,
    # ...any other needed utils...
)

MAX_NUM_QUESTIONS_TO_WRITE = 5
def extract_left_right_questions_from_file(annotations_csv_path):
    """
    Reads the given annotations.csv, finds consistent left-right pairs,
    and returns a list of (question, answer) tuples.

    For each pair (A, B) meaning A is always left of B, we produce TWO lines:
      1) "Is A to the left of B?" => "yes"
      2) "Is B to the left of A?" => "no"
    This ensures one question has answer "yes" and the other has answer "no".
    """
    frames = []
    # Read the CSV
    with open(annotations_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            visible_str = row["visible_objects"]
            frame_objs = parse_visible_objects(visible_str)
            frames.append(frame_objs)

    # Determine the consistent pairs (A, B) => A < B
    consistent_pairs = build_left_right_relations(frames)

    # Generate questions using ground truth
    q_and_a = []
    for A, B in consistent_pairs:
        if A == B:
            continue

        shortA = shorten_name(A)
        shortB = shorten_name(B)

        sublist = []

        # "Is A to the left of B?" => yes
        q_left = f"Is the {shortA} to the left of the {shortB}?"
        a_left = "yes"
        sublist.append((q_left, a_left))

        q_right = f"Is the {shortA} to the right of the {shortB}?"
        a_right = "no"
        sublist.append((q_right, a_right))

        # "Is B to the left of A?" => no
        q_left_reverse = f"Is the {shortB} to the left of the {shortA}?"
        a_left_reverse = "no"
        sublist.append((q_left_reverse, a_left_reverse))

        q_right_reverse = f"Is the {shortB} to the right of the {shortA}?"
        a_right_reverse = "yes"
        sublist.append((q_right_reverse, a_right_reverse))

        random.shuffle(sublist)
        question = sublist[0]
        q_and_a.append(question)

    return q_and_a


def main():
    """
    1. From the src directory, find all ../trajectories/*-run/images/annotations.csv files.
    2. For each, generate left-right questions with ground truth (mix of yes/no).
    3. Write them to left_right_questions.csv in THAT SAME directory.
    """
    import glob
    import os

    # 1. Locate all annotation files
    all_annotations = glob.glob("src/trajectories/*-run/images/annotations.csv")

    # 2. Process each annotations.csv individually
    for anno_file in all_annotations:
        q_and_a = extract_left_right_questions_from_file(anno_file)[:MAX_NUM_QUESTIONS_TO_WRITE]

        # We'll write the output CSV in the same folder as anno_file
        anno_dir = os.path.dirname(anno_file)
        output_csv_path = os.path.join(anno_dir, "left_right_questions.csv")

        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])  # optional header
            for question, answer in q_and_a:
                writer.writerow([question, answer])

        print(f"Wrote {len(q_and_a)} Q&A entries to {output_csv_path}")

    print("[INFO] Generating left-right directional questions.")


if __name__ == "__main__":
    main()

