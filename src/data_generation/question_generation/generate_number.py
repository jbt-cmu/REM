#!/usr/bin/env python3
import csv
import glob
import os
import random

from src.data_generation.question_generation.common_utils import COLORS, SHAPES, read_annotations

# --------------------------------------------
# Configurable probability & maximum to write
# --------------------------------------------
P_ADD_UNSEEN = 0.1           # Probability of actually adding unseen questions
MAX_QUESTIONS_TO_WRITE = 5  # How many total questions to write to the CSV

# A map of singular -> plural shapes
SHAPE_PLURAL_MAP = {
    "cone": "cones",
    "sphere": "spheres",
    "cuboid": "cuboids",
}

def pluralize_shape(s: str) -> str:
    """
    Return the plural form of shape `s` using SHAPE_PLURAL_MAP if present,
    else default to appending 's'.
    """
    return SHAPE_PLURAL_MAP.get(s, f"{s}s")

def main():
    annotation_paths = glob.glob("src/trajectories/*-run/images/annotations.csv")
    for csv_path in annotation_paths:
        data = read_annotations(csv_path)
        color_count_list = list(data["color_count"].items())
        shape_count_list = list(data["shape_count"].items())
        color_shape_count_list = list(data["color_shape_count"].items())

        # -----------------------------------------------------
        # Gather duplicates, singletons, and unseen for colors
        # -----------------------------------------------------
        duplicates_color = [(c, n) for (c, n) in color_count_list if n > 1]
        singletons_color = [(c, n) for (c, n) in color_count_list if n == 1]
        unseen_colors = [(c, 0) for c in COLORS if c not in data["color_count"]]

        # -----------------------------------------------------
        # Gather duplicates, singletons, and unseen for shapes
        # -----------------------------------------------------
        duplicates_shape = [(s, n) for (s, n) in shape_count_list if n > 1]
        singletons_shape = [(s, n) for (s, n) in shape_count_list if n == 1]
        unseen_shapes = [(s, 0) for s in SHAPES if s not in data["shape_count"]]

        # ----------------------------------------------------------------
        # Gather duplicates, singletons, and unseen for color-shape pairs
        # ----------------------------------------------------------------
        duplicates_c_s = [
            ((c, s), n) for ((c, s), n) in color_shape_count_list if n > 1
        ]
        singletons_c_s = [
            ((c, s), n) for ((c, s), n) in color_shape_count_list if n == 1
        ]
        unseen_c_s = [
            ((c, s), 0)
            for c in COLORS
            for s in SHAPES
            if (c, s) not in data["color_shape_count"]
        ]

        # We'll accumulate all questions here (question, answer)
        exhaustive_number_questions = []
        duplicates_to_add = []

        # ----------------------------------------------------------------
        # COLOR QUESTIONS (leave as "How many red objects...")
        # ----------------------------------------------------------------
        for c, count_gt in duplicates_color:
            question = f"How many {c} objects are there?"
            answer = str(count_gt)
            exhaustive_number_questions.append((question, answer))

        for c, count_gt in singletons_color:
            question = f"How many {c} objects are there?"
            answer = str(count_gt)
            exhaustive_number_questions.append((question, answer))

        # Only add unseen color questions with probability P_ADD_UNSEEN
        for c, count_gt in unseen_colors:
            if random.random() < P_ADD_UNSEEN:
                question = f"How many {c} objects are there?"
                answer = str(count_gt)
                exhaustive_number_questions.append((question, answer))

        # ----------------------------------------------------------------
        # SHAPE QUESTIONS (use plural)
        # ----------------------------------------------------------------
        for s, count_gt in duplicates_shape:
            s_plural = pluralize_shape(s)
            question = f"How many {s_plural} are there?"
            answer = str(count_gt)
            exhaustive_number_questions.append((question, answer))

        for s, count_gt in singletons_shape:
            s_plural = pluralize_shape(s)
            question = f"How many {s_plural} are there?"
            answer = str(count_gt)
            exhaustive_number_questions.append((question, answer))

        for s, count_gt in unseen_shapes:
            if random.random() < P_ADD_UNSEEN:
                s_plural = pluralize_shape(s)
                question = f"How many {s_plural} are there?"
                answer = str(count_gt)
                exhaustive_number_questions.append((question, answer))

        # ----------------------------------------------------------------
        # COLOR-SHAPE QUESTIONS (use shape in plural form)
        # ----------------------------------------------------------------
        for (c, s), count_gt in duplicates_c_s:
            s_plural = pluralize_shape(s)
            question = f"How many {c} {s_plural} are there?"
            answer = str(count_gt)
            exhaustive_number_questions.append((question, answer))

        for (c, s), count_gt in singletons_c_s:
            s_plural = pluralize_shape(s)
            question = f"How many {c} {s_plural} are there?"
            answer = str(count_gt)
            exhaustive_number_questions.append((question, answer))

        for (c, s), count_gt in unseen_c_s:
            if random.random() < P_ADD_UNSEEN:
                s_plural = pluralize_shape(s)
                question = f"How many {c} {s_plural} are there?"
                answer = str(count_gt)
                exhaustive_number_questions.append((question, answer))

        # (Optional) Shuffle the full list of questions before taking the slice.
        random.shuffle(exhaustive_number_questions)

        # Now we only write up to MAX_QUESTIONS_TO_WRITE
        final_questions = exhaustive_number_questions[:MAX_QUESTIONS_TO_WRITE]
        final_questions.extend(duplicates_to_add)

        output_csv = os.path.join(os.path.dirname(csv_path), "number_questions.csv")
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            for question, answer in final_questions:
                writer.writerow([question, answer])

        print("[INFO] Generated exhaustive number questions.")
        print(f"Total generated (before truncation): {len(exhaustive_number_questions)}")
        print(f"Wrote {len(final_questions)} number questions to {output_csv}")


if __name__ == "__main__":
    main()

