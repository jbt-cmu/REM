import os
import glob
import csv
import random
from src.data_generation.question_generation.common_utils import (
    read_annotations,
    parse_object_name,
    shorten_name,
    COLORS,
    SHAPES,
)

NUM_ORDER_QUESTIONS = 5
MAX_ATTEMPTS = 1000

def main():
    annotation_paths = glob.glob("src/trajectories/*-run/images/annotations.csv")
    for csv_path in annotation_paths:
        data = read_annotations(csv_path)
        object_frames = data["object_frames"]
        seen_objects_full = sorted(data["seen_objects"])
        seen_objects = []
        for object in seen_objects_full:
            seen_objects.append(shorten_name(object))
        

        # earliest_frame[obj] = the earliest frame the object appears
        earliest_frame = {}
        for obj in seen_objects:
            frames_set = object_frames.get(obj, set())
            if frames_set:
                earliest_frame[obj] = min(frames_set)
            

        # We want to gather up to NUM_ORDER_QUESTIONS
        order_preserving_questions = []
        used_questions = set()  # to avoid exact-duplicate text

        if len(seen_objects) < 2:
            # If there's only 0 or 1 object, no possible comparisons
            pass
        else:
            # 1) First, try to collect 'before' or 'after' questions
            attempts = 0
            while len(order_preserving_questions) < NUM_ORDER_QUESTIONS and attempts < MAX_ATTEMPTS:
                attempts += 1
                # Pick two distinct objects at random
                x, y = random.sample(seen_objects, 2)

                question_text = f"Did we see the {x} before, after, or at the same time as the {y}?"
                # Avoid duplicates
                if question_text in used_questions:
                    continue

                fx = earliest_frame.get(x, None)
                fy = earliest_frame.get(y, None)
                if fx is None or fy is None:
                    continue

                # We ONLY want "before"/"after"
                if fx < fy:
                    answer = "before"
                elif fx > fy:
                    answer = "after"
                else:
                    # same time => skip here, we will handle it later
                    continue

                # Record the question
                order_preserving_questions.append((question_text, answer))
                used_questions.add(question_text)

            # 2) If we still don't have enough questions, fall back to 'same time'
            if len(order_preserving_questions) < NUM_ORDER_QUESTIONS:
                attempts2 = 0
                while len(order_preserving_questions) < NUM_ORDER_QUESTIONS and attempts2 < (MAX_ATTEMPTS * 2):
                    attempts2 += 1
                    x, y = random.sample(seen_objects, 2)

                    question_text = f"Did we see the {x} before, after, or same time as the {y}?"
                    if question_text in used_questions:
                        continue

                    fx = earliest_frame.get(x, None)
                    fy = earliest_frame.get(y, None)
                    if fx is None or fy is None:
                        continue

                    # Now we specifically look for same-time
                    if fx == fy:
                        answer = "same time"
                        order_preserving_questions.append((question_text, answer))
                        used_questions.add(question_text)

        # Write results
        out_csv = os.path.join(os.path.dirname(csv_path), "order_preserving_questions.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            for q, a in order_preserving_questions:
                writer.writerow([q, a])

        print(
            f"Wrote {len(order_preserving_questions)} order-preserving questions to {out_csv}"
        )


if __name__ == "__main__":
    main()

