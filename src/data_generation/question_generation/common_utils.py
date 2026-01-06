import csv
import random
from collections import defaultdict
import os

# Common utilities extracted from the old scripts:

COLORS = ["red", "green", "blue", "yellow", "black", "purple", "brown", "orange"]
SHAPES = ["sphere", "cuboid", "cone"]


def parse_object_name(obj_name: str):
    """
    Attempt to parse something like "medium_green_sphere_12" into: (color, shape).
    We'll handle two typical patterns:
      1) [size, color, shape, index]
      2) [color, shape, index]
    """
    parts = obj_name.split("_")
    if len(parts) >= 4:
        # e.g. "medium_green_sphere_12" => color=parts[1], shape=parts[2]
        color = parts[1]
        shape = parts[2]
        return color, shape
    elif len(parts) == 3:
        # e.g. "green_sphere_12" => color=parts[0], shape=parts[1]
        color = parts[0]
        shape = parts[1]
        return color, shape
    else:
        return None, None


def parse_visible_objects(visible_str):
    """
    Convert a CSV cell containing something like:
      "green_cuboid_3 (0.20%), black_sphere_18 (0.58%), GroundPlane (44.59%)"
    into a list of object names (e.g. ["green_cuboid_3", "black_sphere_18"]),
    ignoring GroundPlane. Assumes objects appear in left-to-right order.
    """
    parts = [p.strip() for p in visible_str.split(",")]
    objects = []
    for p in parts:
        if "(" in p:
            obj_name = p.split("(")[0].strip()
            if obj_name != "GroundPlane":
                objects.append(obj_name)
        else:
            if p and p != "GroundPlane":
                objects.append(p)
    return objects


def pick_unseen_color(color_count):
    """
    Return (unseen_color, 0) if available, else (None, None).
    'unseen_color' means a color in COLORS that does not appear in color_count.
    """
    unseen_candidates = [c for c in COLORS if c not in color_count]
    if unseen_candidates:
        c = random.choice(unseen_candidates)
        return (c, 0)
    return (None, None)


def pick_unseen_shape(shape_count):
    """
    Return (unseen_shape, 0) if available, else (None, None).
    """
    unseen_candidates = [s for s in SHAPES if s not in shape_count]
    if unseen_candidates:
        s = random.choice(unseen_candidates)
        return (s, 0)
    return (None, None)


def pick_unseen_color_shape(color_shape_count):
    """
    Return ((unseen_color, unseen_shape), 0) if available, else (None, None).
    """
    unseen_combos = []
    for c in COLORS:
        for s in SHAPES:
            if (c, s) not in color_shape_count:
                unseen_combos.append((c, s))
    if unseen_combos:
        return (random.choice(unseen_combos), 0)
    return (None, None)


def comparison_answer(count_x, count_y, label_x, label_y):
    """
    Compare count_x and count_y, return "X", "Y", or "equal".
    """
    if count_x > count_y:
        return label_x
    elif count_y > count_x:
        return label_y
    else:
        return "equal"


def build_left_right_relations(list_of_frames):
    """
    Given a list of frames, each frame being a list of objects in left-to-right order,
    determine which object pairs (A, B) have a consistent left-right relationship
    across ALL frames where both appear.

    Returns a list of (A, B) meaning "A is always left of B" (the ground truth).
    """
    pair_relationships = defaultdict(set)

    # Record "A<B" for all pairs (A, B) within each frame
    for frame_objects in list_of_frames:
        for i in range(len(frame_objects)):
            for j in range(i + 1, len(frame_objects)):
                A = frame_objects[i]
                B = frame_objects[j]
                if A == B:
                    continue

                # A is left of B in this frame
                pair_relationships[(A, B)].add("A<B")
                pair_relationships[(B, A)].add("B>A")

    # Determine which are consistent across all frames
    consistent_pairs = []

    # Gather all objects
    all_objects = set()
    for A, B in pair_relationships.keys():
        all_objects.add(A)
        all_objects.add(B)
    all_objects = sorted(all_objects)

    # For each distinct unordered pair
    for i in range(len(all_objects)):
        for j in range(i + 1, len(all_objects)):
            A = all_objects[i]
            B = all_objects[j]
            if A == B:
                continue

            rel_AB = pair_relationships.get((A, B), set())
            rel_BA = pair_relationships.get((B, A), set())

            # If they never co-occur, skip
            if not rel_AB and not rel_BA:
                continue

            # A always left of B
            if rel_AB == {"A<B"} and rel_BA == {"B>A"}:
                # That means the consistent ordering is A < B
                consistent_pairs.append((A, B))
            # B always left of A
            elif rel_AB == {"B<A"} and rel_BA == {"A>B"}:
                # That means the consistent ordering is B < A
                consistent_pairs.append((B, A))
            # Otherwise no single consistent ordering

    return consistent_pairs


def shorten_name(object_name):
    """
    Takes a string like "green_cuboid_3" and removes the trailing integer,
    returning "green cuboid". We assume each object_name is color_shape_int.
    """
    parts = object_name.split("_")
    # If the last part is all digits, remove it
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    # Join the remaining parts with a space (e.g. "green cuboid")
    return " ".join(parts)


def read_annotations(csv_path):
    """
    Reads the annotations CSV and returns a dictionary with counts and frames
    for colors, shapes, and color_shape pairs, plus seen objects.
    This version counts duplicate objects (like 'red_sphere_1' and
    'red_sphere_2') as separate objects.
    """
    from collections import defaultdict

    color_count = defaultdict(int)
    shape_count = defaultdict(int)
    color_shape_count = defaultdict(int)

    color_frames = defaultdict(set)
    shape_frames = defaultdict(set)
    color_shape_frames = defaultdict(set)

    # For frames, we'll still use the shortened names:
    object_frames = defaultdict(set)

    # Keep a separate set for counting full object names (with indices)
    seen_objects_full = set()

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for frame_index, row in enumerate(reader):
            if "visible_objects" not in row:
                # In case the CSV doesn't match our expectations
                continue

            objects = parse_visible_objects(row["visible_objects"])

            for obj_name in objects:
                # For frame grouping, use short_name
                short_name = shorten_name(obj_name)
                object_frames[short_name].add(frame_index)

                color, shape = parse_object_name(obj_name)
                if color and shape:
                    color_frames[color].add(frame_index)
                    shape_frames[shape].add(frame_index)
                    color_shape_frames[(color, shape)].add(frame_index)

                # Now only increment the counts if we haven't seen this exact object before
                if obj_name not in seen_objects_full:
                    seen_objects_full.add(obj_name)

                    if color and shape:
                        color_count[color] += 1
                        shape_count[shape] += 1
                        color_shape_count[(color, shape)] += 1

    data = {
        "color_count": dict(color_count),
        "shape_count": dict(shape_count),
        "color_shape_count": dict(color_shape_count),
        "color_frames": dict(color_frames),
        "shape_frames": dict(shape_frames),
        "color_shape_frames": dict(color_shape_frames),
        "object_frames": dict(object_frames),
        # Store the full set of seen objects (including trailing index)
        "seen_objects": sorted(seen_objects_full),
    }

    # Write a subset of the data dictionary to CSV in the same directory
    data_dict_csv = os.path.join(os.path.dirname(csv_path), "data_dict.csv")
    with open(data_dict_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "key", "value"])
        for c, count in data["color_count"].items():
            writer.writerow(["color_count", c, count])
        for s, count in data["shape_count"].items():
            writer.writerow(["shape_count", s, count])
        for (c, s), count in data["color_shape_count"].items():
            writer.writerow(["color_shape_count", f"{c}_{s}", count])

        # NEW: write color_shape_frames
        for (c, s), frames_set in data["color_shape_frames"].items():
            # Convert the set of frames to a sorted, comma-separated string
            frames_str = ",".join(str(frame) for frame in sorted(frames_set))
            writer.writerow(["color_shape_frames", f"{c}_{s}", frames_str])
            
    print(f"[INFO] Wrote data dictionary to {data_dict_csv}")

    return data


def pick_seen_color(color_count):
    """
    Picks a random color from color_count if available, else returns "red".
    """
    if color_count:
        c, _ = random.choice(list(color_count.items()))
        return c
    return "red"


def pick_seen_shape(shape_count):
    """
    Picks a random shape from shape_count if available, else returns "sphere".
    """
    if shape_count:
        s, _ = random.choice(list(shape_count.items()))
        return s
    return "sphere"


def pick_seen_color_shape(color_shape_count):
    """
    Picks a random (color, shape) pair from color_shape_count if available,
    else ("red", "sphere").
    """
    if color_shape_count:
        (c, s), _ = random.choice(list(color_shape_count.items()))
        return (c, s)
    return ("red", "sphere")


def pluralize_shape(shape_name: str) -> str:
    if shape_name == "sphere":
        return "spheres"
    elif shape_name == "cube":
        return "cubes"
    elif shape_name == "cuboid":
        return "cuboids"
    elif shape_name == "cylinder":
        return "cylinders"
    elif shape_name == "cone":
        return "cones"
    return shape_name + "s"


def get_color_count_val(data, color):
    return data["color_count"].get(color, 0)


def get_shape_count_val(data, shape):
    return data["shape_count"].get(shape, 0)


def get_color_shape_count_val(data, color, shape):
    return data["color_shape_count"].get((color, shape), 0)


def get_color_frames(data, color):
    return data["color_frames"].get(color, set())


def get_shape_frames(data, shape):
    return data["shape_frames"].get(shape, set())


def get_color_shape_frames(data, color, shape):
    return data["color_shape_frames"].get((color, shape), set())

