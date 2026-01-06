#!/usr/bin/env python

import argparse
import json
import math
import os
import random
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

# Constants
GRID_SIZE = 12  # Half-size of the grid (total size 20m x 20m) for object placement
BLOCK_RADIUS = 1.0  # Radius around trajectory points to block object placement
ROD_HEIGHT = 1.5  # Height of the rod

# Configuration Data
@dataclass
class Shape:
    name: str
    type: str
    config: Dict[str, Any]

@dataclass
class Color:
    name: str
    diffuse_color: Tuple[float, float, float]

@dataclass
class Size:
    name: str
    scale: float

@dataclass
class Texture:
    name: str
    roughness: float
    metallic: float

SHAPES = [
    Shape(name="cuboid", type="cuboid", config={"size": [1, 1, 1]}),
    Shape(name="cone", type="cone", config={"radius": 0.5, "height": 1}),
    Shape(name="sphere", type="sphere", config={"radius": 0.5}),
    # Shape(name="cylinder", type="cylinder", config={"radius": 0.5, "height": 1}),
]

COLORS = [
    Color(name="red", diffuse_color=(0.9, 0.08, 0.08)),
    Color(name="green", diffuse_color=(0.114, 0.412, 0.078)),
    Color(name="blue", diffuse_color=(0.165, 0.294, 0.843)),
    Color(name="yellow", diffuse_color=(1.000, 0.933, 0.200)),
    Color(name="purple", diffuse_color=(0.506, 0.149, 0.753)),
    Color(name="brown", diffuse_color=(0.213, 0.122, 0.041)),
    Color(name="black", diffuse_color=(0.005, 0.005, 0.005)),
    Color(name="orange", diffuse_color=(0.75, 0.2, 0.05)),
]

SIZES = [
    Size(name="small", scale=0.5),
    Size(name="medium", scale=0.75),
    Size(name="large", scale=1.0),
]

TEXTURES = [
    Texture(name="rubber", roughness=0.8, metallic=0.0),
    Texture(name="metallic", roughness=0.2, metallic=1.0),
]

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Scene Generation Script")

    parser.add_argument(
        "--num-shapes",
        type=int,
        default=24,
        help="Total number of shapes to generate in the scene (minimum 8).",
    )

    parser.add_argument(
        "--times",
        type=int,
        nargs="*",
        default=[],
        help="Number of duplicates for each selected object. Each value t means the object appears t+1 times.",
    )

    parser.add_argument(
        "--camera-height",
        type=float,
        default=None,
        help="Specify the camera height in meters.",
    )
    parser.add_argument(
        "--trajectory-size",
        type=int,
        default=8,
        help="Number of image frames in the trajectory.",
    )
    parser.add_argument(
        "--turn-prob",
        type=float,
        default=0.3,
        help="(Unused in new trajectory generation.)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--same-size",
        action="store_true",
        help="If set, all generated shapes are only large (1m).",
    )
    parser.add_argument(
        "--homogenous-texture",
        action="store_true",
        help="If set, all objects have rubber texture and texture is omitted from names.",
    )
    # New argument to define the bounding box half-size for the trajectory
    parser.add_argument(
        "--bbox",
        type=float,
        default=6,
        help="Half-size of the x,y bounding box for trajectory generation.",
    )
    return parser.parse_args()

def setup_run_directory() -> str:
    """Sets up the run directory based on the run name."""
    run_name = "run"
    trajectories_dir = os.path.join(os.path.dirname(__file__), "..", "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)
    existing_runs = [
        d for d in os.listdir(trajectories_dir)
        if os.path.isdir(os.path.join(trajectories_dir, d))
    ]
    run_numbers = [
        int(d.split("-")[0]) for d in existing_runs if d.split("-")[0].isdigit()
    ]
    next_run_number = max(run_numbers, default=0) + 1
    run_folder_name = f"{next_run_number:04d}-{run_name}"
    run_folder_path = os.path.join(trajectories_dir, run_folder_name)
    os.makedirs(run_folder_path, exist_ok=True)
    return run_folder_path

def set_random_seed(seed: Optional[int]) -> None:
    """Sets the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

def generate_trajectory(
    trajectory_size: int,
    grid_size: float = 8,
) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    Generates a trajectory for the agent.
    The agent moves forward in 1m increments.
    Upon reaching the edge of the scene (defined by |x| <= grid_size and |y| <= grid_size),
    the agent performs a turn. The total turn angle is randomly chosen between 120 and 240 degrees
    (in increments of 15 degrees) and is decomposed into multiple "turn left/right 15 degrees" moves.
    The trajectory_size defines the number of image frames in the trajectory.
    """
    x = -3.0
    y = 0.0
    orientation = 0.0  # Start facing east (0 degrees)

    moves = []
    positions = [(x, y)]  # Initial position

    forward_moves = 0
    # Allowed turn angles from 120 to 240 degrees in 15 degree increments.
    possible_turn_angles = list(range(120, 241, 15))

    while forward_moves < trajectory_size:
        # Compute next position if moving straight 1m in current orientation
        rad = math.radians(orientation)
        next_x = x + math.cos(rad)
        next_y = y + math.sin(rad)

        # Check if next position is within the bounding box
        if abs(next_x) <= grid_size and abs(next_y) <= grid_size:
            moves.append("straight")
            x, y = next_x, next_y
            positions.append((x, y))
            forward_moves += 1
        else:
            # At edge: perform a random turn in increments of 15 degrees.
            angle = random.choice(possible_turn_angles)
            turn_direction = random.choice(["left", "right"])
            increments = angle // 15  # number of 15-degree increments
            for _ in range(increments):
                moves.append(f"turn {turn_direction} 15 degrees")
                if turn_direction == "left":
                    orientation = (orientation + 15) % 360.0
                else:
                    orientation = (orientation - 15) % 360.0
    return moves, positions

def is_blocked(x: int, y: int, blocked_areas: Optional[List[Dict[str, float]]]) -> bool:
    """
    Checks if a grid point is within any blocked area.
    """
    if not blocked_areas:
        return False
    for area in blocked_areas:
        dx = x - area["x"]
        dy = y - area["y"]
        distance = math.hypot(dx, dy)
        if distance < area["radius"]:
            return True
    return False

def generate_grid_points(
    grid_size: int, blocked_areas: Optional[List[Dict[str, float]]]
) -> List[Tuple[int, int]]:
    """
    Generates a list of valid grid points for object placement.
    """
    grid_points = []
    for x in range(-grid_size, grid_size + 1):
        for y in range(-grid_size, grid_size + 1):
            if (x, y) != (0, 0) and not is_blocked(x, y, blocked_areas):
                grid_points.append((x, y))
    return grid_points

def define_blocked_areas(
    positions: List[Tuple[float, float]], radius: float
) -> List[Dict[str, float]]:
    """
    Defines blocked areas around given positions.
    """
    return [{"x": pos[0], "y": pos[1], "radius": radius} for pos in positions]

def generate_scene_config(
    num_shapes: int,
    times: List[int],
    camera_height: Optional[float],
    save_config_path: str,
    blocked_areas: Optional[List[Dict[str, float]]] = None,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    same_size: bool = False,
    homogenous_texture: bool = False,
) -> None:
    """
    Generates the scene configuration and saves it to a JSON file.
    'times' defines the duplicates: each value t means one duplicated object that appears t+1 times.
    The total must not exceed num_shapes, and we fill the remainder with unique objects.
    """
    max_objects = 200  # Adjust as needed
    total_combinations = len(SHAPES) * len(COLORS) * len(SIZES) * len(TEXTURES)
    if num_shapes > max_objects:
        raise ValueError(
            f"Total number of objects ({num_shapes}) exceeds maximum allowed ({max_objects})"
        )
    sum_instances = sum((t + 1) for t in times)
    if sum_instances > num_shapes:
        raise ValueError(
            f"Requested duplicates produce {sum_instances} objects, but only {num_shapes} allowed."
        )
    shapes = SHAPES.copy()
    sizes = [size for size in SIZES if size.name == "large"] if same_size else SIZES
    textures = [TEXTURES[0]] if homogenous_texture else TEXTURES
    all_combinations = list(itertools.product(shapes, COLORS, sizes, textures))
    unique_objects = []
    for shape, color, size, texture in all_combinations:
        if same_size:
            obj_name = f"{color.name}_{shape.name}" if homogenous_texture else f"{color.name}_{texture.name}_{shape.name}"
        else:
            obj_name = f"{size.name}_{color.name}_{shape.name}" if homogenous_texture else f"{size.name}_{color.name}_{texture.name}_{shape.name}"
        shape_config = dict(shape.config)
        scale = size.scale
        if shape.type in ["cuboid", "rod"]:
            shape_config["size"] = [s * scale for s in shape_config["size"]]
        elif shape.type == "sphere":
            shape_config["radius"] *= scale
        elif shape.type in ["cone", "cylinder"]:
            shape_config["radius"] *= scale
            shape_config["height"] *= scale
        obj_attributes = {
            "shape": shape.name,
            "color": color.name,
            "size": size.name,
            "texture": texture.name,
            "diffuse_color": color.diffuse_color,
            "roughness": texture.roughness,
            "metallic": texture.metallic,
        }
        obj = {
            "name": obj_name,
            "type": shape.type,
            "config": shape_config,
            "attributes": obj_attributes,
        }
        unique_objects.append(obj)
    duplicated_objects = []
    num_object_duplicates = len(times)
    if num_object_duplicates > 0:
        chosen_for_duplication = random.sample(unique_objects, num_object_duplicates)
        for ch in chosen_for_duplication:
            unique_objects.remove(ch)
        for i, base_obj in enumerate(chosen_for_duplication):
            total_count = times[i] + 1
            instances = [dict(base_obj) for _ in range(total_count)]
            duplicated_objects.extend(instances)
    else:
        duplicated_objects = []
        chosen_for_duplication = []
    used_from_duplicates = sum_instances
    remainder = num_shapes - used_from_duplicates
    if remainder > len(unique_objects):
        raise ValueError(
            f"Not enough unique objects left ({len(unique_objects)}) to fill the remainder ({remainder})."
        )
    remainder_unique = random.sample(unique_objects, remainder)
    objects_list = duplicated_objects + remainder_unique
    random.shuffle(objects_list)
    scene_objects = _generate_random_configuration(objects_list, blocked_areas)
    for idx, obj in enumerate(scene_objects):
        obj["name"] = f"{obj['name']}_{idx}"
    duplication_info = []
    if num_object_duplicates > 0:
        for i, base_obj in enumerate(chosen_for_duplication):
            info = {
                "base_object": base_obj["name"],
                "duplicate_count": times[i],
                "total_count": times[i] + 1,
            }
            duplication_info.append(info)
    _save_configuration(
        scene_objects,
        save_config_path,
        camera_height=camera_height,
        trajectory=trajectory,
        duplication_info=duplication_info,
        times=times,
    )

def _generate_random_configuration(
    objects_list: List[Dict],
    blocked_areas: Optional[List[Dict[str, float]]] = None,
) -> List[Dict]:
    """
    Generates random positions for objects, avoiding overlaps and blocked areas.
    Instead of filling from the center outward aggressively, grid points are divided into quadrants
    and then interleaved, so each part of the scene is used more equally.
    The minimum distance between objects is increased by 50%.
    """
    grid_points = generate_grid_points(GRID_SIZE, blocked_areas)
    if len(grid_points) < len(objects_list):
        raise ValueError(
            f"Not enough grid points ({len(grid_points)}) to place all objects ({len(objects_list)})."
        )
    # Group grid points by quadrant.
    q1, q2, q3, q4 = [], [], [], []
    for p in grid_points:
        x, y = p
        if x >= 0 and y >= 0:
            q1.append(p)
        elif x < 0 and y >= 0:
            q2.append(p)
        elif x < 0 and y < 0:
            q3.append(p)
        elif x >= 0 and y < 0:
            q4.append(p)
    # Within each quadrant, sort by distance from the center.
    for q in (q1, q2, q3, q4):
        q.sort(key=lambda p: math.hypot(p[0], p[1]))
    # Interleave points from each quadrant.
    ordered_grid_points = []
    quadrant_lists = [q1, q2, q3, q4]
    still_has_points = True
    while still_has_points:
        still_has_points = False
        for q in quadrant_lists:
            if q:
                ordered_grid_points.append(q.pop(0))
                still_has_points = True

    placed_objects = []
    scene_objects = []
    for obj in objects_list:
        obj_type = obj["type"]
        obj_config = obj["config"]
        if obj_type in ["cuboid", "rod"]:
            collision_radius = max(obj_config["size"]) / 2
        elif obj_type == "sphere":
            collision_radius = obj_config["radius"]
        elif obj_type in ["cone", "cylinder"]:
            collision_radius = obj_config["radius"]
        else:
            collision_radius = 0.5
        placed = False
        while ordered_grid_points and not placed:
            x, y = ordered_grid_points.pop(0)
            overlap = False
            for placed_obj in placed_objects:
                dx = x - placed_obj["position"][0]
                dy = y - placed_obj["position"][1]
                # Increase the minimum distance by 50%
                min_distance = 1.5 * (collision_radius + placed_obj["collision_radius"] + 0.01)
                if math.hypot(dx, dy) < min_distance:
                    overlap = True
                    break
            if not overlap:
                placed = True
                z_pos = round(_get_object_z_position(obj_type, obj_config), 2)
                scene_object = {
                    "name": obj["name"],
                    "type": obj_type,
                    "position": [x, y, z_pos],
                    "config": obj_config,
                }
                if obj.get("attributes"):
                    scene_object["attributes"] = obj["attributes"]
                scene_objects.append(scene_object)
                placed_objects.append({
                    "name": obj["name"],
                    "position": (x, y),
                    "collision_radius": round(collision_radius, 2),
                })
        if not placed:
            raise ValueError("Could not place all objects without overlap.")
    return scene_objects

def _get_object_z_position(obj_type: str, obj_config: Dict[str, Any]) -> float:
    if obj_type in ["cuboid", "rod"]:
        return obj_config["size"][2] / 2
    elif obj_type == "sphere":
        return obj_config["radius"]
    elif obj_type in ["cone", "cylinder"]:
        return obj_config["height"] / 2
    else:
        return 0.5

def _save_configuration(
    scene_objects: List[Dict],
    save_config_path: str,
    camera_height: Optional[float] = None,
    trajectory: Optional[List[Tuple[float, float]]] = None,
    duplication_info: Optional[List[Dict[str, Any]]] = None,
    times: List[int] = None,
) -> None:
    """
    Saves the scene configuration to a JSON file and generates a plot.
    """
    args = parse_arguments()
    config = {
        "scene_settings": {
            "camera_height": camera_height,
            "trajectory": trajectory,
        },
        "generation_flags": {
            "num_shapes": args.num_shapes,
            "times": args.times,
            "camera_height": args.camera_height,
            "trajectory_size": args.trajectory_size,
            "seed": args.seed,
            "same_size": args.same_size,
        },
        "objects": scene_objects,
    }
    if duplication_info:
        config["scene_settings"]["duplicated_objects"] = duplication_info
    with open(save_config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {save_config_path}")
    _generate_scene_plot(scene_objects, save_config_path, trajectory=trajectory)

def _generate_scene_plot(
    scene_objects: List[Dict],
    save_config_path: str,
    trajectory: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """
    Generates a plot of the scene.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = ["serif"]
    plt.rcParams["font.serif"] = ["DejaVu Sans"]
    plt.rcParams["font.weight"] = "medium"
    plt.rcParams["font.size"] = 10.0
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    fig, ax = plt.subplots(figsize=(14, 14), dpi=300)
    plt.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)
    ax.set_facecolor("#f0f0f0")
    ax.set_xticks(range(-GRID_SIZE, GRID_SIZE + 1))
    ax.set_yticks(range(-GRID_SIZE, GRID_SIZE + 1))
    ax.grid(which="both", color="white", linewidth=0.8, linestyle="-", alpha=0.9)
    if trajectory:
        traj_x = [pos[0] for pos in trajectory]
        traj_y = [pos[1] for pos in trajectory]
        ax.plot(traj_x, traj_y, marker="o", color="#303030",
                linestyle="--", label="Agent Trajectory", linewidth=2, markersize=6)
        for pos in trajectory:
            circ = Circle((pos[0], pos[1]), radius=BLOCK_RADIUS, color="#808080", alpha=0.15)
            ax.add_patch(circ)
    for obj in scene_objects:
        x, y, _ = obj["position"]
        obj_type = obj["type"]
        attributes = obj.get("attributes", {})
        size_name = attributes.get("size", "medium")
        size_scale_map = {"small": 0.25, "medium": 0.5, "large": 1.0}
        size_scale = size_scale_map.get(size_name, 0.5)
        rgb_color = attributes.get("diffuse_color", (0, 0, 0))
        if obj_type in ["cuboid", "rod"]:
            size = size_scale * 1.0
            rect = Rectangle((x - size/2, y - size/2), size, size, color=rgb_color, alpha=0.8)
            ax.add_patch(rect)
        elif obj_type == "sphere":
            circ = Circle((x, y), radius=size_scale * 0.5, color=rgb_color, alpha=0.8)
            ax.add_patch(circ)
        elif obj_type == "cone":
            radius = size_scale * 0.5
            triangle = Polygon([[x, y - radius], [x - radius, y + radius], [x + radius, y + radius]],
                               color=rgb_color, alpha=0.8)
            ax.add_patch(triangle)
        elif obj_type == "cylinder":
            circ = Circle((x, y), radius=size_scale * 0.5, color=rgb_color, alpha=0.8)
            ax.add_patch(circ)
        else:
            print(f"Unknown object type: {obj_type}")
            continue
        ax.text(x, y + size_scale * 0.6, obj["name"], ha="center", va="bottom",
                fontsize=8, color="black", family="DejaVu Sans", weight="medium", zorder=100)
    ax.set_aspect("equal")
    ax.set_xlim(-GRID_SIZE - 1, GRID_SIZE + 1)
    ax.set_ylim(-GRID_SIZE - 1, GRID_SIZE + 1)
    ax.set_xlabel("X Position (meters)", fontsize=14, weight="medium", labelpad=15, family="DejaVu Sans")
    ax.set_ylabel("Y Position (meters)", fontsize=14, weight="medium", labelpad=15, family="DejaVu Sans")
    plt.title("Scene Configuration with Object Placement and Agent Trajectory",
              fontsize=16, weight="medium", pad=25, family="DejaVu Sans")
    ax.tick_params(labelsize=12, length=6, width=1)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family("DejaVu Sans")
        label.set_weight("medium")
    if trajectory:
        legend = ax.legend(fontsize=12, framealpha=0.95, loc="upper right")
        for text in legend.get_texts():
            text.set_family("DejaVu Sans")
            text.set_weight("medium")
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color("#404040")
    dir_path = os.path.dirname(save_config_path)
    plot_file_path = os.path.join(dir_path, "scene_plot.png")
    plt.savefig(plot_file_path, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Scene plot saved to {plot_file_path}")

def save_trajectory_moves(moves: List[str], config_path: str) -> None:
    """
    Saves the trajectory moves to a text file in the same directory as the config file.
    Each turn is recorded as individual 15° increments.
    """
    dir_path = os.path.dirname(config_path)
    moves_path = os.path.join(dir_path, "trajectory_moves.txt")
    with open(moves_path, "w") as f:
        for move in moves:
            f.write(f"{move}\n")
    print(f"Trajectory moves saved to {moves_path}")

def main():
    args = parse_arguments()
    run_folder_path = setup_run_directory()
    set_random_seed(args.seed)
    moves, positions = generate_trajectory(trajectory_size=args.trajectory_size, grid_size=args.bbox)
    print("\nGenerated Trajectory:")
    print(moves)
    print(positions)
    blocked_areas = define_blocked_areas(positions, BLOCK_RADIUS)
    scene_config_path = os.path.join(run_folder_path, "scene_config.json")
    save_trajectory_moves(moves, scene_config_path)
    generate_scene_config(
        num_shapes=args.num_shapes,
        times=args.times,
        camera_height=args.camera_height,
        save_config_path=scene_config_path,
        blocked_areas=blocked_areas,
        trajectory=positions,
        same_size=args.same_size,
        homogenous_texture=args.homogenous_texture,
    )

if __name__ == "__main__":
    main()

