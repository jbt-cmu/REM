# render_scene.py
"""
This script replicates the functionality of the original run_simulation.py but using Blender.
It reads a scene configuration from a JSON file, creates the scene in Blender,
moves the camera along a trajectory, captures images, and saves annotations.

To run this script, use Blender in background mode:

blender --background --python render_scene.py -- [arguments]

Arguments:
--draft: Use draft mode (faster rendering)
--no-description: Exclude description from annotations.
--no-position: Exclude position from annotations.
--config-file: Path to a configuration directory containing the scene_config.json file.
--debug: Enable debug mode with additional logging and outputs.
"""

import argparse
import csv
import json
import math
import os
import platform
import sys
from collections import Counter

import bpy

# Import mathutils for transformations
import mathutils
from bpy_extras.object_utils import world_to_camera_view


def parse_args():
    # Extract args passed after '--'
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Run Simulation Script.")
    parser.add_argument(
        "--no-description",
        action="store_true",
        help="Exclude description from annotations.",
    )
    parser.add_argument(
        "--no-position", action="store_true", help="Exclude position from annotations."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to a configuration directory containing the scene_config.json file.",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Use draft mode for faster rendering with lower quality",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging and outputs",
    )
    args_cli = parser.parse_args(argv)
    return args_cli


def load_scene_config(config_dir):
    config_file = os.path.join(config_dir, "scene_config.json")
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found.")
        sys.exit(1)
    with open(config_file, "r") as f:
        config_data = json.load(f)
    return config_data


def load_trajectory_moves(config_dir):
    """
    Loads the trajectory moves from the trajectory_moves.txt file.
    """
    trajectory_file = os.path.join(config_dir, "trajectory_moves.txt")
    if not os.path.exists(trajectory_file):
        print(f"Trajectory file not found: {trajectory_file}")
        return []
    with open(trajectory_file, "r") as f:
        moves = [line.strip() for line in f if line.strip()]
    return moves


def create_material(name, diffuse_color, roughness=0.2, metallic=0.0):
    """
    Creates a material with the specified properties.
    Now accepts roughness and metallic parameters from the object attributes.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*diffuse_color, 1.0)  # RGBA
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def adjust_object_position(obj):
    """
    Adjust the object's Z position so that it sits correctly on the ground by aligning
    the lowest point of its bounding box to Z=0.
    """
    # Ensure the object has mesh data
    if obj.type != "MESH" or not obj.data:
        return

    bpy.context.view_layer.update()

    world_matrix = obj.matrix_world
    bbox = [world_matrix @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_z = min([v.z for v in bbox])
    offset_z = -min_z
    obj.location.z += offset_z


def create_object(obj_data):
    obj_name = obj_data["name"]
    obj_type = obj_data["type"]
    attributes = obj_data.get("attributes", {})
    obj_config_data = obj_data.get("config", {})

    diffuse_color = attributes.get("diffuse_color", [1.0, 0.0, 0.0])
    roughness = attributes.get("roughness", 0.8)
    metallic = attributes.get("metallic", 0.0)

    mat = create_material(
        obj_name + "_mat", diffuse_color, roughness=roughness, metallic=metallic
    )

    if obj_type == "cuboid" or obj_type == "rod":
        size = obj_config_data.get("size", [1.0, 1.0, 1.0])
        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.dimensions = size

        bevel_mod = obj.modifiers.new(name="Bevel", type="BEVEL")
        bevel_mod.width = min(size) * 0.03
        bevel_mod.segments = 4
        bevel_mod.limit_method = "ANGLE"
        bevel_mod.angle_limit = math.radians(30)

        for face in obj.data.polygons:
            face.use_smooth = True

    elif obj_type == "sphere":
        radius = obj_config_data.get("radius", 1.0)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius, segments=64, ring_count=32, location=(0, 0, 0)
        )
        obj = bpy.context.active_object
        for face in obj.data.polygons:
            face.use_smooth = True

    elif obj_type == "cone":
        radius = obj_config_data.get("radius", 1.0)
        height = obj_config_data.get("height", 2.0)
        vertices = 64
        bpy.ops.mesh.primitive_cone_add(
            radius1=radius, depth=height, vertices=vertices, location=(0, 0, 0)
        )
        obj = bpy.context.active_object
        for face in obj.data.polygons:
            face.use_smooth = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.mark_sharp(clear=True)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.edges_select_sharp(sharpness=0.5236)
        bpy.ops.mesh.mark_sharp()
        bpy.ops.object.mode_set(mode="OBJECT")

        edge_split = obj.modifiers.new("EdgeSplit", type="EDGE_SPLIT")
        edge_split.use_edge_angle = False
        edge_split.use_edge_sharp = True

    elif obj_type == "cylinder":
        radius = obj_config_data.get("radius", 1.0)
        height = obj_config_data.get("height", 2.0)
        vertices = 64
        bev_width = radius * 0.15

        bpy.ops.mesh.primitive_cylinder_add(
            radius=radius, depth=height, vertices=vertices, location=(0, 0, 0)
        )
        obj = bpy.context.active_object
        bevel_mod = obj.modifiers.new(name="Bevel", type="BEVEL")
        bevel_mod.width = bev_width
        bevel_mod.segments = 5
        bevel_mod.limit_method = "ANGLE"
        bevel_mod.angle_limit = math.radians(30)

        for face in obj.data.polygons:
            face.use_smooth = True

    else:
        print(f"Unknown object type: {obj_type}")
        return None

    obj.name = obj_name
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    position = obj_data["position"]
    obj.location.x = position[0]
    obj.location.y = position[1]
    adjust_object_position(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    return obj


def setup_scene(config_data, draft_mode=False):
    is_macos = platform.system() == "Darwin"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False

    if draft_mode:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = 32
        scene.eevee.use_ssr = False
        scene.eevee.use_ssr_refraction = False
        scene.eevee.use_gtao = False
        scene.eevee.use_bloom = False
        scene.eevee.use_shadow_high_bitdepth = False
        scene.eevee.volumetric_samples = 32
        scene.eevee.shadow_cube_size = "256"
        scene.eevee.shadow_cascade_size = "256"
        scene.eevee.use_soft_shadows = False
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 256
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.005
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 12
        scene.cycles.min_bounces = 3
        scene.cycles.caustics_reflective = True
        scene.cycles.caustics_refractive = True
        scene.cycles.blur_glossy = 1.0
        scene.cycles.device = "GPU"

        preferences = bpy.context.preferences
        cycles_prefs = preferences.addons["cycles"].preferences

        if is_macos:
            cycles_prefs.compute_device_type = "METAL"
            cycles_prefs.get_devices()
            for device in cycles_prefs.devices:
                device.use = True
            print("==== Using METAL ====")
        else:
            # Attempt to use OPTIX first, then CUDA, else quit
            cycles_prefs.get_devices()
            if any(d.type == "OPTIX" for d in cycles_prefs.devices):
                cycles_prefs.compute_device_type = "OPTIX"
                cycles_prefs.get_devices()
                for device in cycles_prefs.devices:
                    device.use = device.type == "OPTIX"
                print("==== Using OPTIX ====")
            elif any(d.type == "CUDA" for d in cycles_prefs.devices):
                cycles_prefs.compute_device_type = "CUDA"
                cycles_prefs.get_devices()
                for device in cycles_prefs.devices:
                    device.use = device.type == "CUDA"
                print("==== Using CUDA ====")
            else:
                print("Error: No OPTIX or CUDA device found! Exiting.")
                import sys

                sys.exit(1)

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.exposure = 0.0

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output_node = nodes.new(type="ShaderNodeOutputWorld")
    background_node = nodes.new(type="ShaderNodeBackground")
    sky_texture = nodes.new(type="ShaderNodeTexSky")
    sky_texture.sky_type = "PREETHAM"
    sky_texture.sun_elevation = math.radians(60)
    sky_texture.sun_rotation = math.radians(135)
    sky_texture.turbidity = 2.0
    sky_texture.ground_albedo = 0.4
    links.new(sky_texture.outputs["Color"], background_node.inputs["Color"])
    links.new(background_node.outputs["Background"], output_node.inputs["Surface"])
    background_node.inputs["Strength"].default_value = 1.5

    bpy.ops.mesh.primitive_plane_add(size=200)
    ground = bpy.context.active_object
    ground.name = "GroundPlane"
    ground.location = (0, 0, 0)
    ground_mat = create_material(
        "GroundPlane_mat", (0.1, 0.1, 0.1), roughness=0.8, metallic=0.2
    )
    ground.data.materials.append(ground_mat)

    if not draft_mode:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=5)
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.shade_smooth()

    light_data = bpy.data.lights.new(name="SunLight", type="SUN")
    light_data.energy = 8.0
    light_data.angle = math.radians(0.5)
    light_data.color = (1.0, 0.95, 0.85)
    light_obj = bpy.data.objects.new(name="SunLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, -5, 10)
    light_obj.rotation_euler = (math.radians(60), 0, math.radians(45))

    fill_light_data = bpy.data.lights.new(name="FillLight", type="AREA")
    fill_light_data.energy = 70.0
    fill_light_data.size = 10.0
    fill_light_data.color = (0.8, 0.85, 1.0)
    fill_light_obj = bpy.data.objects.new(name="FillLight", object_data=fill_light_data)
    bpy.context.collection.objects.link(fill_light_obj)
    fill_light_obj.location = (-5, -5, 5)
    fill_light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))

    scene_settings = config_data.get("scene_settings", {})
    camera_height = scene_settings.get("camera_height", 1.25)
    if camera_height is None:
        camera_height = 1.25

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(90)
    cam.location = (-5, 0, camera_height)
    direction = mathutils.Vector((1.0, 0.0, 0.0))
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()
    scene.camera = cam

    config_objects = config_data.get("objects", [])
    for obj_data in config_objects:
        create_object(obj_data)

    return cam


def move_camera_forward(camera, distance):
    direction = camera.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
    camera.location += direction * distance


def rotate_camera(camera, angle_degrees):
    # Convert degrees to radians
    angle_radians = math.radians(angle_degrees)

    # Create a rotation matrix around the global Z axis
    R = mathutils.Matrix.Rotation(angle_radians, 4, "Z")

    # Get the camera's current world position
    loc = camera.matrix_world.to_translation()

    # Create translation matrices to move the camera to (0,0,0) and back
    T_to_origin = mathutils.Matrix.Translation(-loc)
    T_back = mathutils.Matrix.Translation(loc)

    # Apply the translate-rotate-translate sequence to camera.matrix_world
    camera.matrix_world = T_back @ R @ T_to_origin @ camera.matrix_world


def execute_trajectory_moves(env, moves):
    for move in moves:
        parts = move.split()
        if not parts:
            continue

        if parts[0] == "straight":
            env.move_forward(1)
        elif parts[0] == "turn":
            if len(parts) >= 4:
                direction = parts[1]
                try:
                    angle = float(parts[2])
                    if direction == "left":
                        env.rotate(angle)
                    elif direction == "right":
                        env.rotate(-angle)
                    else:
                        print(f"Unknown turn direction: {direction}")
                except ValueError:
                    print(f"Invalid angle in turn command: {move}")
        else:
            print(f"Unknown trajectory move: {move}")


class CameraAgentEnv:
    def __init__(
        self, image_dir, include_description=True, include_position=True, debug=False
    ):
        self.image_dir = image_dir
        self.include_description = include_description
        self.include_position = include_position
        self.debug = debug
        self.image_counter = 0
        self.time_index = -1
        self.semantic_data = []
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        self.annotations_file = os.path.join(self.image_dir, "annotations.csv")
        with open(self.annotations_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_filename", "annotation", "visible_objects"])

    def move_forward(self, total_distance):
        if total_distance <= 0:
            print("Total distance must be positive.")
            return
        num_steps = int(total_distance)
        for step in range(num_steps):
            move_camera_forward(self.camera, 1)
            self.capture_image(
                f"move_forward_1m",
                f"This image was taken after moving forward by 1 meter.",
            )
        remaining_distance = total_distance - num_steps
        if remaining_distance > 0:
            move_camera_forward(self.camera, remaining_distance)
            self.capture_image(
                f"move_forward_{remaining_distance}m",
                f"This image was taken after moving forward by {remaining_distance} meters.",
            )

    def rotate(self, total_angle):
        if total_angle % 15 != 0:
            print("Total angle must be a multiple of 15 degrees.")
            return
        num_steps = int(abs(total_angle) // 15)
        angle_step = 15 if total_angle > 0 else -15
        for step in range(num_steps):
            rotate_camera(self.camera, angle_step)
            self.capture_image(
                f"rotate_{'left' if angle_step > 0 else 'right'}_15",
                f"This image was taken after rotating {abs(angle_step)} degrees.",
            )
        remaining_angle = total_angle - angle_step * num_steps
        if remaining_angle != 0:
            rotate_camera(self.camera, remaining_angle)
            self.capture_image(
                f"rotate_{'left' if remaining_angle > 0 else 'right'}_{abs(remaining_angle)}",
                f"This image was taken after rotating {abs(remaining_angle)} degrees.",
            )

    def is_object_visible(self, obj):
        scene = bpy.context.scene
        camera = self.camera
        if not obj.visible_get():
            return False
        if obj.type != "MESH":
            return False
        obj_bound_box = [
            obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box
        ]
        for corner in obj_bound_box:
            co_ndc = world_to_camera_view(scene, camera, corner)
            if 0.0 <= co_ndc.x <= 1.0 and 0.0 <= co_ndc.y <= 1.0 and co_ndc.z >= 0.0:
                return True
        return False

    def calculate_object_pixel_percentages(self):
        scene = bpy.context.scene
        render = scene.render

        if not self.camera or not isinstance(self.camera, bpy.types.Object):
            raise ValueError("Camera reference is invalid.")

        original_filepath = render.filepath
        original_engine = render.engine
        original_film_transparent = render.film_transparent
        original_world = scene.world
        original_taa_render_samples = scene.eevee.taa_render_samples
        original_view_transform = scene.view_settings.view_transform

        original_materials = {}
        original_use_nodes = {}
        hidden_objects = []

        try:
            render.engine = "BLENDER_EEVEE_NEXT"
            render.film_transparent = False
            scene.eevee.taa_render_samples = 1
            scene.view_settings.view_transform = "Raw"

            scene.world = bpy.data.worlds.new("ShadelessWorld")
            scene.world.use_nodes = True
            nodes = scene.world.node_tree.nodes
            links = scene.world.node_tree.links
            nodes.clear()
            bg_node = nodes.new(type="ShaderNodeBackground")
            bg_node.inputs["Color"].default_value = (0, 0, 0, 1)
            output_node = nodes.new(type="ShaderNodeOutputWorld")
            links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

            object_colors = {}
            used_colors = set()
            color_steps = list(range(0, 256, 32))
            color_generator = (
                (r, g, b)
                for r in color_steps
                for g in color_steps
                for b in color_steps
                if not (r == g == b) and not (r == 0 and g == 0 and b == 0)
            )

            for obj in scene.objects:
                if obj.type == "MESH" and obj.visible_get():
                    if obj.active_material:
                        original_materials[obj.name] = obj.active_material
                        original_use_nodes[obj.name] = obj.active_material.use_nodes
                    else:
                        original_materials[obj.name] = None
                        original_use_nodes[obj.name] = False

                    mat = bpy.data.materials.new(name=f"Shadeless_Mat_{obj.name}")
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    while nodes:
                        nodes.remove(nodes[0])

                    emission_node = nodes.new(type="ShaderNodeEmission")
                    emission_node.inputs["Strength"].default_value = 1.0

                    max_attempts = 1000
                    attempt = 0
                    while attempt < max_attempts:
                        try:
                            color = next(color_generator)
                            if color not in used_colors:
                                used_colors.add(color)
                                break
                        except StopIteration:
                            raise ValueError(
                                f"Unable to assign unique color to object {obj.name}."
                            )
                        attempt += 1

                    if attempt >= max_attempts:
                        raise ValueError(
                            f"Failed to find unique color after {max_attempts} attempts"
                        )

                    normalized_color = tuple(c / 255.0 for c in color)
                    emission_node.inputs["Color"].default_value = (
                        *normalized_color,
                        1,
                    )
                    object_colors[color] = obj.name

                    output_node = nodes.new(type="ShaderNodeOutputMaterial")
                    links.new(
                        emission_node.outputs["Emission"], output_node.inputs["Surface"]
                    )

                    obj.data.materials.clear()
                    obj.data.materials.append(mat)

            for obj in scene.objects:
                if obj.type not in ["MESH", "CAMERA"]:
                    if not obj.hide_render:
                        hidden_objects.append(obj)
                        obj.hide_render = True

            shadeless_image_filename = f"shadeless_{self.image_counter:04d}.png"
            shadeless_image_path = os.path.join(
                self.image_dir, shadeless_image_filename
            )
            render.filepath = shadeless_image_path
            bpy.ops.render.render(write_still=True)

            img = bpy.data.images.load(shadeless_image_path)
            pixels = list(img.pixels)
            img_width = img.size[0]
            img_height = img.size[1]
            total_pixels = img_width * img_height

            pixel_data = []
            for i in range(0, len(pixels), 4):
                r = int(pixels[i] * 255 + 0.5)
                g = int(pixels[i + 1] * 255 + 0.5)
                b = int(pixels[i + 2] * 255 + 0.5)
                pixel_data.append((r, g, b))
            color_counts = Counter(pixel_data)

            object_percentages = {}
            tolerance = 5

            for color, count in color_counts.items():
                if color == (0, 0, 0):
                    continue
                matched = False
                for obj_color in object_colors.keys():
                    if all(abs(color[i] - obj_color[i]) <= tolerance for i in range(3)):
                        object_name = object_colors[obj_color]
                        percentage = (count / total_pixels) * 100
                        object_percentages[object_name] = (
                            object_percentages.get(object_name, 0.0) + percentage
                        )
                        matched = True
                        break

            bpy.data.images.remove(img)
            if not self.debug and os.path.exists(shadeless_image_path):
                os.remove(shadeless_image_path)

        except Exception as e:
            print(
                f"Error in calculate_object_pixel_percentages for image {self.image_counter}: {e}"
            )
            raise

        finally:
            render.filepath = original_filepath
            render.engine = original_engine
            render.film_transparent = original_film_transparent
            scene.eevee.taa_render_samples = original_taa_render_samples
            scene.view_settings.view_transform = original_view_transform

            if scene.world.name == "ShadelessWorld":
                bpy.data.worlds.remove(scene.world, do_unlink=True)
                scene.world = original_world

            for obj in scene.objects:
                if obj.type == "MESH" and obj.visible_get():
                    obj.data.materials.clear()
                    if obj.name in original_materials and original_materials[obj.name]:
                        obj.data.materials.append(original_materials[obj.name])
                        obj.active_material.use_nodes = original_use_nodes.get(
                            obj.name, False
                        )
                    else:
                        obj.active_material = None

            for obj in hidden_objects:
                obj.hide_render = False

        return object_percentages

    def capture_image(self, label, annotation):
        self.time_index += 1
        image_filename = f"image_{self.image_counter:04d}_{label}.png"
        image_path = os.path.join(self.image_dir, image_filename)
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)
        position = self.camera.matrix_world.translation
        x = position.x
        y = position.y
        x_formatted = f"{x:.3f}"
        y_formatted = f"{y:.3f}"
        if self.include_position:
            if annotation:
                annotation += " "
            annotation += (
                f"Position relative to beginning (x, y): ({x_formatted}, {y_formatted})"
            )

        object_percentages = self.calculate_object_pixel_percentages()

        # Reorder visible objects from left to right based on their camera-space x coordinate
        scene = bpy.context.scene
        camera = self.camera
        object_list = []
        for obj_name, percentage in object_percentages.items():
            obj = bpy.data.objects.get(obj_name)
            if obj is not None:
                co_ndc = world_to_camera_view(scene, camera, obj.location)
                object_list.append((co_ndc.x, obj_name, percentage))

        object_list.sort(key=lambda x: x[0])
        visible_objects = [
            f"{obj_name} ({percentage:.2f}%)" for _, obj_name, percentage in object_list
        ]

        with open(self.annotations_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([image_filename, annotation, ", ".join(visible_objects)])

        self.image_counter += 1

    def reset_camera_orientation(self):
        direction = mathutils.Vector((1.0, 0.0, 0.0))
        rot_quat = direction.to_track_quat("-Z", "Y")
        self.camera.rotation_euler = rot_quat.to_euler()

    def reset(self):
        self.capture_image("init", "This image was taken at the initial position.")


def main():
    args = parse_args()
    include_description = not args.no_description
    include_position = not args.no_position
    config_dir = args.config_file
    draft_mode = args.draft
    debug_mode = args.debug

    if not config_dir:
        print("Please provide a configuration directory using --config-file")
        sys.exit(1)

    config_dir = os.path.abspath(config_dir)
    if not os.path.isdir(config_dir):
        print(f"Configuration directory {config_dir} not found.")
        sys.exit(1)

    config_data = load_scene_config(config_dir)
    images_dir = os.path.join(config_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    camera = setup_scene(config_data, draft_mode)

    env = CameraAgentEnv(
        image_dir=images_dir,
        include_description=include_description,
        include_position=include_position,
        debug=debug_mode,
    )
    env.camera = camera

    env.reset()

    trajectory_moves = load_trajectory_moves(config_dir)
    execute_trajectory_moves(env, trajectory_moves)


if __name__ == "__main__":
    main()

