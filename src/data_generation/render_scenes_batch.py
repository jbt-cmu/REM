import argparse
import os
import subprocess
import sys

# Path to the directory containing trajectories
TRAJECTORY_DIR = os.path.join(".", "src", "trajectories")

# Path to the Blender executable
# On Windows, you might need to provide the full path, e.g.,
# BLENDER_EXE = r"C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
BLENDER_EXE = "blender"  # Adjust this if Blender is not in your system PATH

# Path to the render_scene.py script
PYTHON_SCRIPT = os.path.join(".", "src", "data_generation", "render_scene.py")


def main():
    parser = argparse.ArgumentParser(description="Run all trajectories.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a trajectory index, e.g. '0003' to skip lower indices.",
    )
    args = parser.parse_args()

    # Get a list of all trajectory folders
    trajectory_folders = [
        os.path.join(TRAJECTORY_DIR, name)
        for name in os.listdir(TRAJECTORY_DIR)
        if os.path.isdir(os.path.join(TRAJECTORY_DIR, name))
    ]

    # Sort the folders for consistent processing order
    trajectory_folders.sort()

    # If --resume is set, skip folders with index lower than this
    if args.resume:
        try:
            resume_int = int(args.resume)
            filtered_folders = []
            for folder in trajectory_folders:
                folder_name = os.path.basename(folder)
                prefix = folder_name.split("-")[
                    0
                ]  # e.g. "0003" if folder is named "0003-run"
                try:
                    if int(prefix) >= resume_int:
                        filtered_folders.append(folder)
                except ValueError:
                    # If folder name doesn't parse as int, we ignore or skip
                    pass
            trajectory_folders = filtered_folders
        except ValueError:
            print(f"Invalid resume value: {args.resume}, expected a numeric string.")
            sys.exit(1)

    # Process each trajectory folder
    for folder in trajectory_folders:
        print(f"Processing trajectory folder: {folder}")
        command = [
            BLENDER_EXE,
            "--background",
            "--python",
            PYTHON_SCRIPT,
            "--",
            "--config-file",
            folder,
        ]
        try:
            subprocess.run(command, check=True)
            print(f"Completed processing of {folder}\n")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {folder}: {e}\n")


if __name__ == "__main__":
    main()

