import logging
import time
from pathlib import Path
from typing import List, Tuple

# Import functions from create_scene_config.py
from create_scene_config import (
    parse_arguments,
    setup_run_directory,
    set_random_seed,
    generate_trajectory,
    define_blocked_areas,
    generate_scene_config,
    save_trajectory_moves,
    BLOCK_RADIUS,
)

# Configure logging
logging.basicConfig(
    # make the logging minimal (set to INFO)
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dataset_generation.log")],
)

TRAJECTORY_SIZES = [256]
NUM_SHAPES_LIST = [30]  # Allowed number of shapes
NUM_EXAMPLES = 50
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Allowed times arrays:
ALLOWED_TIMES_SETS = [
    [2,2,1,1],
]


class ArgumentNamespace:
    """Mock ArgumentNamespace to simulate parse_arguments() return value"""

    def __init__(self, num_shapes, trajectory_size, times):
        self.num_shapes = num_shapes
        self.times = times
        self.trajectory_size = trajectory_size
        self.camera_height = None
        self.seed = None
        self.same_size = True
        self.homogenous_texture = True


def get_valid_combinations() -> List[Tuple[int, int, List[int]]]:
    """
    Generate valid combinations of parameters:
    - num_shapes in [8, 16, 24]
    - trajectory_size in TRAJECTORY_SIZES
    - times in ALLOWED_TIMES_SETS
    Each combination must satisfy sum(t+1 for t in times) <= num_shapes
    """
    valid_combinations = []
    for num_shapes in NUM_SHAPES_LIST:
        for traj_size in TRAJECTORY_SIZES:
            for times_set in ALLOWED_TIMES_SETS:
                sum_instances = sum((t + 1) for t in times_set)
                if sum_instances <= num_shapes:
                    valid_combinations.append((num_shapes, traj_size, times_set))
    return valid_combinations


def run_single_generation(
    num_shapes: int, trajectory_size: int, times: List[int], example_num: int
) -> bool:
    """
    Run a single generation with retries.
    """
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(
                f"Running example {example_num}: num_shapes={num_shapes}, "
                f"trajectory_size={trajectory_size}, times={times} "
                f"(Attempt {attempt + 1}/{MAX_RETRIES})"
            )

            args = ArgumentNamespace(num_shapes, trajectory_size, times)

            # Setup run directory
            run_folder_path = setup_run_directory()

            # Generate trajectory
            moves, positions = generate_trajectory(
                trajectory_size=args.trajectory_size,
            )

            # Define blocked areas
            blocked_areas = define_blocked_areas(positions, BLOCK_RADIUS)

            # Generate scene config
            scene_config_path = Path(run_folder_path) / "scene_config.json"

            # Save trajectory moves
            save_trajectory_moves(moves, scene_config_path)

            # Generate scene configuration
            generate_scene_config(
                num_shapes=args.num_shapes,
                times=args.times,
                camera_height=args.camera_height,
                save_config_path=str(scene_config_path),
                blocked_areas=blocked_areas,
                trajectory=positions,
                same_size=args.same_size,
                homogenous_texture=args.homogenous_texture,
            )

            return True

        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                logging.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error("Max retries reached. Moving to next combination.")
                return False

    return False


def main():
    logging.info("Starting dataset generation")

    # Get valid combinations
    combinations = get_valid_combinations()
    total_combinations = len(combinations) * NUM_EXAMPLES

    # Log the combinations that will be generated
    logging.info("\nValid combinations to be generated:")
    for num_shapes, traj, times_set in combinations:
        logging.info(
            f"Num Shapes: {num_shapes}, Trajectory: {traj}, Times: {times_set}"
        )
    logging.info(f"\nTotal unique combinations: {len(combinations)}")
    logging.info(f"Total examples to generate: {total_combinations}")

    # Initialize counters
    successful = 0
    failed = 0
    current = 0

    # Track start time
    start_time = time.time()

    try:
        for num_shapes, traj_size, times_set in combinations:
            for example in range(NUM_EXAMPLES):
                current += 1

                # Log progress
                progress = (current / total_combinations) * 100
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / current
                estimated_remaining = avg_time_per_item * (total_combinations - current)

                logging.info(
                    f"\nProgress: {progress:.1f}% ({current}/{total_combinations})"
                )
                logging.info(
                    f"Estimated time remaining: {estimated_remaining/60:.1f} minutes"
                )

                if run_single_generation(num_shapes, traj_size, times_set, example):
                    successful += 1
                else:
                    failed += 1

    except KeyboardInterrupt:
        logging.info("\nGeneration interrupted by user")
    finally:
        total_time = time.time() - start_time

        logging.info("\n=== Generation Summary ===")
        logging.info(f"Total combinations attempted: {current}/{total_combinations}")
        logging.info(f"Successful generations: {successful}")
        logging.info(f"Failed generations: {failed}")
        logging.info(f"Total time elapsed: {total_time/60:.1f} minutes")
        if current > 0:
            logging.info(
                f"Average time per generation: {total_time/current:.1f} seconds"
            )

        if failed > 0:
            logging.warning(f"Warning: {failed} generations failed")

        if successful == 0:
            logging.error("Error: No successful generations")
            return 1

        logging.info("Dataset generation completed")
        return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

