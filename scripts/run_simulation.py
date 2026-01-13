# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Benchmark joint motion in Isaac Sim")
parser.add_argument("--robot-name", type=str, required=True, help="Robot name")
parser.add_argument(
    "--motion-source",
    type=str,
    choices=["amass"],
    required=True,
    help="Source of the motion",
)
parser.add_argument(
    "--motion-files",
    type=str,
    required=True,
    help="Path to motion file(s) or directory. Can be a single file, multiple files "
    "(comma-separated), or a directory.",
)
parser.add_argument("--valid-joints-file", type=str, required=False, default=None, help="Path to valid joints file")
parser.add_argument("--output-folder", type=str, required=True, help="Path to output folder")
parser.add_argument("--fix-root", action="store_true", default=False, help="Fix root joint")
parser.add_argument("--dt", type=float, default=0.005, help="Physics dt in seconds")
parser.add_argument("--decimation", type=int, default=4, help="Number of physics steps per control step")
parser.add_argument("--render-interval", type=int, default=4, help="Number of physics steps per render step")
parser.add_argument(
    "--original-control-freq", type=int, default=None, help="Original control frequency of motion files"
)
parser.add_argument("--motion-speed-factor", type=float, default=1.0, help="Speed factor for motion playback")
parser.add_argument(
    "--kp",
    nargs="+",
    type=float,
    default=None,
    help="Joint stiffness override. Single value or list of values for each valid joint",
)
parser.add_argument(
    "--kd",
    nargs="+",
    type=float,
    default=None,
    help="Joint damping override. Single value or list of values for each valid joint",
)
parser.add_argument("--solver-type", type=int, default=1, help="Solver type TGS = 1 PGS = 0")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse command line arguments
args = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

from sage.simulation import JointMotionBenchmark, get_motion_files, get_motion_name, log_message  # noqa: E402


def main():
    """Main function."""
    # Get list of motion files to process
    motion_files = get_motion_files(args.motion_files)
    if not motion_files:
        raise ValueError(f"No motion files found in {args.motion_files}")

    log_message(f"Found {len(motion_files)} motion files to process")

    # Initialize benchmark environment
    benchmark = JointMotionBenchmark(args)

    # Process each motion file
    for motion_file in motion_files:
        try:
            motion_name = get_motion_name(motion_file)
            log_message(f"################### PROCESSING {motion_file} ###################")
            # Set up and run benchmark for this motion
            benchmark.setup_motion(motion_file, motion_name)
            benchmark.run_benchmark()

        except Exception as e:
            log_message(f"Error processing {motion_file}: {str(e)}")
            continue


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
