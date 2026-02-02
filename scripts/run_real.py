# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import glob
import os
import time

try:
    from sage.real_realman.realman_collector import realman_collector_main
except ImportError:
    realman_collector_main = None

from sage.real_so101.so101_lerobot_collector import so101_collector_main

try:
    from sage.real_unitree.unitree_collector import unitree_collector_main
except ImportError:
    unitree_collector_main = None

REST_PERIOD_SECONDS = 15


def run_motion(
    robot_name,
    motion_file,
    output_dir,
    motion_name=None,
    auto_start=False,
    robot_port=None,
    robot_type=None,
    robot_id=None,
):
    """Run a single motion file for the specified robot."""
    if robot_name == "h12" or robot_name == "g1":
        if unitree_collector_main is None:
            raise RuntimeError("Unitree collector not available. Install required dependencies.")
        unitree_collector_main(robot_name, motion_file, output_dir)
    elif robot_name == "realman":
        if realman_collector_main is None:
            raise RuntimeError("Realman collector not available. Install required dependencies.")
        realman_collector_main(motion_file, output_dir)
    elif robot_name == "so101":
        so101_collector_main(
            motion_file,
            output_dir,
            robot_port=robot_port,
            robot_type=robot_type,
            robot_id=robot_id,
            auto_start=auto_start,
            motion_name=motion_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run motion files on a robot and collect data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific motion files:
  python scripts/run_real.py --robot-name so101 --motion-files custom/file1.txt custom/file2.txt --output-folder output

  # Run all motion files in a folder:
  python scripts/run_real.py --robot-name so101 --motion-folder custom --output-folder output --auto-start
        """,
    )
    parser.add_argument(
        "--robot-name", action="store", type=str, help="Robot name: realman, h12, g1, or so101", required=True
    )
    parser.add_argument(
        "--motion-files",
        action="store",
        type=str,
        nargs="+",
        help="One or more motion files (relative to motion_files/<robot>/)",
        default=[],
    )
    parser.add_argument(
        "--motion-folder",
        action="store",
        type=str,
        help="Run all .txt motion files in this folder (relative to motion_files/<robot>/)",
        default=None,
    )
    parser.add_argument(
        "--output-folder", action="store", type=str, help="Output folder for saving data", required=True
    )
    parser.add_argument(
        "--repeats", action="store", type=int, default=1, help="Number of times to repeat each motion (default: 1)"
    )
    parser.add_argument(
        "--auto-start",
        "-y",
        action="store_true",
        help="Automatically start motion without waiting for user confirmation",
    )
    # SO-101 specific arguments
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO-101 robot (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so101_follower",
        help="Robot type for SO-101 (default: so101_follower)",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="my_awesome_follower_arm",
        help="Robot ID for SO-101 (default: my_awesome_follower_arm)",
    )
    args = parser.parse_args()

    # Validate that at least one motion source is provided
    if not args.motion_files and not args.motion_folder:
        parser.error("You must provide either --motion-files or --motion-folder")

    home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Collect motion files
    motion_files = list(args.motion_files)  # Start with explicitly listed files

    # Add files from folder if specified
    if args.motion_folder:
        folder_path = os.path.join(home_dir, "motion_files", args.robot_name, args.motion_folder)
        if not os.path.isdir(folder_path):
            print(f"Error: Motion folder not found: {folder_path}")
            exit(1)

        # Find all .txt files in the folder
        txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))

        if not txt_files:
            print(f"Warning: No .txt files found in {folder_path}")
        else:
            # Convert to relative paths (relative to motion_files/<robot>/)
            for txt_file in txt_files:
                rel_path = os.path.relpath(txt_file, os.path.join(home_dir, "motion_files", args.robot_name))
                if rel_path not in motion_files:  # Avoid duplicates
                    motion_files.append(rel_path)
            print(f"Found {len(txt_files)} motion files in {args.motion_folder}/")

    if not motion_files:
        print("Error: No motion files to run")
        exit(1)

    total_runs = len(motion_files) * args.repeats
    current_run = 0

    for motion_filename in motion_files:
        # Get motion name without extension for folder naming
        motion_name = os.path.splitext(os.path.basename(motion_filename))[0]

        # Extract output_subfolder from motion file path (e.g., "custom/motion.txt" -> "custom")
        output_subfolder = os.path.dirname(motion_filename) or "amass"

        # Build full motion file path
        motion_file = os.path.join(home_dir, "motion_files", args.robot_name, motion_filename)

        if not os.path.exists(motion_file):
            print(f"Warning: Motion file not found: {motion_file}")
            continue

        for run_idx in range(1, args.repeats + 1):
            current_run += 1

            # Output directory: output/real/<robot>/<source>/
            # (motion_name subdirectory created by collector's save function)
            # For repeats > 1, append run suffix to motion name
            output_dir = os.path.join(home_dir, args.output_folder, f"real/{args.robot_name}/{output_subfolder}")
            os.makedirs(output_dir, exist_ok=True)

            # Modify motion name for repeated runs to avoid overwriting
            effective_motion_name = motion_name if args.repeats == 1 else f"{motion_name}_run_{run_idx}"

            print(f"\n{'='*60}")
            print(f"Running motion: {effective_motion_name} (run {run_idx}/{args.repeats})")
            print(f"Progress: {current_run}/{total_runs} total runs")
            print(f"Output: {output_dir}/{effective_motion_name}/")
            print(f"{'='*60}\n")

            run_motion(
                args.robot_name,
                motion_file,
                output_dir,
                motion_name=effective_motion_name,
                auto_start=args.auto_start,
                robot_port=args.robot_port,
                robot_type=args.robot_type,
                robot_id=args.robot_id,
            )

            # Rest period between runs (skip after the last run)
            if current_run < total_runs:
                print(f"\n*** Resting for {REST_PERIOD_SECONDS} seconds to let robot cool off ***\n")
                time.sleep(REST_PERIOD_SECONDS)

    print(f"\n{'='*60}")
    print(f"All motions completed! Total runs: {total_runs}")
    print(f"{'='*60}\n")
