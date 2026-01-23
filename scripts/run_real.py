# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import time

# from sage.real_realman.realman_collector import realman_collector_main
from sage.real_so101.so101_lerobot_collector import so101_collector_main

# from sage.real_unitree.unitree_collector import unitree_collector_main

REST_PERIOD_SECONDS = 15


def run_motion(robot_name, motion_file, output_dir, auto_start=False, robot_port=None, robot_type=None, robot_id=None):
    """Run a single motion file for the specified robot."""
    if robot_name == "h12" or robot_name == "g1":
        unitree_collector_main(robot_name, motion_file, output_dir)
    elif robot_name == "realman":
        realman_collector_main(motion_file, output_dir)
    elif robot_name == "so101":
        so101_collector_main(
            motion_file,
            output_dir,
            robot_port=robot_port,
            robot_type=robot_type,
            robot_id=robot_id,
            auto_start=auto_start,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-name", action="store", type=str, help="Robot name: realman, h12, g1, or so101", required=True
    )
    parser.add_argument(
        "--motion-files",
        action="store",
        type=str,
        nargs="+",
        help="One or more motion files (relative to motion_files/<robot>/)",
        required=True,
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

    home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Determine motion source subfolder based on robot
    if args.robot_name == "so101":
        motion_subfolder = ""  # SO-101 uses custom folder structure
        output_subfolder = "custom"
    else:
        motion_subfolder = "amass"
        output_subfolder = "amass"

    total_runs = len(args.motion_files) * args.repeats
    current_run = 0

    for motion_filename in args.motion_files:
        # Get motion name without extension for folder naming
        motion_name = os.path.splitext(os.path.basename(motion_filename))[0]

        # Build full motion file path
        motion_file = os.path.join(home_dir, "motion_files", args.robot_name, motion_filename)

        if not os.path.exists(motion_file):
            print(f"Warning: Motion file not found: {motion_file}")
            continue

        for run_idx in range(1, args.repeats + 1):
            current_run += 1

            # Create output directory: output/<robot>/<output_subfolder>/<motion_name>/run_<N>/
            output_dir = os.path.join(
                home_dir, args.output_folder, f"real/{args.robot_name}/{output_subfolder}/{motion_name}/run_{run_idx}"
            )
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Running motion: {motion_name} (run {run_idx}/{args.repeats})")
            print(f"Progress: {current_run}/{total_runs} total runs")
            print(f"Output: {output_dir}")
            print(f"{'='*60}\n")

            run_motion(
                args.robot_name,
                motion_file,
                output_dir,
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
