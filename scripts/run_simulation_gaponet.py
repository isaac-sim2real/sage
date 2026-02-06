#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Run SAGE simulation with GapONet actuator (Modular Version).

This script uses a modular architecture similar to run_simulation.py
but with IsaacLab and GapONet support.

"""

import argparse

# Import SAGE utilities
from sage.simulation import get_motion_files, get_motion_name, log_message


def main():
    parser = argparse.ArgumentParser(description="SAGE simulation with GapONet actuator")

    # SAGE arguments (same as run_simulation.py)
    parser.add_argument("--robot-name", type=str, required=True, help="Robot name (h1_2)")
    parser.add_argument("--motion-files", type=str, required=True, help="Path to motion file(s) or directory")
    parser.add_argument("--valid-joints-file", type=str, default=None, help="Path to valid joints file")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to output folder")
    parser.add_argument("--fix-root", action="store_true", default=False, help="Fix root joint")
    parser.add_argument("--physics-freq", type=int, default=200, help="Physics timestep frequency")
    parser.add_argument("--render-freq", type=int, default=200, help="Render timestep frequency")
    parser.add_argument("--control-freq", type=int, default=50, help="Control frequency")
    parser.add_argument(
        "--original-control-freq",
        type=int,
        default=None,
        help="Original motion data frequency (for resampling, e.g., 30 for AMASS)",
    )
    parser.add_argument("--kp", type=float, default=100.0, help="Joint stiffness")
    parser.add_argument("--kd", type=float, default=2.0, help="Joint damping")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--enable-realtime", action="store_true", help="Enable realtime sync (slower but matches real time)"
    )

    # GapONet arguments
    parser.add_argument(
        "--gaponet-model", type=str, default=None, help="Path to GapONet model (.pt) - required if --use-gaponet"
    )
    parser.add_argument("--gaponet-action-scale", type=float, default=1.0, help="GapONet action scale")
    parser.add_argument("--use-gaponet", action="store_true", default=False, help="Use GapONet actuator")
    parser.add_argument(
        "--no-use-gaponet",
        action="store_false",
        dest="use_gaponet",
        help="Disable GapONet, use ImplicitActuator (default)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.use_gaponet and not args.gaponet_model:
        parser.error("--gaponet-model is required when --use-gaponet is specified")

    # Initialize IsaacLab app
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import IsaacLab modules after app initialization
    from sage.simulation_lab import GapONetBenchmark  # noqa: E402

    # Get list of motion files
    motion_files = get_motion_files(args.motion_files)
    if not motion_files:
        raise ValueError(f"No motion files found in {args.motion_files}")

    log_message(f"Found {len(motion_files)} motion files to process")

    # Initialize benchmark
    benchmark = GapONetBenchmark(args)

    # Process each motion file
    for motion_file in motion_files:
        try:
            motion_name = get_motion_name(motion_file)
            log_message(f"################### PROCESSING {motion_file} ###################")

            # Set up and run benchmark
            benchmark.set_motion(motion_file, motion_name)
            benchmark.run_benchmark()

        except Exception as e:
            log_message(f"Error processing {motion_file}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    log_message("All motions processed!")

    # Cleanup
    benchmark.cleanup()
    simulation_app.close()


if __name__ == "__main__":
    main()
