# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os

#from sage.real_realman.realman_collector import realman_collector_main
from sage.real_so101.so101_lerobot_collector import so101_collector_main
#from sage.real_unitree.unitree_collector import unitree_collector_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-name", action="store", type=str, help="Robot name: realman, h12, g1, or so101", required=False)
    parser.add_argument("--motion-files", action="store", type=str, help="Motion file text", required=False)
    parser.add_argument(
        "--output-folder", action="store", type=str, help="Output folder for saving data", required=False
    )
    args = parser.parse_args()

    home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(home_dir, args.output_folder, f"real/{args.robot_name}/amass")
    motion_file = os.path.join(home_dir, "motion_files", args.robot_name, "amass", args.motion_files)

    if args.robot_name == "h12" or args.robot_name == "g1":
        unitree_collector_main(args.robot_name, motion_file, output_dir)
    elif args.robot_name == "realman":
        realman_collector_main(motion_file, output_dir)
    elif args.robot_name == "so101":
        # SO-101 uses custom motion source folder, not amass
        output_dir = os.path.join(home_dir, args.output_folder, f"real/{args.robot_name}/custom")
        motion_file = os.path.join(home_dir, "motion_files", args.robot_name, args.motion_files)
        so101_collector_main(motion_file, output_dir)
