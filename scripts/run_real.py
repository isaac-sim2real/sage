import argparse
import os

from joint_motion_gap.real_realman.real_realman import real_realman_main
from joint_motion_gap.real_unitree.hardware_data_collect import unitree_robot_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-name", action="store", type=str, help="Robot name: realman, h12 or g1", required=False)
    parser.add_argument("--motion-files", action="store", type=str, help="Motion file text", required=False)
    parser.add_argument(
        "--output-folder", action="store", type=str, help="Output folder for saving data", required=False
    )
    args = parser.parse_args()

    home_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(home_dir, args.output_folder, f"real/{args.robot_name}/amass")
    motion_file = os.path.join(home_dir, "motion_files", args.robot_name, "amass", args.motion_files)

    if args.robot_name == "h12" or args.robot_name == "g1":
        unitree_robot_main(args.robot_name, motion_file, output_dir)
    elif args.robot_name == "realman":
        real_realman_main(motion_file, output_dir)
