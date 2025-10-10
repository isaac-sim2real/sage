import os
import argparse

from joint_motion_gap.real_unitree.hardware_data_collect import unitree_robot_main
from joint_motion_gap.real_realman.real_realman import real_realman_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-name', action='store', type=str, help='Robot name: realman, h12 or g1', required=False)
    parser.add_argument('--motion-files', action='store', type=str, help='Motion file text', required=False)
    parser.add_argument('--output-folder', action='store', type=str, help='Output folder for saving data', required=False)
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_folder, f"real/{args.robot_name}/amass")
    
    if args.robot_name == "h12" or args.robot_name == "g1":
        unitree_robot_main(args, output_dir)
    elif args.robot_name == "realman":
        real_realman_main(args.motion_files, output_dir)