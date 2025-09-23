import os
import argparse

from joint_motion_gap.real_h1_2.real_h1_2 import real_h1_2_main
from joint_motion_gap.real_realman.real_realman import real_realman_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-name', action='store', type=str, help='Task name: stand, stand_w_waist, wb, squat', required=False, default='stand')
    parser.add_argument('--motion-files', action='store', type=str, help='Output folder for saving data', required=False, default='data_output')
    parser.add_argument('--output-folder', action='store', type=str, help='Output folder for saving data', required=False, default='data_output')
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_folder, f"real/{args.robot_name}/amass")
    
    if args.robot_name == "h1_2":
        real_h1_2_main(output_dir)
    elif args.robot_name == "realman":
        real_realman_main(args.motion_files, output_dir)