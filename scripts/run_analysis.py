# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from pathlib import Path

from sage.analysis import RobotDataComparator


def main():
    parser = argparse.ArgumentParser(description="Generate image plots and metrics statistics for sim and real data.")
    parser.add_argument("--robot-name", type=str, required=True, help="Name of the robot (h1_2)")
    parser.add_argument(
        "--motion-source",
        type=str,
        choices=["amass"],
        required=True,
        help="Source of the motion",
    )

    parser.add_argument(
        "--motion-names",
        type=str,
        default="*",
        help=(
            "Motion names separated by commas (e.g., 'motion1,motion2'). Use '*' to include all motions in the"
            " {output_folder}/sim/{robot_name}/{motion_source} and {output_folder}/real/{robot_name}/{motion_source}"
            " directory."
        ),
    )

    parser.add_argument(
        "--sample-freq",
        type=int,
        default=200,
        help=(
            "Sample frequency (Hz). When calculating metrics, simulation and real data will be linearly interpolated "
            "according to the sample time step (dt = 1/sample-freq). Integer frequency values are more convenient "
            "to specify than decimal time steps, which is why frequency is used as input."
        ),
    )

    parser.add_argument("--output-folder", type=str, required=True, default=None, help="Output folder")
    parser.add_argument(
        "--valid-joints-file",
        type=str,
        default=None,
        help=(
            "Path to the valid joints file. If provided, the system will read valid joints from this file. If not"
            " provided, the system will automatically search for a file named '{robot_name}_valid_joints.txt' in the"
            " configs folder. If no valid joints file is found through either method, all robot joints will be used for"
            " visualization and data metrics."
        ),
    )

    parser.add_argument(
        "--metrics-summary-file-name",
        type=str,
        default=None,
        help=(
            "Specify the name of the metrics summary file. By default, the metrics summary will be saved as an Excel"
            " file named 'metrics_summary.xlsx' in the directory:"
            " '{args.output_folder}/metrics/{args.robot_name}/{args.motion_source}'. However, since a single motion"
            " source may be tested multiple times with different motion names, users may want to generate separate"
            " metrics summary files for each test. This parameter allows users to specify a custom file name for the"
            " metrics summary file."
        ),
    )

    args = parser.parse_args()
    args.robot_name = args.robot_name.lower()
    args.motion_source = args.motion_source.lower()

    data_comparator = RobotDataComparator(
        robot_name=args.robot_name,
        motion_source=args.motion_source,
        motion_names=args.motion_names,
        valid_joints_file=args.valid_joints_file,
        result_folder=args.output_folder,
        sample_dt=1 / args.sample_freq,
    )

    data_comparator.visualize_all_comparisons(
        output_dir=args.output_folder,
    )

    metrics_save_dir = Path(f"{args.output_folder}/metrics/{args.robot_name}/{args.motion_source}")

    data_comparator.analyze_all_data(metrics_save_dir, metrics_summary_file_name=args.metrics_summary_file_name)


if __name__ == "__main__":
    main()
