# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import csv
import os
import time

import numpy as np
import pandas as pd
from realman_dual_arm import HumanoidDualArmCollector
from scipy.interpolate import interp1d


def interpolate_motion(seq, original_freq, target_freq):
    n_frames, n_joints = seq.shape
    duration = (n_frames - 1) / original_freq
    t_src = np.linspace(0, duration, n_frames)
    new_n_frames = int(duration * target_freq) + 1
    t_dst = np.linspace(0, duration, new_n_frames)
    seq_interp = np.zeros((new_n_frames, n_joints))
    for j in range(n_joints):
        f = interp1d(t_src, seq[:, j], kind="linear")
        seq_interp[:, j] = f(t_dst)
    return seq_interp


def load_motion_from_txt(txt_file_path):
    """
    Load motion data from TXT file

    Args:
        txt_file_path: TXT file path

    Returns:
        left_seq: left arm sequence (n, 7)
        right_seq: right arm sequence (n, 7)
        robot_joint_names: list of joint names
        motion_frequence: motion frequency (default 50Hz)
        motion_name: motion name
    """
    # Read TXT file in CSV format
    data_df = pd.read_csv(txt_file_path)

    # Get joint names
    robot_joint_names = data_df.columns.tolist()

    # Convert to numpy array
    data = data_df.values  # (n, 14)

    # Separate left and right arm data
    # first 7 columns: left arm
    left_seq = data[:, :7]
    # last 7 columns: right arm
    right_seq = data[:, 7:]

    # Extract motion name from file name
    motion_name = os.path.splitext(os.path.basename(txt_file_path))[0]

    # Default frequency is 50Hz (exported at 50Hz)
    motion_frequence = 50

    return left_seq, right_seq, robot_joint_names, motion_frequence, motion_name


def synchronize_dual_arm_data(data1, data2):
    """
    Synchronize dual-arm data, align two time series to the same time points

    Args:
        data1: left arm data dict
        data2: right arm data dict

    Returns:
        sync_data1: synchronized left arm data
        sync_data2: synchronized right arm data
        common_time: common time series
    """
    time1 = np.array(data1["time"])
    time2 = np.array(data2["time"])

    # Find common time range
    start_time = max(time1[0], time2[0])
    end_time = min(time1[-1], time2[-1])

    # Filter valid time range
    valid_idx1 = (time1 >= start_time) & (time1 <= end_time)
    valid_idx2 = (time2 >= start_time) & (time2 <= end_time)

    time1_valid = time1[valid_idx1]
    time2_valid = time2[valid_idx2]

    # Choose the one with fewer samples as the target time series (reduce interpolation error)
    if len(time1_valid) <= len(time2_valid):
        # Use time1 as the target time series and interpolate data2
        target_time = time1_valid

        # data1 uses valid range directly
        sync_data1 = {
            "time": time1_valid,
            "angle_rad": np.array(data1["angle_rad"])[valid_idx1],
            "velocity_rads": np.array(data1["velocity_rads"])[valid_idx1],
            "current": np.array(data1["current"])[valid_idx1],
        }

        # Interpolate data2
        sync_data2 = {}
        sync_data2["time"] = target_time

        for key in ["angle_rad", "velocity_rads", "current"]:
            data2_array = np.array(data2[key])[valid_idx2]

            # Interpolate each joint separately
            if data2_array.ndim == 2:  # (n_samples, n_joints)
                n_joints = data2_array.shape[1]
                interpolated_data = np.zeros((len(target_time), n_joints))

                for joint_idx in range(n_joints):
                    # Use linear interpolation
                    interp_func = interp1d(
                        time2_valid,
                        data2_array[:, joint_idx],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    interpolated_data[:, joint_idx] = interp_func(target_time)

                sync_data2[key] = interpolated_data
            else:  # 1D array
                interp_func = interp1d(
                    time2_valid, data2_array, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                sync_data2[key] = interp_func(target_time)

    else:
        # Interpolate data1 using time2 as the target time series
        target_time = time2_valid

        # data2 uses valid range directly
        sync_data2 = {
            "time": time2_valid,
            "angle_rad": np.array(data2["angle_rad"])[valid_idx2],
            "velocity_rads": np.array(data2["velocity_rads"])[valid_idx2],
            "current": np.array(data2["current"])[valid_idx2],
        }

        # Interpolate data1
        sync_data1 = {}
        sync_data1["time"] = target_time

        for key in ["angle_rad", "velocity_rads", "current"]:
            data1_array = np.array(data1[key])[valid_idx1]

            # Interpolate each joint separately
            if data1_array.ndim == 2:  # (n_samples, n_joints)
                n_joints = data1_array.shape[1]
                interpolated_data = np.zeros((len(target_time), n_joints))

                for joint_idx in range(n_joints):
                    # Use linear interpolation
                    interp_func = interp1d(
                        time1_valid,
                        data1_array[:, joint_idx],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    interpolated_data[:, joint_idx] = interp_func(target_time)

                sync_data1[key] = interpolated_data
            else:  # 1D array
                interp_func = interp1d(
                    time1_valid, data1_array, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                sync_data1[key] = interp_func(target_time)

    common_time = target_time

    print("Data synchronization completed:")
    print(f"  Original data1 length: {len(data1['time'])}, Original data2 length: {len(data2['time'])}")
    print(f"  Synchronized data length: {len(common_time)}")
    print(f"  Time range: {start_time:.3f}s - {end_time:.3f}s")

    return sync_data1, sync_data2, common_time


def save_motion_csv_files(
    output_dir, motion_name, command_time_list, command_val_list, robot_joint_names, data1, data2
):
    """
    Save 3 CSV files: control.csv, event.csv, state_motor.csv

    Args:
        output_dir: output directory
        motion_name: motion name
        command_time_list: command time list
        command_val_list: command value list (n, 14)
        robot_joint_names: list of joint names
        data1: left arm data dict
        data2: right arm data dict
    """
    # Create motion-specific directory
    motion_dir = os.path.join(output_dir, motion_name)
    os.makedirs(motion_dir, exist_ok=True)

    # Synchronize dual-arm data
    sync_data1, sync_data2, common_time = synchronize_dual_arm_data(data1, data2)

    # 1. Save joint_list.txt (joint name list)
    joint_list_file = os.path.join(motion_dir, "joint_list.txt")
    with open(joint_list_file, "w") as f:
        for joint_name in robot_joint_names:
            f.write(f"{joint_name}\n")

    # 2. Save control.csv (command data)
    control_file = os.path.join(motion_dir, "control.csv")
    with open(control_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions"])

        for i, (time_val, command_val) in enumerate(zip(command_time_list, command_val_list)):
            writer.writerow(["CONTROL", time_val * 1e5, command_val.tolist()])

    # 3.Save event data .csv
    event_file = os.path.join(motion_dir, "event.csv")
    with open(event_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "event"])
        # Add motion start and end events
        if len(command_time_list) > 0:
            writer.writerow(["EVENT", 0, "CONTROL_MODE_ENABLE"])
            writer.writerow(["EVENT", command_time_list[0] * 1e5, "MOTION_START"])
            writer.writerow(["EVENT", command_time_list[-1] * 1e5, "MOTION_END"])

    # 4. Save state_motor.csv (actual state data)
    state_motor_file = os.path.join(motion_dir, "state_motor.csv")
    with open(state_motor_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])

        # Use synchronized data
        for i in range(len(common_time)):
            timestamp = common_time[i]

            # Merge position data (7 left + 7 right)
            positions = np.concatenate([sync_data1["angle_rad"][i], sync_data2["angle_rad"][i]])

            # Merge velocity data
            velocities = np.concatenate([sync_data1["velocity_rads"][i], sync_data2["velocity_rads"][i]])

            # Merge torque data (use current as torque proxy)
            torques = np.concatenate([sync_data1["current"][i], sync_data2["current"][i]])

            writer.writerow(["STATE_MOTOR", timestamp * 1e5, positions.tolist(), velocities.tolist(), torques.tolist()])

    print(f"CSV files saved to directory: {motion_dir}")
    print(f"  - {joint_list_file}")
    print(f"  - {control_file}")
    print(f"  - {event_file}")
    print(f"  - {state_motor_file}")


def collect_motion_data_from_txt(
    collector: HumanoidDualArmCollector, txt_file_path, save_device_dir_path=None, slowdown_factor=1, payload=0
):
    """
    Collect motion data from TXT file and save as CSV

    Args:
        collector: robot collector object
        txt_file_path: TXT file path
        save_device_dir_path: save directory path
        slowdown_factor: slowdown factor
        payload: payload weight

    Returns:
        error_code: error code
        error_joint_list: error joint list
    """
    # load motion data
    left_seq, right_seq, robot_joint_names, motion_frequence, motion_name = load_motion_from_txt(txt_file_path)

    # targeted control frame rate
    target_motion_freq = 50
    left_seq = interpolate_motion(left_seq, motion_frequence, target_motion_freq)
    right_seq = interpolate_motion(right_seq, motion_frequence, target_motion_freq)

    n_steps = left_seq.shape[0]

    collector.home()
    collector.movej_target(np.degrees(left_seq[0]), np.degrees(right_seq[0]))

    command_time_list = []
    command_val_list = []

    # Control frequency fixed at 200Hz
    control_frequence = 200
    assert slowdown_factor in (1, 2, 3, 5, 10), "slowdown_factor only supports 1, 2, 3, 5 or 10"

    steps_per_motion = control_frequence // target_motion_freq * slowdown_factor
    loop_time = 1.0 / control_frequence  # 200Hz period

    start_monotonic = time.monotonic()
    collector.set_starttime(start_monotonic)
    for i in range(n_steps):
        # Interpolate command
        left_rad_start = left_seq[i]
        right_rad_start = right_seq[i]
        if i < n_steps - 1:
            left_rad_end = left_seq[i + 1]
            right_rad_end = right_seq[i + 1]
        else:
            left_rad_end = left_rad_start
            right_rad_end = right_rad_start
        t = time.monotonic() - start_monotonic
        command_time_list.append(t)
        command_val_list.append(np.concatenate([left_rad_start, right_rad_start]))

        for j in range(steps_per_motion):
            # Interpolation calculation
            inter = j / steps_per_motion
            left_rad = (1 - inter) * left_rad_start + inter * left_rad_end
            right_rad = (1 - inter) * right_rad_start + inter * right_rad_end
            left_deg = np.degrees(left_rad)
            right_deg = np.degrees(right_rad)

            # Execute action and timing
            loop_start = time.monotonic()
            collector.arm1.rm_movej_canfd(left_deg.tolist(), True, 0, 0, 0)
            collector.arm2.rm_movej_canfd(right_deg.tolist(), True, 0, 0, 0)
            loop_end = time.monotonic()
            elapsed = loop_end - loop_start
            sleeptime = max(0, loop_time - elapsed)

            error_code, error_joint_list = collector.report_errors()
            if error_code:
                # Clear data cache
                collector.clean_data()
                return error_code, error_joint_list

            if collector.stop_trigger:
                # Clear data cache
                collector.clean_data()
                # Return force stop status
                return 3, []

            if sleeptime > 0:
                time.sleep(sleeptime)
            if (i % 50 == 0) and (j == 0):
                print(
                    f"Motion Step {i}/{n_steps} | "
                    f"Control {j+1}/{steps_per_motion} | "
                    f"Loop cost: {elapsed:.4f}s | "
                    f"sleep: {sleeptime:.4f}s"
                )

    # Get and process data
    data = collector.get_data()
    data1 = data[collector.arm1_port]
    data2 = data[collector.arm2_port]

    # Set save path
    if save_device_dir_path:
        output_dir = save_device_dir_path
    else:
        output_dir = "data/runs_csv"

    # Save CSV files
    save_motion_csv_files(
        output_dir=output_dir,
        motion_name=motion_name,
        command_time_list=command_time_list,
        command_val_list=command_val_list,
        robot_joint_names=robot_joint_names,
        data1=data1,
        data2=data2,
    )

    collector.clean_data()  # Clear data cache

    return 0, []  # Return success status and empty error list


def realman_collector_main(txt_file_path, save_data_path):
    # Single file collection example
    arm1_ip = "192.168.1.188"
    arm2_ip = "192.168.1.188"
    arm1_port = 8080
    arm2_port = 8576
    recv_port1 = 8089
    recv_port2 = 8090

    collector = HumanoidDualArmCollector(arm1_ip, arm2_ip, arm1_port, arm2_port, recv_port1, recv_port2)
    collector.setup_callbacks()

    try:
        error_code, error_joint_list = collect_motion_data_from_txt(
            collector, txt_file_path, save_device_dir_path=save_data_path, slowdown_factor=1, payload=0
        )

        if error_code != 0:
            print(f"Error occurred during data collection (code: {error_code}), error joints: {error_joint_list}")
        else:
            print("Data collection successful")

    finally:
        collector.close()
