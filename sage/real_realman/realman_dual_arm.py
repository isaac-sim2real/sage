# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import time

import numpy as np
from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_realtime_arm_state_callback_ptr,
    rm_realtime_push_config_t,
    rm_thread_mode_e,
    rm_udp_custom_config_t,
)


class HumanoidDualArmCollector:
    """
    - Use port to distinguish left and right arms (not ip)
    - __init__ parameters need to pass two ips, two connection ports, and two receive ports respectively
    """

    def __init__(self, arm1_ip, arm2_ip, arm1_port, arm2_port, recv_port1, recv_port2):
        # Save port info
        self.arm1_port = arm1_port
        self.arm2_port = arm2_port

        # Save two sets of data, key is port
        self.arm_data = {
            self.arm1_port: {
                "time": [],
                "angle_rad": [],
                "velocity_rads": [],
                "current": [],
                "temperature": [],
            },
            self.arm2_port: {
                "time": [],
                "angle_rad": [],
                "velocity_rads": [],
                "current": [],
                "temperature": [],
            },
        }
        self.start_monotonic = None
        self.stop_trigger = False

        # Initialize two robotic arms
        self.arm1 = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.arm1_handle = self.arm1.rm_create_robot_arm(arm1_ip, arm1_port)
        self.arm1.rm_set_arm_run_mode(1)
        print(self.arm1.rm_set_self_collision_enable(True))

        self.arm2 = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.arm2_handle = self.arm2.rm_create_robot_arm(arm2_ip, arm2_port)
        self.arm2.rm_set_arm_run_mode(1)
        print(self.arm2.rm_set_self_collision_enable(True))

        # Save ip info (can also add to self as member variable if needed)
        self.arm1_ip = arm1_ip
        self.arm2_ip = arm2_ip

        # Save receive ports
        self.recv_port1 = recv_port1
        self.recv_port2 = recv_port2

        self.arm_err = {
            self.arm1_port: {
                "error_code": [0, 0, 0, 0, 0, 0, 0],
                "en_flag": [True, True, True, True, True, True, True],
            },
            self.arm2_port: {
                "error_code": [0, 0, 0, 0, 0, 0, 0],
                "en_flag": [True, True, True, True, True, True, True],
            },
        }

    def arm_state_callback(self, data):
        if self.start_monotonic is None:
            return

        port = data.arm_port
        if port in self.arm_data:
            t = time.monotonic() - self.start_monotonic
            joint_angle_deg = [data.joint_status.joint_position[i] for i in range(7)]
            joint_speed_deg = [data.joint_status.joint_speed[i] for i in range(7)]
            joint_current = [data.joint_status.joint_current[i] for i in range(7)]
            joint_temperature = [data.joint_status.joint_temperature[i] for i in range(7)]

            self.arm_err[port]["error_code"] = [data.joint_status.joint_err_code[i] for i in range(7)]
            self.arm_err[port]["en_flag"] = [data.joint_status.joint_en_flag[i] for i in range(7)]

            joint_angle_rad = np.radians(joint_angle_deg)
            joint_speed_rad = np.radians(joint_speed_deg)

            self.arm_data[port]["time"].append(t)
            self.arm_data[port]["angle_rad"].append(joint_angle_rad)
            self.arm_data[port]["velocity_rads"].append(joint_speed_rad)
            self.arm_data[port]["current"].append(joint_current)
            self.arm_data[port]["temperature"].append(joint_temperature)

    def setup_callbacks(self):
        # arm1
        self.custom1 = rm_udp_custom_config_t()
        self.custom1.joint_speed = 1
        self.custom1.lift_state = 0
        self.custom1.expand_state = 0
        config1 = rm_realtime_push_config_t(1, True, self.recv_port1, 0, self.arm1_ip, self.custom1)
        print(self.arm1.rm_set_realtime_push(config1))
        print(self.arm1.rm_get_realtime_push())

        # arm2
        self.custom2 = rm_udp_custom_config_t()
        self.custom2.joint_speed = 1
        self.custom2.lift_state = 0
        self.custom2.expand_state = 0
        config2 = rm_realtime_push_config_t(1, True, self.recv_port2, 0, self.arm2_ip, self.custom2)
        print(self.arm2.rm_set_realtime_push(config2))
        print(self.arm2.rm_get_realtime_push())

        # Register callback
        self._callback_ptr = rm_realtime_arm_state_callback_ptr(self.arm_state_callback)
        self.arm1.rm_realtime_arm_state_call_back(self._callback_ptr)
        self.arm2.rm_realtime_arm_state_call_back(self._callback_ptr)

    def get_data(self):
        """
        Return collected data with port as key, structure:
        {
            port1: {
                'time': np.ndarray,
                'angle_rad': np.ndarray,
                'velocity_rads': np.ndarray,
                'current': np.ndarray,
                'temperature': np.ndarray
            },
            ...
        }
        """
        self.start_monotonic = None
        res = {}
        for port in [self.arm1_port, self.arm2_port]:
            entry = self.arm_data[port]
            # Use shortest length to ensure data alignment
            min_len = (
                min(
                    len(entry["time"]),
                    len(entry["angle_rad"]),
                    len(entry["velocity_rads"]),
                    len(entry["current"]),
                    len(entry["temperature"]),
                )
                - 1
            )
            res[port] = {
                "time": np.array(entry["time"][:min_len]),
                "angle_rad": np.array(entry["angle_rad"][:min_len]),
                "velocity_rads": np.array(entry["velocity_rads"][:min_len]),
                "current": np.array(entry["current"][:min_len]),
                "temperature": np.array(entry["temperature"][:min_len]),
            }
        return res

    def home(self):
        print(self.arm1.rm_movej([0, 0, 0, 0, 0, 0, 0], 20, 0, 0, 1))  # degree
        print(self.arm2.rm_movej([0, 0, 0, 0, 0, 0, 0], 20, 0, 0, 1))

    def movej_target(self, left_arm_target, right_arm_target):
        print(self.arm1.rm_movej(left_arm_target.tolist(), 20, 0, 0, 1))  # degree
        print(self.arm2.rm_movej(right_arm_target.tolist(), 20, 0, 0, 1))

    def close(self):
        self.arm1.rm_delete_robot_arm()
        self.arm2.rm_delete_robot_arm()

    def set_starttime(self, start_monotonic):
        self.start_monotonic = start_monotonic

    def report_errors(self):
        """
        Check both arms, prioritize dropped enable (error_code=1), otherwise check general joint errors (error_code=2).
        Return (error_code, joints_list), joints_list is 0~6(arm1), 7~13(arm2).
        """
        all_enflags = []
        all_errcodes = []

        for port in [self.arm1_port, self.arm2_port]:
            all_enflags.extend(self.arm_err[port]["en_flag"])
            all_errcodes.extend(self.arm_err[port]["error_code"])

        # Check dropped enable
        dropped_joints = [i for i, en in enumerate(all_enflags) if not en]
        if dropped_joints:
            return 1, dropped_joints  # error_code=1, means dropped enable

        # Check other errors
        error_joints = [i for i, code in enumerate(all_errcodes) if code != 0]
        if error_joints:
            return 2, error_joints  # error_code=2, joint error

        return 0, []  # no error

    def clean_data(self):
        """
        Clear data and reset state.
        """
        self.arm_data = {
            self.arm1_port: {
                "time": [],
                "angle_rad": [],
                "velocity_rads": [],
                "current": [],
                "temperature": [],
            },
            self.arm2_port: {
                "time": [],
                "angle_rad": [],
                "velocity_rads": [],
                "current": [],
                "temperature": [],
            },
        }
        self.start_monotonic = None
        self.arm_err = {
            self.arm1_port: {
                "error_code": [0, 0, 0, 0, 0, 0, 0],
                "en_flag": [True, True, True, True, True, True, True],
            },
            self.arm2_port: {
                "error_code": [0, 0, 0, 0, 0, 0, 0],
                "en_flag": [True, True, True, True, True, True, True],
            },
        }
