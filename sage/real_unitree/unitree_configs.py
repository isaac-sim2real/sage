# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotConfig:
    """Robot configuration container"""

    name: str
    hw_dof: int
    num_actions: int
    num_upper_body_joints: int
    leg_joints: list
    upper_body_joints: list
    joint_names: list

    # PD gains
    p_gains: np.ndarray
    d_gains: np.ndarray

    # Joint limits
    joint_limit_lo: np.ndarray
    joint_limit_hi: np.ndarray

    # Default positions
    default_dof_pos: np.ndarray

    # Observation scales
    scale_lin_vel: float
    scale_ang_vel: float
    scale_orn: float
    scale_dof_pos: float
    scale_dof_vel: float
    scale_action: float

    # Control frequency
    control_frequency: int

    # Motion file path
    motion_file_path: str


# G1 Robot Configuration
G1_CONFIG = RobotConfig(
    name="G1",
    hw_dof=29,
    num_actions=12,
    num_upper_body_joints=19,
    leg_joints=list(range(12)),
    upper_body_joints=list(range(12, 29)),
    joint_names=[
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "neck_yaw_joint",
        "neck_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    # PD gains for G1
    p_gains=np.array(
        [
            80.0,
            80.0,
            80.0,
            160.0,
            20.0,
            20.0,
            80.0,
            80.0,
            80.0,
            160.0,
            20.0,
            20.0,
            400.0,
            400.0,
            400.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
            40.0,
        ]
    ),
    d_gains=np.array(
        [
            2,
            2,
            2,
            4,
            0.5,
            0.5,
            2,
            2,
            2,
            4,
            0.5,
            0.5,
            5,
            5,
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    ),
    # Joint limits for G1
    joint_limit_lo=np.array(
        [
            -2.5307,
            -0.5236,
            -2.7576,
            -0.087267,
            -np.inf,
            -np.inf,
            -2.5307,
            -2.9671,
            -2.7576,
            -0.087267,
            -np.inf,
            -np.inf,
            -2.618,
            -0.52,
            -0.52,
            -3.0892,
            -1.5882,
            -2.618,
            -1.0472,
            -1.972222054,
            -1.614429558,
            -1.614429558,
            -3.0892,
            -2.2515,
            -2.618,
            -1.0472,
            -1.972222054,
            -1.614429558,
            -1.614429558,
        ]
    ),
    joint_limit_hi=np.array(
        [
            2.8798,
            2.9671,
            2.7576,
            2.8798,
            np.inf,
            np.inf,
            2.8798,
            0.5236,
            2.7576,
            2.8798,
            np.inf,
            np.inf,
            2.618,
            0.52,
            0.52,
            2.6704,
            2.2515,
            2.618,
            2.0944,
            1.972222054,
            1.614429558,
            1.614429558,
            2.6704,
            1.5882,
            2.618,
            2.0944,
            1.972222054,
            1.614429558,
            1.614429558,
        ]
    ),
    # Default positions for G1
    default_dof_pos=np.array(
        [
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.2,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ),
    scale_lin_vel=2.0,
    scale_ang_vel=0.25,
    scale_orn=1.0,
    scale_dof_pos=1.0,
    scale_dof_vel=0.05,
    scale_action=0.25,
    control_frequency=50,
    motion_file_path="g1_amass_test.pkl",
)

# H1-2 Robot Configuration
H12_CONFIG = RobotConfig(
    name="H1-2",
    hw_dof=27,
    num_actions=12,
    num_upper_body_joints=15,
    leg_joints=list(range(12)),
    upper_body_joints=list(range(12, 27)),
    joint_names=[
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    # PD gains for H1-2
    p_gains=np.array(
        [
            300.0,
            200.0,
            200.0,
            300.0,
            40.0,
            40.0,
            300.0,
            200.0,
            200.0,
            300.0,
            40.0,
            40.0,
            300.0,
            200.0,
            200.0,
            200.0,
            100.0,
            20.0,
            20.0,
            20.0,
            200.0,
            200.0,
            200.0,
            100.0,
            20.0,
            20.0,
            20.0,
        ]
    ),
    d_gains=np.array(
        [
            3.0,
            2.5,
            2.5,
            4.0,
            2.0,
            2.0,
            3.0,
            2.5,
            2.5,
            4.0,
            2.0,
            2.0,
            5.0,
            4.0,
            4.0,
            4.0,
            1.0,
            0.5,
            0.5,
            0.5,
            4.0,
            4.0,
            4.0,
            1.0,
            0.5,
            0.5,
            0.5,
        ]
    ),
    # Joint limits for H1-2
    joint_limit_lo=np.array(
        [
            -0.43,
            -3.14,
            -0.43,
            -0.26,
            -np.inf,
            -np.inf,
            -0.43,
            -3.14,
            -3.14,
            -0.24,
            -np.inf,
            -np.inf,
            -2.618,
            -3.0892,
            -1.5882,
            -2.618,
            -1.0472,
            -1.972222054,
            -1.614429558,
            -1.614429558,
            -3.0892,
            -2.2515,
            -2.618,
            -1.0472,
            -1.972222054,
            -1.614429558,
            -1.614429558,
        ]
    ),
    joint_limit_hi=np.array(
        [
            0.43,
            2.5,
            3.14,
            2.05,
            np.inf,
            np.inf,
            0.43,
            2.5,
            0.43,
            2.0,
            np.inf,
            np.inf,
            2.618,
            2.6704,
            2.2515,
            2.618,
            2.0944,
            1.972222054,
            1.614429558,
            1.614429558,
            2.6704,
            1.5882,
            2.618,
            2.0944,
            1.972222054,
            1.614429558,
            1.614429558,
        ]
    ),
    # Default positions for H1-2
    default_dof_pos=np.array(
        [
            0.0,
            -0.16,
            0.0,
            0.36,
            -0.2,
            0.0,
            0.0,
            -0.16,
            0.0,
            0.36,
            -0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ),
    scale_lin_vel=2.0,
    scale_ang_vel=0.25,
    scale_orn=1.0,
    scale_dof_pos=1.0,
    scale_dof_vel=0.05,
    scale_action=0.25,
    control_frequency=50,
    motion_file_path="h12_amass_test.pkl",
)

# Robot configuration registry
ROBOT_CONFIGS = {"g1": G1_CONFIG, "h12": H12_CONFIG}


def get_robot_config(robot_name: str) -> RobotConfig:
    """
    Get robot configuration by name
    Args:
        robot_name: Name of the robot (g1 or h12)
    Returns:
        RobotConfig: Robot configuration object
    Raises:
        ValueError: If robot name is not supported
    """
    if robot_name not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot: {robot_name}. Available robots: {list(ROBOT_CONFIGS.keys())}")

    return ROBOT_CONFIGS[robot_name]


def list_available_robots() -> list:
    """List all available robot configurations"""
    return list(ROBOT_CONFIGS.keys())


def print_robot_info(config: RobotConfig):
    """Print robot configuration information"""
    print(f"Robot: {config.name}")
    print(f"Hardware DOF: {config.hw_dof}")
    print(f"Leg joints: {len(config.leg_joints)} joints")
    print(f"Upper body joints: {len(config.upper_body_joints)} joints")
    print(f"Control frequency: {config.control_frequency}Hz")
    print(f"Motion file: {config.motion_file_path}")
