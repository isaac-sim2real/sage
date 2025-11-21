# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Configuration for Unitree robots."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.unitree import G1_29DOF_CFG, H1_CFG

##
# Configuration
##

H1_2_CFG = H1_CFG.copy()
H1_2_CFG.spawn.usd_path = "assets/h1_2/h1_2.usd"
H1_2_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.05),
    joint_pos={
        ".*_hip_yaw_joint": 0.0,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_pitch_joint": -0.28,  # -16 degrees
        ".*_knee_joint": 0.79,  # 45 degrees
        ".*_ankle_pitch_joint": -0.52,  # -30 degrees
        "torso_joint": 0.0,
        ".*_shoulder_pitch_joint": 0.28,
        ".*_shoulder_roll_joint": 0.0,
        ".*_shoulder_yaw_joint": 0.0,
        ".*_elbow_joint": 0.52,
    },
    joint_vel={".*": 0.0},
)
H1_2_CFG.actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "torso_joint",
        ],
        effort_limit_sim=300,
        stiffness={
            ".*_hip_yaw_joint": 150.0,
            ".*_hip_roll_joint": 150.0,
            ".*_hip_pitch_joint": 200.0,
            ".*_knee_joint": 200.0,
            "torso_joint": 200.0,
        },
        damping={
            ".*_hip_yaw_joint": 5.0,
            ".*_hip_roll_joint": 5.0,
            ".*_hip_pitch_joint": 5.0,
            ".*_knee_joint": 5.0,
            "torso_joint": 5.0,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=100,
        stiffness={".*_ankle_pitch_joint": 20.0, ".*_ankle_roll_joint": 20.0},
        damping={".*_ankle_pitch_joint": 4.0, ".*_ankle_roll_joint": 4.0},
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim=300,
        stiffness={
            ".*_shoulder_pitch_joint": 40.0,
            ".*_shoulder_roll_joint": 40.0,
            ".*_shoulder_yaw_joint": 40.0,
            ".*_elbow_joint": 40.0,
            ".*_wrist_roll_joint": 40.0,
            ".*_wrist_pitch_joint": 40.0,
            ".*_wrist_yaw_joint": 40.0,
        },
        damping={
            ".*_shoulder_pitch_joint": 10.0,
            ".*_shoulder_roll_joint": 10.0,
            ".*_shoulder_yaw_joint": 10.0,
            ".*_elbow_joint": 10.0,
            ".*_wrist_roll_joint": 10.0,
            ".*_wrist_pitch_joint": 10.0,
            ".*_wrist_yaw_joint": 10.0,
        },
    ),
    "hands": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_index_.*",
            ".*_middle_.*",
            ".*_pinky_.*",
            ".*_ring_.*",
            ".*_thumb_.*",
        ],
        effort_limit_sim=30.0,
        velocity_limit_sim=10.0,
        stiffness=20,
        damping=2,
        armature=0.001,
    ),
}
"""Configuration for Unitree H1_2 humanoid robot."""


G1_CFG = G1_29DOF_CFG.copy()
"""Configuration for Unitree G1 humanoid robot."""
