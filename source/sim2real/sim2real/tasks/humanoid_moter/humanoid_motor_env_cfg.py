# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from sim2real_assets import H1_2_CFG_WITHOUT_HAND, H1_2_CFG_WITH_HAND, H1_2_CFG_WITH_HAND_FIX, H1_2_WITH_HAND_FIX_URDF_PATH

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


from .motions.joint_names import ROBOT_JOINT_NAME_DICT_URDF

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

ROBOT_DICT = {
    "h1_2_without_hand": {"model": H1_2_CFG_WITHOUT_HAND, "motion_dir": "edited_27dof"},
    "h1_2_with_hand": {"model": H1_2_CFG_WITH_HAND, "motion_dir": "edited_51dof"},
    "h1_2_with_hand_fix": {"model": H1_2_CFG_WITH_HAND_FIX, "motion_dir": "edited_27dof", "urdf_path": H1_2_WITH_HAND_FIX_URDF_PATH},
    }



@configclass
class HumanoidMotorEnvCfg(DirectRLEnvCfg):

    # robot_name: str = "h1_2_without_hand"
    # robot_name: str = "h1_2_with_hand"
    robot_name: str = "h1_2_with_hand_fix"

    if "urdf_path" in ROBOT_DICT[robot_name]:
        if_torque_input = True
        urdf_model_path = ROBOT_DICT[robot_name]["urdf_path"]
        package_dirs = os.path.dirname(urdf_model_path)

        urdf_joint_name = ROBOT_JOINT_NAME_DICT_URDF[f"{robot_name}_joints"]

    # env
    episode_length_s = 1.0
    decimation = 4

    mode: None | str = None   # train 或 play，会在 train.py 或 play.py 中指定

    # spaces
    observation_space = 25
    action_space = 1
    state_space = 0
    num_amp_observations = 5     # history length
    amp_observation_space = 5    # dim of a single obs

    early_termination = True
    termination_height = 0.8

    motion_file: str = MISSING  # type: ignore
    reference_body = "torso_link"
    reset_strategy = "random"  # default, random, random-start
    
    """
    Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**24,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ROBOT_DICT[robot_name]["model"].replace(prim_path="/World/envs/env_.*/Robot")   # type: ignore

    motion_dir = MOTIONS_DIR
    motion_joint = None
    motion_file = os.path.join(motion_dir, f"motion_perjoint_all/{ROBOT_DICT[robot_name]['motion_dir']}")

@configclass
class HumanoidMotorPerJointEnvCfg(HumanoidMotorEnvCfg):
    nothing = None