 # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from sim2real.tasks.humanoid_moter.humanoid_motor_env_cfg import HumanoidMotorEnvCfg
from isaaclab.utils import configclass


@configclass
class HumanoidDeepONetEnvCfg(HumanoidMotorEnvCfg):
    """Configuration for DeepONet-based humanoid environment"""
    # DeepONet specific configurations
    branch_input_dims: list[int] = [12, 6, 3]  # Dimensions for different resolutions (total 21)
    trunk_input_dim: int = 54   # Dimension of trunk network input (joint_pos + joint_vel)
    sequence_length: int = 7    # Number of frames to use for sequence input
    
    # Override observation space for DeepONet
    observation_space = 21  # Total observation space size (sum of branch dimensions)
    action_space = 1  # Keep the same action space as motor task