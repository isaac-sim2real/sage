# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Tuple, List
import torch
import gym.spaces
import numpy as np

from sim2real.tasks.humanoid_moter.humanoid_motor_env import HumanoidMotorEnv, HumanoidMotorPerJointEnvCfg
from sim2real.tasks.humanoid_moter.humanoid_deeponet_env_cfg import HumanoidDeepONetEnvCfg


class HumanoidDeepONetEnv(HumanoidMotorEnv):
    """Environment for DeepONet-based humanoid control"""
    cfg: HumanoidDeepONetEnvCfg

    def __init__(self, cfg: HumanoidDeepONetEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize observation spaces for branch and trunk networks
        self.branch_observation_spaces = [
            gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(dim,)
            ) for dim in self.cfg.branch_input_dims
        ]
        
        # Update trunk observation space to account for sequence length
        self.trunk_observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.cfg.trunk_input_dim * self.cfg.sequence_length,)
        )

    def _get_observations(self) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """Get observations for both branch and trunk networks"""
        # Get base observations
        base_obs = super()._get_observations()
        
        # Get motion features for different resolutions
        motion_features = self.amp_observation_buffer.clone()
        
        # Split motion features into different resolutions
        branch_obs = []
        start_idx = 0
        for dim in self.cfg.branch_input_dims:
            end_idx = start_idx + dim
            branch_obs.append(motion_features[:, start_idx:end_idx])
            start_idx = end_idx
        
        # Get robot state features for the sequence
        robot_state_features = torch.cat([
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            # self.robot.data.body_pos_w[:, self.ref_body_index],
            # self.robot.data.body_quat_w[:, self.ref_body_index],
        ], dim=-1)
        
        # Create sequence buffer for trunk network if it doesn't exist
        if not hasattr(self, 'trunk_sequence_buffer'):
            self.trunk_sequence_buffer = torch.zeros(
                (self.num_envs, self.cfg.sequence_length, self.cfg.trunk_input_dim),
                device=self.device
            )
        
        # Update sequence buffer by shifting and adding new state
        self.trunk_sequence_buffer = torch.roll(self.trunk_sequence_buffer, shifts=-1, dims=1)
        self.trunk_sequence_buffer[:, -1] = robot_state_features
        
        # Flatten the sequence for trunk network input
        trunk_obs = self.trunk_sequence_buffer.reshape(self.num_envs, -1)
        
        return {
            "branch": branch_obs,
            "trunk": trunk_obs,
            "policy": base_obs["policy"]  # Keep original policy observation for compatibility
        }

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards for the environment"""
        # Use the same reward structure as the base environment
        return super()._get_rewards()

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get done flags for the environment"""
        # Use the same done conditions as the base environment
        return super()._get_dones()