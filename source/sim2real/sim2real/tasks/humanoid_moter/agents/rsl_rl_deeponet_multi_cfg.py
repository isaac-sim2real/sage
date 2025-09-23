# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from typing import Literal, List
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

@configclass
class DeepONetActorCriticCfg:
    class_name: str = "ActorCritic"

    """Configuration for DeepONet Actor-Critic network"""
    # Branch network configuration
    branch_input_dims: List[int] = [12, 6, 3]  # Input dimensions for different resolutions
    branch_hidden_dim: int = 128  # Hidden dimension for branch networks
    
    # Trunk network configuration
    trunk_input_dim: int = 54  # Input dimension for trunk network
    trunk_hidden_dims: list[int] = [128, 128]  # Hidden dimensions for trunk network
    
    # Output configuration
    action_dim: int = 10  # Dimension of action space
    activation: str = "elu"  # Activation function
    
    # Critic network configuration
    critic_hidden_dims: list[int] = [128, 128]  # Hidden dimensions for critic network

@configclass
class HumanoidDeepONetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for DeepONet PPO runner"""
    num_steps_per_env = 32
    max_iterations = 120
    save_interval = 50
    experiment_name = "humanoid_deeponet_direct"
    empirical_normalization = True
    
    # DeepONet policy configuration
    policy = DeepONetActorCriticCfg()
    
    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
    
    # Logger configuration
    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    wandb_project: str = "humanoid-deeponet-direct"