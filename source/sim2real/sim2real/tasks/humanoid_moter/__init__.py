# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Humanoid-Motor-Direct-v0",
    entry_point=f"{__name__}.humanoid_motor_env:HumanoidMotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_motor_env_cfg:HumanoidMotorPerJointEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_motor_amp_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
    },
)
