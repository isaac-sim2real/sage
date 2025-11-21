# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Robot asset registry and lookup utilities."""

from isaaclab.assets import ArticulationCfg

from .robots.realman import WR75S_CFG
from .robots.unitree import G1_CFG, H1_2_CFG

##
# Robot Registry
##

ROBOT_CFGS = {
    "h1_2": H1_2_CFG,
    "g1": G1_CFG,
    "wr75s": WR75S_CFG,
}
"""Mapping of robot names to ArticulationCfg instances."""


def get_robot_cfg(robot_name: str) -> ArticulationCfg:
    """Get ArticulationCfg for a robot.

    Args:
        robot_name: Robot name (h1_2, g1, wr75s)

    Returns:
        ArticulationCfg: ArticulationCfg instance

    Raises:
        ValueError: If robot_name is not recognized
    """
    robot_name = robot_name.lower()
    if robot_name not in ROBOT_CFGS:
        raise ValueError(f"Unknown robot: {robot_name}. Available: {list(ROBOT_CFGS.keys())}")
    return ROBOT_CFGS[robot_name]
