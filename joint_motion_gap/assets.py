# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import List

import carb


class RobotAssetConfig:
    """Configuration class for robot assets."""

    def __init__(self, name: str, usd_path: str, offset: List[float], prim_path: str = "/World/Robot", **kwargs):
        """
        Initialize robot asset configuration.

        Args:
            name: Robot name identifier
            usd_path: Path to the robot USD file (can be relative to repo root)
            offset: Translation offset [x, y, z] for robot placement
            prim_path: USD prim path for the robot (default: "/World/Robot")
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.usd_path = usd_path
        self.offset = offset
        self.prim_path = prim_path

        # Store additional configuration parameters
        self.config = kwargs

    def get_config_value(self, key: str, default=None):
        """Get additional configuration value by key."""
        return self.config.get(key, default)


# Isaac Sim Nucleus Asset Directory
NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"

# Robot asset configurations
ROBOT_ASSETS = {
    "h1_2": RobotAssetConfig(
        name="h1_2",
        usd_path="assets/h1_2/h1_2.usd",
        offset=[0.0, 0.0, 1.1],
        default_kp=50.0,
        default_kd=1.0,
    ),
    "g1": RobotAssetConfig(
        name="g1",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        offset=[0.0, 0.0, 0.82],
        default_kp=50.0,
        default_kd=1.0,
    ),
}


def list_available_robots() -> List[str]:
    """Get list of available robot names."""
    return list(ROBOT_ASSETS.keys())


def get_robot_config(robot_name: str) -> RobotAssetConfig:
    """Get robot configuration by name."""
    robot_name = robot_name.lower()
    if robot_name not in ROBOT_ASSETS:
        available_robots = list_available_robots()
        raise ValueError(f"Unsupported robot: {robot_name}. Available robots: {available_robots}")

    return ROBOT_ASSETS[robot_name]
