# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Any, Dict, List, Optional, Tuple

# Isaac Sim modules - will be imported after SimulationApp initialization
# This will be set as a module-level variable by the calling script
carb = None


# ============================================================================
# Robot Configuration Data
# ============================================================================
# To add a new robot, simply add an entry to this dictionary.
#
# Fields:
#   - usd_path: Path to USD file (local or NVIDIA Nucleus with placeholders)
#       Placeholders: {NUCLEUS_ASSET_ROOT_DIR}, {NVIDIA_NUCLEUS_DIR},
#                     {ISAAC_NUCLEUS_DIR}, {ISAACLAB_NUCLEUS_DIR}
#   - offset: (x, y, z) translation offset for robot placement
#   - prim_path: (optional) USD prim path, defaults to "/World/Robot"
#   - Additional fields like default_kp, default_kd are passed as kwargs
# ============================================================================

ROBOT_CONFIGS = {
    "h1_2": {
        "usd_path": "assets/h1_2/h1_2.usd",
        "offset": (0.0, 0.0, 1.1),
        "default_kp": 50.0,
        "default_kd": 1.0,
    },
    "g1": {
        "usd_path": "{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        "offset": (0.0, 0.0, 0.82),
        "default_kp": 50.0,
        "default_kd": 1.0,
    },
}


class RobotAssetConfig:
    """Configuration class for robot assets."""

    def __init__(
        self,
        name: str,
        usd_path: str,
        offset: Tuple[float, float, float],
        prim_path: str = "/World/Robot",
        **kwargs,
    ):
        """Initialize robot asset configuration."""
        self._name = name
        self._usd_path = usd_path
        self._offset = offset
        self._prim_path = prim_path
        self._config = kwargs

    @property
    def name(self) -> str:
        """Robot name identifier."""
        return self._name

    @property
    def usd_path(self) -> str:
        """Path to the robot USD file."""
        return self._usd_path

    @property
    def offset(self) -> Tuple[float, float, float]:
        """Translation offset (x, y, z) for robot placement."""
        return self._offset

    @property
    def prim_path(self) -> str:
        """USD prim path for the robot."""
        return self._prim_path

    @property
    def config(self) -> Dict[str, Any]:
        """Additional configuration parameters."""
        return self._config

    def get_config_value(self, key: str, default=None):
        """Get additional configuration value by key."""
        return self._config.get(key, default)


class RobotAssetRegistry:
    """Registry for managing robot asset configurations."""

    def __init__(self, config_dict: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize the registry with lazy loading."""
        self._config_dict = config_dict if config_dict is not None else ROBOT_CONFIGS
        self._assets: Optional[Dict[str, RobotAssetConfig]] = None

    def _get_nucleus_asset_root_dir(self) -> str:
        """Get the nucleus asset root directory."""
        if carb is None:
            raise RuntimeError("carb module not initialized.")
        return carb.settings.get_settings().get("/persistent/isaac/asset_root/default")

    def _resolve_path_placeholders(self, path: str) -> str:
        """Resolve placeholders in USD paths."""
        if "{NUCLEUS_ASSET_ROOT_DIR}" in path:
            nucleus_asset_root_dir = f"{self._get_nucleus_asset_root_dir()}"
            path = path.replace("{NUCLEUS_ASSET_ROOT_DIR}", nucleus_asset_root_dir)
        elif "{NVIDIA_NUCLEUS_DIR}" in path:
            nvidia_nucleus_dir = f"{self._get_nucleus_asset_root_dir()}/NVIDIA"
            path = path.replace("{NVIDIA_NUCLEUS_DIR}", nvidia_nucleus_dir)
        elif "{ISAAC_NUCLEUS_DIR}" in path:
            isaac_nucleus_dir = f"{self._get_nucleus_asset_root_dir()}/Isaac"
            path = path.replace("{ISAAC_NUCLEUS_DIR}", isaac_nucleus_dir)
        elif "{ISAACLAB_NUCLEUS_DIR}" in path:
            isaaclab_nucleus_dir = f"{self._get_nucleus_asset_root_dir()}/IsaacLab"
            path = path.replace("{ISAACLAB_NUCLEUS_DIR}", isaaclab_nucleus_dir)
        return path

    def _create_robot_config(self, robot_name: str, config_data: Dict[str, Any]) -> RobotAssetConfig:
        """Create a RobotAssetConfig from dictionary data."""
        # Extract required fields
        usd_path = self._resolve_path_placeholders(config_data["usd_path"])
        offset = config_data["offset"]
        prim_path = config_data.get("prim_path", "/World/Robot")

        # Get additional kwargs (everything except the known fields)
        kwargs = {k: v for k, v in config_data.items() if k not in ["usd_path", "offset", "prim_path"]}

        return RobotAssetConfig(
            name=robot_name,
            usd_path=usd_path,
            offset=offset,
            prim_path=prim_path,
            **kwargs,
        )

    def _initialize_assets(self) -> Dict[str, RobotAssetConfig]:
        """Initialize and return the robot assets dictionary."""
        assets = {}
        for robot_name, config_data in self._config_dict.items():
            assets[robot_name] = self._create_robot_config(robot_name, config_data)
        return assets

    @property
    def assets(self) -> Dict[str, RobotAssetConfig]:
        """Get the robot assets dictionary, initializing if needed."""
        if self._assets is None:
            self._assets = self._initialize_assets()
        return self._assets

    def available_robots(self) -> List[str]:
        """Get list of available robot names."""
        return list(self.assets.keys())

    def get_robot_config(self, robot_name: str) -> RobotAssetConfig:
        """Get robot configuration by name."""
        robot_name = robot_name.lower()
        if robot_name not in self.assets:
            available_robots = self.available_robots()
            raise ValueError(f"Unsupported robot: {robot_name}. Available robots: {available_robots}")
        return self.assets[robot_name]


# Global registry instance
_registry = RobotAssetRegistry()


def available_robots() -> List[str]:
    """Get list of available robot names."""
    return _registry.available_robots()


def get_robot_config(robot_name: str) -> RobotAssetConfig:
    """Get robot configuration by name."""
    return _registry.get_robot_config(robot_name)
