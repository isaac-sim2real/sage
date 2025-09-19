# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

__version__ = "0.1.0"
__author__ = "JointMotionGap Team"

# Import analysis module (no external dependencies)
from .analysis import RobotDataComparator


# Simulation module imports are deferred to avoid omni dependencies when not needed
def _get_simulation_module():
    """Lazy import of simulation module to avoid omni dependencies."""
    from .simulation import JointMotionBenchmark, get_motion_files, get_motion_name, log_message

    return JointMotionBenchmark, get_motion_files, get_motion_name, log_message


# For backward compatibility, provide access to simulation components when explicitly accessed
def __getattr__(name):
    if name in ["JointMotionBenchmark", "get_motion_files", "get_motion_name", "log_message"]:
        JointMotionBenchmark, get_motion_files, get_motion_name, log_message = _get_simulation_module()
        globals().update(
            {
                "JointMotionBenchmark": JointMotionBenchmark,
                "get_motion_files": get_motion_files,
                "get_motion_name": get_motion_name,
                "log_message": log_message,
            }
        )
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["RobotDataComparator", "JointMotionBenchmark", "get_motion_files", "get_motion_name", "log_message"]
