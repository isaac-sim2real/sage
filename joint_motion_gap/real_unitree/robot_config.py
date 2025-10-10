#!/usr/bin/env python3
"""
Robot Configuration System
Supports both G1 and H1-2 robots with different parameters
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RobotConfig:
    """Robot configuration container"""
    name: str
    hw_dof: int
    num_actions: int
    num_upper_body_joints: int
    leg_joints: list
    upper_body_joints: list
    
    # PD gains
    p_gains: np.ndarray
    d_gains: np.ndarray
    
    # Joint limits
    joint_limit_lo: np.ndarray
    joint_limit_hi: np.ndarray
    
    # Default positions
    default_dof_pos: np.ndarray
    
    # Observation scales
    scale_lin_vel: float
    scale_ang_vel: float
    scale_orn: float
    scale_dof_pos: float
    scale_dof_vel: float
    scale_action: float
    
    # Control frequency
    control_frequency: int
    
    # Motion file path
    motion_file_path: str

# G1 Robot Configuration
G1_CONFIG = RobotConfig(
    name="G1",
    hw_dof=29,
    num_actions=12,
    num_upper_body_joints=19,
    leg_joints=list(range(12)),
    upper_body_joints=list(range(12, 29)),
    
    # PD gains for G1
    p_gains = np.array([80.,80.,80.,160.,20.,20.,80.,80.,80.,160.,20.,20.,400.,400.,400.,
                        40.,40.,40.,40.,40.,40.,40.,
                        40.,40.,40.,40.,40.,40.,40.,]),
    d_gains = np.array([2,2,2,4,0.5,0.5,2,2,2,4,0.5,0.5,5,5,5,
                        1,1,1,1,1,1,1,
                        1,1,1,1,1,1,1,]),
    # Joint limits for G1
    joint_limit_lo=np.array([-2.5307, -0.5236, -2.7576, -0.087267, -np.inf, -np.inf, -2.5307,-2.9671,-2.7576,-0.087267,-np.inf,-np.inf,-2.618,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558]),
    joint_limit_hi=np.array([2.8798, 2.9671, 2.7576, 2.8798, np.inf, np.inf, 2.8798, 0.5236, 2.7576, 2.8798, np.inf, np.inf, 2.618, 0.52, 0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558]),
    
    # Default positions for G1
    default_dof_pos=np.array([
                            -0.2, #left hip pitch
                            0.0, #left hip roll
                            0.0, #left hip pitch
                            0.4, #left knee
                            -0.2, #left ankle pitch 
                            0, #left ankle roll 
                            -0.2, #right hip pitch
                            0.0, #right hip roll
                            0.0, #right hip pitch
                            0.4, #right knee
                            -0.2, #right ankle pitch 
                            0, #right ankle roll 
                            0, #waist
                            0, #waist
                            0, #0.12, #waist
                            # 0.,
                            # 0.4,
                            # 0.,
                            # 0.95,
                            # 0.,
                            # 0.,
                            # 0.,
                            # 0.,
                            # -0.4,
                            # 0.,
                            # 0.95,
                            # 0.,
                            # 0.,
                            # 0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            0.,
                            ]),
    
    # Observation scales
    scale_lin_vel=2.0,
    scale_ang_vel=0.25,
    scale_orn=1.0,
    scale_dof_pos=1.0,
    scale_dof_vel=0.05,
    scale_action=0.25,
    
    # Control frequency
    control_frequency=50,
    
    # Motion file path
    motion_file_path="g1_amass_test.pkl"
)

# H1-2 Robot Configuration
H12_CONFIG = RobotConfig(
    name="H1-2",
    hw_dof=27,
    num_actions=12,
    num_upper_body_joints=15,
    leg_joints=list(range(12)),
    upper_body_joints=list(range(12, 27)),
    
    # PD gains for H1-2
    p_gains=np.array([
        300., 200., 200., 300., 40., 40.,  # Left leg
        300., 200., 200., 300., 40., 40.,  # Right leg
        300.,  # Waist
        200., 200., 200., 100., 20., 20., 20.,  # Left arm
        200., 200., 200., 100., 20., 20., 20.   # Right arm
    ]),
    d_gains=np.array([
        3.0, 2.5, 2.5, 4.0, 2.0, 2.0,  # Left leg
        3.0, 2.5, 2.5, 4.0, 2.0, 2.0,  # Right leg
        5.0,  # Waist
        4.0, 4.0, 4.0, 1.0, 0.5, 0.5, 0.5,  # Left arm
        4.0, 4.0, 4.0, 1.0, 0.5, 0.5, 0.5   # Right arm
    ]),
    
    # Joint limits for H1-2
    joint_limit_lo=np.array([
        -0.43, -3.14, -0.43, -0.26, -np.inf, -np.inf,  # Left leg
        -0.43, -3.14, -3.14, -0.24, -np.inf, -np.inf,  # Right leg
        -2.618,  # Waist
        -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558,  # Left arm
        -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.614429558, -1.614429558   # Right arm
    ]),
    joint_limit_hi=np.array([
        0.43, 2.5, 3.14, 2.05, np.inf, np.inf,  # Left leg
        0.43, 2.5, 0.43, 2.0, np.inf, np.inf,  # Right leg
        2.618,  # Waist
        2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558,  # Left arm
        2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.614429558, 1.614429558   # Right arm
    ]),
    
    # Default positions for H1-2
    default_dof_pos=np.array([
        0.0, -0.16, 0.0, 0.36, -0.2, 0,  # Left leg
        0.0, -0.16, 0.0, 0.36, -0.2, 0,  # Right leg
        0,  # Waist
        0.0, 0, 0, 0.0, 0, 0, 0,  # Left arm
        0.0, 0, 0, 0.0, 0, 0, 0   # Right arm
    ]),
    
    # Observation scales
    scale_lin_vel=2.0,
    scale_ang_vel=0.25,
    scale_orn=1.0,
    scale_dof_pos=1.0,
    scale_dof_vel=0.05,
    scale_action=0.25,
    
    # Control frequency
    control_frequency=50,
    
    # Motion file path
    motion_file_path="h12_amass_test.pkl"
)

# Robot configuration registry
ROBOT_CONFIGS = {
    "g1": G1_CONFIG,
    "h12": H12_CONFIG
}

def get_robot_config(robot_name: str) -> RobotConfig:
    """
    Get robot configuration by name
    
    Args:
        robot_name: Name of the robot (g1 or h12)
        
    Returns:
        RobotConfig: Robot configuration object
        
    Raises:
        ValueError: If robot name is not supported
    """
    if robot_name not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot: {robot_name}. Available robots: {list(ROBOT_CONFIGS.keys())}")
    
    return ROBOT_CONFIGS[robot_name]

def list_available_robots() -> list:
    """List all available robot configurations"""
    return list(ROBOT_CONFIGS.keys())

def print_robot_info(config: RobotConfig):
    """Print robot configuration information"""
    print(f"Robot: {config.name}")
    print(f"Hardware DOF: {config.hw_dof}")
    print(f"Leg joints: {len(config.leg_joints)} joints")
    print(f"Upper body joints: {len(config.upper_body_joints)} joints")
    print(f"Control frequency: {config.control_frequency}Hz")
    print(f"Motion file: {config.motion_file_path}")

if __name__ == "__main__":
    # Test the configuration system
    print("Testing robot configuration system...")
    
    # Test G1 config
    g1_config = get_robot_config("g1")
    print("\n" + "="*50)
    print_robot_info(g1_config)
    
    # Test H1-2 config
    h12_config = get_robot_config("h12")
    print("\n" + "="*50)
    print_robot_info(h12_config)
    
    # Test error handling
    try:
        get_robot_config("unknown_robot")
    except ValueError as e:
        print(f"Error handling test: {e}")
    
    print(f"Available robots: {list_available_robots()}")
