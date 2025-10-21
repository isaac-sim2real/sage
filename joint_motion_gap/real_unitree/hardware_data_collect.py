#!/usr/bin/env python3
"""
Multi-Robot Motion Data Collection Pipeline
Supports both G1 and H1-2 robots with unified motion playback system
"""

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import joblib
import numpy as np

# ROS2 imports
import rclpy
from crc import CRC
from gamepad import Gamepad, parse_remote_data
from rclpy.node import Node

# Import robot configuration
from robot_config import get_robot_config, list_available_robots, print_robot_info
from scipy.interpolate import interp1d
from unitree_hg.msg import LowCmd, LowState, MotorCmd

crc = CRC()
# Constants will be set dynamically based on robot config


class MotionState(Enum):
    """Motion playback states"""

    STOPPED = "stopped"
    PLAYING = "playing"
    RETURNING_TO_ZERO = "returning_to_zero"


@dataclass
class MotionInfo:
    """Motion information container"""

    name: str
    total_frames: int
    fps: int
    frame_counter: int = 0
    state: MotionState = MotionState.STOPPED
    return_counter: int = 0
    last_frame_position: Optional[np.ndarray] = None
    playback_speed: float = 0.2  # Playback speed multiplier
    frame_accumulator: float = 0.0  # For non-integer frame stepping


class MotionManager:
    """Manages motion data loading and playback"""

    def __init__(self, motion_file_path: str = "g1_amass_test.pkl", control_frequency: int = 50, robot_config=None):
        self.motion_file_path = motion_file_path
        self.control_frequency = control_frequency
        self.robot_config = robot_config
        self.motion_data: Dict[str, Any] = {}
        self.current_motion_index = 0
        self.motion_names: list = []
        self.current_motion_info: Optional[MotionInfo] = None
        self.interpolated_data: Optional[np.ndarray] = None

        self._load_motion_database()
        self._load_current_motion()

    def _load_motion_database(self) -> None:
        """Load motion data from file"""
        try:
            if not os.path.exists(self.motion_file_path):
                raise FileNotFoundError(f"Motion file not found: {self.motion_file_path}")

            # Check file format and load accordingly
            if self.motion_file_path.endswith(".pkl"):
                self._load_pkl_motion_database()
            elif self.motion_file_path.endswith(".txt"):
                self._load_txt_motion_database()
            else:
                raise ValueError(f"Unsupported file format: {self.motion_file_path}")

        except Exception as e:
            print(f"Failed to load motion database: {e}")
            self.motion_data = {}
            self.motion_names = []

    def _load_pkl_motion_database(self) -> None:
        """Load motion data from pickle file"""
        with open(self.motion_file_path, "rb") as f:
            self.motion_data = joblib.load(f)

        self.motion_names = list(self.motion_data.keys())
        print(f"Loaded {len(self.motion_names)} motions from {self.motion_file_path}")

    def _load_txt_motion_database(self) -> None:
        """Load motion data from TXT file"""
        with open(self.motion_file_path, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError("TXT file must have at least 2 lines (header + data)")

        # Parse joint names from first line
        joint_names = [name.strip() for name in lines[0].strip().split(",")]

        # Parse motion data from subsequent lines (skip first line)
        motion_frames = []
        for line in lines[1:]:
            if not line.strip():  # Skip empty lines
                continue
            values = [float(val.strip()) for val in line.strip().split(",")]
            if len(values) == len(joint_names):
                motion_frames.append(values)

        if not motion_frames:
            raise ValueError("No valid motion frames found in TXT file")

        # Convert to numpy array
        dof_data = np.array(motion_frames, dtype=np.float32)

        # Create motion data structure - motion name is filename without extension
        motion_name = os.path.splitext(os.path.basename(self.motion_file_path))[0]
        self.motion_data = {
            motion_name: {"dof": dof_data, "fps": 50, "joint_names": joint_names}  # TXT files are at 50Hz
        }

        self.motion_names = [motion_name]
        print(f"Loaded motion from TXT: {motion_name} ({dof_data.shape[0]} frames, {len(joint_names)} joints)")

    def _load_current_motion(self) -> None:
        """Load and interpolate current motion"""
        if not self.motion_names:
            return

        motion_name = self.motion_names[self.current_motion_index]
        motion = self.motion_data[motion_name]

        # Extract and interpolate data
        dof_data = motion["dof"]
        fps = motion["fps"]

        self.interpolated_data = self._interpolate_motion(dof_data, fps, target_fps=self.control_frequency)

        # Create motion info
        self.current_motion_info = MotionInfo(
            name=motion_name,
            total_frames=self.interpolated_data.shape[0],
            fps=self.control_frequency,
            playback_speed=0.5,
            frame_accumulator=0.0,
        )

        print(
            f"Loaded motion: {motion_name} "
            f"({self.interpolated_data.shape[0]} frames, "
            f"{self.interpolated_data.shape[0]/self.control_frequency:.1f}s)"
        )

    def _interpolate_motion(self, dof_data: np.ndarray, fps: int, target_fps: int) -> np.ndarray:
        """Interpolate motion data to target frequency"""
        original_frames = dof_data.shape[0]
        target_frames = int(original_frames * target_fps / fps)

        original_time = np.linspace(0, (original_frames - 1) / fps, original_frames)
        target_time = np.linspace(0, (original_frames - 1) / fps, target_frames)

        interpolated = np.zeros((target_frames, dof_data.shape[1]), dtype=np.float32)
        for dof_idx in range(dof_data.shape[1]):
            interpolator = interp1d(
                original_time, dof_data[:, dof_idx], kind="linear", bounds_error=False, fill_value="extrapolate"
            )
            interpolated[:, dof_idx] = interpolator(target_time)

        return interpolated

    def start_playback(self) -> bool:
        """Start motion playback"""
        if not self.current_motion_info or self.interpolated_data is None:
            return False

        self.current_motion_info.state = MotionState.PLAYING
        self.current_motion_info.frame_counter = 0
        print(f"Started playing: {self.current_motion_info.name}")
        return True

    def stop_playback(self) -> None:
        """Stop motion playback"""
        if self.current_motion_info:
            self.current_motion_info.state = MotionState.STOPPED
            print(f"Stopped playing: {self.current_motion_info.name}")

    def next_motion(self) -> bool:
        """Switch to next motion"""
        if not self.motion_names:
            return False

        self.current_motion_index = (self.current_motion_index + 1) % len(self.motion_names)
        self._load_current_motion()
        return True

    def get_current_motion_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current motion"""
        if not self.current_motion_info:
            return None

        motion_data = self.motion_data.get(self.current_motion_info.name, {})
        return {
            "name": self.current_motion_info.name,
            "total_frames": self.current_motion_info.total_frames,
            "current_frame": self.current_motion_info.frame_counter,
            "fps": self.current_motion_info.fps,
            "duration": self.current_motion_info.total_frames / self.current_motion_info.fps,
            "state": self.current_motion_info.state.value,
            "playback_speed": self.current_motion_info.playback_speed,
            "joint_names": motion_data.get("joint_names", []),
            "num_joints": len(motion_data.get("joint_names", [])),
        }

    def get_motion_joint_names(self) -> Optional[list]:
        """Get joint names for current motion"""
        if not self.current_motion_info:
            return None

        motion_data = self.motion_data.get(self.current_motion_info.name, {})
        return motion_data.get("joint_names", [])

    def get_all_motion_names(self) -> list:
        """Get list of all available motion names"""
        return self.motion_names.copy()

    def set_motion_by_index(self, index: int) -> bool:
        """Set current motion by index"""
        if 0 <= index < len(self.motion_names):
            self.current_motion_index = index
            self._load_current_motion()
            return True
        return False

    def set_motion_by_name(self, name: str) -> bool:
        """Set current motion by name"""
        if name in self.motion_names:
            self.current_motion_index = self.motion_names.index(name)
            self._load_current_motion()
            return True
        return False

    def set_playback_speed(self, speed: float) -> None:
        """Set playback speed (0.1 to 2.0)"""
        if self.current_motion_info:
            self.current_motion_info.playback_speed = np.clip(speed, 0.1, 2.0)
            print(f"Playback speed: {self.current_motion_info.playback_speed:.1f}x")

    def get_playback_speed(self) -> float:
        """Get current playback speed"""
        return self.current_motion_info.playback_speed if self.current_motion_info else 1.0

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current motion frame with smooth speed control"""
        if not self.current_motion_info or self.interpolated_data is None:
            return None

        if self.current_motion_info.state != MotionState.PLAYING:
            return None

        if self.current_motion_info.frame_counter >= self.current_motion_info.total_frames:
            self._handle_motion_completion()
            return None

        # Use speed control for smooth interpolation
        speed = self.current_motion_info.playback_speed
        self.current_motion_info.frame_accumulator += speed

        # Advance frame when accumulated frames reach 1
        if self.current_motion_info.frame_accumulator >= 1.0:
            self.current_motion_info.frame_counter += int(self.current_motion_info.frame_accumulator)
            self.current_motion_info.frame_accumulator -= int(self.current_motion_info.frame_accumulator)

            # Check if out of range
            if self.current_motion_info.frame_counter >= self.current_motion_info.total_frames:
                self._handle_motion_completion()
                return None

        # Get current and next frames for interpolation
        current_frame_idx = int(self.current_motion_info.frame_counter)
        next_frame_idx = min(current_frame_idx + 1, self.current_motion_info.total_frames - 1)

        # Calculate interpolation weight
        interpolation_weight = self.current_motion_info.frame_accumulator

        # Get current and next frames
        current_frame = self.interpolated_data[current_frame_idx]
        next_frame = self.interpolated_data[next_frame_idx]

        # Linear interpolation
        interpolated_frame = current_frame + interpolation_weight * (next_frame - current_frame)

        return interpolated_frame

    def _handle_motion_completion(self) -> None:
        """Handle motion completion - start return to zero"""
        if self.current_motion_info is not None:
            self.current_motion_info.state = MotionState.RETURNING_TO_ZERO
        if self.current_motion_info is not None:
            self.current_motion_info.return_counter = 0
        if self.robot_config:
            if self.current_motion_info is not None and self.interpolated_data is not None:
                index = self.current_motion_info.frame_counter - 1
                if 0 <= index < len(self.interpolated_data):
                    self.current_motion_info.last_frame_position = self.interpolated_data[index][
                        self.robot_config.upper_body_joints
                    ]
        print("Motion completed, returning to zero position")


class UpperBodyController:
    """Controls upper body motion based on motion data"""

    def __init__(self, default_positions: np.ndarray, robot_config=None):
        self.robot_config = robot_config
        if robot_config:
            self.default_positions = default_positions[robot_config.upper_body_joints]
        else:
            self.default_positions = default_positions
        self.return_steps = 50

    def get_upper_body_targets(self, motion_manager: MotionManager) -> np.ndarray:
        """Get upper body target positions based on motion state"""
        if motion_manager.current_motion_info is None:
            return self.default_positions

        state = motion_manager.current_motion_info.state

        if state == MotionState.STOPPED:
            return self.default_positions

        elif state == MotionState.PLAYING:
            frame = motion_manager.get_current_frame()
            if frame is not None and self.robot_config:
                return frame[self.robot_config.upper_body_joints]
            return self.default_positions

        elif state == MotionState.RETURNING_TO_ZERO:
            return self._handle_return_to_zero(motion_manager)

        return self.default_positions

    def _handle_return_to_zero(self, motion_manager: MotionManager) -> np.ndarray:
        """Handle smooth return to zero position"""
        info = motion_manager.current_motion_info
        if info is not None:
            info.return_counter += 1

        if info is not None:
            if info.return_counter >= self.return_steps:
                info.state = MotionState.STOPPED
                info.return_counter = 0
                print("Return to zero completed")
                return self.default_positions

        # Smooth interpolation to zero
        if info is not None:
            progress = info.return_counter / self.return_steps
            if info.last_frame_position is not None:
                return info.last_frame_position * (1 - progress)

        return self.default_positions


class RobotConfig:
    """Robot configuration container"""

    def __init__(self, config):
        self.name = config.name
        self.hw_dof = config.hw_dof
        self.num_actions = config.num_actions
        self.num_upper_body_joints = config.num_upper_body_joints
        self.leg_joints = config.leg_joints
        self.upper_body_joints = config.upper_body_joints

        # PD gains
        self.p_gains = config.p_gains
        self.d_gains = config.d_gains

        # Joint limits
        self.joint_limit_lo = config.joint_limit_lo
        self.joint_limit_hi = config.joint_limit_hi

        # Default positions
        self.default_dof_pos = config.default_dof_pos

        # Observation scales
        self.scale_lin_vel = config.scale_lin_vel
        self.scale_ang_vel = config.scale_ang_vel
        self.scale_orn = config.scale_orn
        self.scale_dof_pos = config.scale_dof_pos
        self.scale_dof_vel = config.scale_dof_vel
        self.scale_action = config.scale_action

        # Control frequency
        self.control_frequency = config.control_frequency

        # Motion file path
        self.motion_file_path = config.motion_file_path


class MultiRobotMotionControlNode(Node):
    """Main ROS2 node for multi-robot motion control"""

    def __init__(self, robot_config, output_dir):
        super().__init__("multi_robot_motion_control")

        # Store robot configuration
        self.robot_config = robot_config
        self.output_dir = output_dir

        # Initialize components
        self.motion_manager = MotionManager(
            motion_file_path=robot_config.motion_file_path,
            control_frequency=robot_config.control_frequency,
            robot_config=robot_config,
        )
        self.upper_body_controller = UpperBodyController(robot_config.default_dof_pos, robot_config)

        # ROS2 setup
        self._setup_ros2()

        # Control state
        self.motion_active = False
        self.current_speed = 1.0  # Current playback speed

        # Data recording control
        self.is_recording = False
        self.recording_data = {
            "time": [],
            "command_time": [],
            "command_val": [],
            "joint_positions": [],
            "joint_velocities": [],
            "joint_torques": [],
            "joint_temperatures": [],
            "imu_data": [],
            "ang_vel_data": [],
        }
        self.start_time = time.monotonic()

        print(f"{robot_config.name} Motion Data Collection Pipeline initialized")
        print(
            "Controls: L1=Emergency stop, B=Start/Stop motion, "
            "A=Decrease speed, Y=Increase speed, "
            "X=Reset speed, R1=Exit"
        )

    def _setup_ros2(self):
        """Setup ROS2 subscribers and publishers"""
        # Subscribers
        self.state_sub = self.create_subscription(LowState, "lowstate", self._state_callback, 1)

        # Publishers
        self.cmd_pub = self.create_publisher(LowCmd, "lowcmd_buffer", 1)

        # Command message
        self.cmd_msg = LowCmd()
        self.cmd_msg.mode_pr = 0
        self.cmd_msg.mode_machine = 5

        self.motor_cmd = []
        for id in range(self.robot_config.hw_dof):
            cmd = MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=1, reserve=0)
            self.motor_cmd.append(cmd)
        for id in range(self.robot_config.hw_dof, 35):
            cmd = MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0, reserve=0)
            self.motor_cmd.append(cmd)
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        # Gamepad
        self.gamepad = Gamepad()
        # self.crc = CRC()

        # State variables
        self.joint_positions = np.zeros(self.robot_config.hw_dof)
        self.joint_velocities = np.zeros(self.robot_config.hw_dof)
        self.joint_tau = np.zeros(self.robot_config.hw_dof)
        self.joint_temp = np.zeros(self.robot_config.hw_dof)
        self.imu_data = None

    def _state_callback(self, msg: LowState):
        """Handle incoming robot state"""
        # Parse gamepad input
        joystick_data = msg.wireless_remote
        parsed_data = parse_remote_data(joystick_data)
        self.gamepad.update(parsed_data)

        # Handle button presses
        self._handle_buttons()

        # Update robot state
        self.joint_positions = np.array([msg.motor_state[i].q for i in range(self.robot_config.hw_dof)])
        self.joint_velocities = np.array([msg.motor_state[i].dq for i in range(self.robot_config.hw_dof)])
        self.imu_data = msg.imu_state

        self.msg_tick = msg.tick / 1000
        self.roll, self.pitch, self.yaw = self.imu_data.rpy
        self.obs_ang_vel = np.array(self.imu_data.gyroscope) * 0.25
        self.obs_imu = np.array([self.roll, self.pitch, self.yaw])
        self.obs_root_rot = np.array(self.imu_data.quaternion, dtype=np.float32)
        # motor data
        self.joint_tau = [msg.motor_state[i].tau_est for i in range(self.robot_config.hw_dof)]
        self.joint_temp = [msg.motor_state[i].temperature for i in range(self.robot_config.hw_dof)]

    def _handle_buttons(self):
        """Handle gamepad button presses"""
        if self.gamepad.L1.pressed:
            self.motion_active = False
            print("Emergency stop!")

        if self.gamepad.R1.pressed:
            print("Exiting...")
            rclpy.shutdown()

        if self.gamepad.B.pressed:
            self._handle_motion_button()

        # Speed control buttons
        if self.gamepad.A.pressed:
            self._decrease_speed()

        if self.gamepad.Y.pressed:
            self._increase_speed()

        if self.gamepad.X.pressed:
            self._reset_speed()

    def _handle_motion_button(self):
        """Handle B button for motion control"""
        if not self.motion_active:
            # Start motion and recording
            if self.motion_manager.start_playback():
                self.motion_active = True
                self.start_recording()
                print("Motion playback and data recording started")
            else:
                print("Failed to start motion")
        else:
            # Stop motion, stop recording, and switch to next
            self.motion_manager.stop_playback()
            self.motion_active = False
            self.stop_recording()

            if self.motion_manager.next_motion():
                print("Switched to next motion")
            else:
                print("Failed to switch motion")

    def _compute_target_positions(self):
        """Compute target joint positions"""
        # Get upper body targets from motion
        upper_body_targets = self.upper_body_controller.get_upper_body_targets(self.motion_manager)

        # Create full target vector (legs always default, upper body from motion)
        leg_targets = self.robot_config.default_dof_pos[self.robot_config.leg_joints]
        full_targets = np.concatenate([leg_targets, upper_body_targets])

        # Apply joint limits
        target_positions = np.clip(full_targets, self.robot_config.joint_limit_lo, self.robot_config.joint_limit_hi)

        return target_positions

    def _send_commands(self, target_positions: np.ndarray):
        """Send motor commands to robot"""
        for i in range(self.robot_config.hw_dof):
            self.motor_cmd[i].q = target_positions[i]
            self.motor_cmd[i].kp = self.robot_config.p_gains[i]
            self.motor_cmd[i].kd = self.robot_config.d_gains[i]

        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.cmd_msg.crc = crc.Crc(self.cmd_msg)
        self.cmd_pub.publish(self.cmd_msg)

    def _decrease_speed(self):
        """Decrease playback speed"""
        self.current_speed = max(0.1, self.current_speed - 0.1)
        self.motion_manager.set_playback_speed(self.current_speed)

    def _increase_speed(self):
        """Increase playback speed"""
        self.current_speed = min(2.0, self.current_speed + 0.1)
        self.motion_manager.set_playback_speed(self.current_speed)

    def _reset_speed(self):
        """Reset playback speed to 1.0x"""
        self.current_speed = 1.0
        self.motion_manager.set_playback_speed(self.current_speed)

    def synchronize_dual_arm_data(self, data1, data2):
        """
        Synchronize dual-arm data, align two time series to the same time points

        Args:
            data1: left arm data dict
            data2: right arm data dict

        Returns:
            sync_data1: synchronized left arm data
            sync_data2: synchronized right arm data
            common_time: common time series
        """
        time1 = np.array(data1["time"])
        time2 = np.array(data2["time"])

        # Find common time range
        start_time = max(time1[0], time2[0])
        end_time = min(time1[-1], time2[-1])

        # Filter valid time range
        valid_idx1 = (time1 >= start_time) & (time1 <= end_time)
        valid_idx2 = (time2 >= start_time) & (time2 <= end_time)

        time1_valid = time1[valid_idx1]
        time2_valid = time2[valid_idx2]

        # Choose the one with fewer samples as the target time series
        if len(time1_valid) <= len(time2_valid):
            target_time = time1_valid

            # data1 uses valid range directly
            sync_data1 = {
                "time": time1_valid,
                "angle_rad": np.array(data1["angle_rad"])[valid_idx1],
                "velocity_rads": np.array(data1["velocity_rads"])[valid_idx1],
                "current": np.array(data1["current"])[valid_idx1],
            }

            # Interpolate data2
            sync_data2 = {}
            sync_data2["time"] = target_time

            for key in ["angle_rad", "velocity_rads", "current"]:
                data2_array = np.array(data2[key])[valid_idx2]

                # Interpolate each joint separately
                if data2_array.ndim == 2:  # (n_samples, n_joints)
                    n_joints = data2_array.shape[1]
                    interpolated_data = np.zeros((len(target_time), n_joints))

                    for joint_idx in range(n_joints):
                        # Use linear interpolation
                        interp_func = interp1d(
                            time2_valid,
                            data2_array[:, joint_idx],
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        interpolated_data[:, joint_idx] = interp_func(target_time)

                    sync_data2[key] = interpolated_data
                else:  # 1D array
                    interp_func = interp1d(
                        time2_valid, data2_array, kind="linear", bounds_error=False, fill_value="extrapolate"
                    )
                    sync_data2[key] = interp_func(target_time)

        else:
            # Use time2 as target time series, interpolate data1
            target_time = time2_valid

            # data2 uses valid range directly
            sync_data2 = {
                "time": time2_valid,
                "angle_rad": np.array(data2["angle_rad"])[valid_idx2],
                "velocity_rads": np.array(data2["velocity_rads"])[valid_idx2],
                "current": np.array(data2["current"])[valid_idx2],
            }

            # Interpolate data1
            sync_data1 = {}
            sync_data1["time"] = target_time

            for key in ["angle_rad", "velocity_rads", "current"]:
                data1_array = np.array(data1[key])[valid_idx1]

                # Interpolate each joint separately
                if data1_array.ndim == 2:  # (n_samples, n_joints)
                    n_joints = data1_array.shape[1]
                    interpolated_data = np.zeros((len(target_time), n_joints))

                    for joint_idx in range(n_joints):
                        # Use linear interpolation
                        interp_func = interp1d(
                            time1_valid,
                            data1_array[:, joint_idx],
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        interpolated_data[:, joint_idx] = interp_func(target_time)

                    sync_data1[key] = interpolated_data
                else:  # 1D array
                    interp_func = interp1d(
                        time1_valid, data1_array, kind="linear", bounds_error=False, fill_value="extrapolate"
                    )
                    sync_data1[key] = interp_func(target_time)

        common_time = target_time

        print(f"Data synchronized: {len(common_time)} samples, {start_time:.3f}s - {end_time:.3f}s")

        return sync_data1, sync_data2, common_time

    def save_motion_csv_files(
        self, output_dir, motion_name, command_time_list, command_val_list, robot_joint_names, data1, data2
    ):
        """
        Save 3 CSV files: control.csv, event.csv, state_motor.csv

        Args:
            output_dir: output directory
            motion_name: motion name
            command_time_list: command time list
            command_val_list: command value list (n, 29)
            robot_joint_names: list of joint names
            data1: left arm data dict
            data2: right arm data dict
        """
        # Create motion-specific directory
        motion_dir = os.path.join(output_dir, motion_name)
        os.makedirs(motion_dir, exist_ok=True)

        # Synchronize dual-arm data
        sync_data1, sync_data2, common_time = self.synchronize_dual_arm_data(data1, data2)

        # 1. Save joint_list.txt (joint name list)
        joint_list_file = os.path.join(motion_dir, "joint_list.txt")
        with open(joint_list_file, "w") as f:
            for joint_name in robot_joint_names:
                f.write(f"{joint_name}\n")

        # 2. Save control.csv (command data)
        control_file = os.path.join(motion_dir, "control.csv")
        with open(control_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "positions"])

            for i, (time_val, command_val) in enumerate(zip(command_time_list, command_val_list)):
                writer.writerow(["CONTROL", time_val * 1e5, command_val.tolist()])

        # 3. Save event.csv (event data)
        event_file = os.path.join(motion_dir, "event.csv")
        with open(event_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "event"])
            # Add motion start and end events
            if len(command_time_list) > 0:
                writer.writerow(["EVENT", 0, "CONTROL_MODE_ENABLE"])
                writer.writerow(["EVENT", command_time_list[0] * 1e5, "MOTION_START"])
                writer.writerow(["EVENT", command_time_list[-1] * 1e5, "MOTION_END"])

        # 4. Save state_motor.csv (actual state data)
        state_motor_file = os.path.join(motion_dir, "state_motor.csv")
        with open(state_motor_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])

            # Use synchronized data
            for i in range(len(common_time)):
                timestamp = common_time[i]

                # Merge position data (17 left + 17 right = 34 total)
                positions = np.concatenate([sync_data1["angle_rad"][i], sync_data2["angle_rad"][i]])

                # Merge velocity data
                velocities = np.concatenate([sync_data1["velocity_rads"][i], sync_data2["velocity_rads"][i]])

                # Merge torque data (use current as torque proxy)
                torques = np.concatenate([sync_data1["current"][i], sync_data2["current"][i]])

                writer.writerow(
                    ["STATE_MOTOR", timestamp * 1e5, positions.tolist(), velocities.tolist(), torques.tolist()]
                )

        print(f"CSV files saved to: {motion_dir}")

    def start_recording(self):
        """Start data recording"""
        self.is_recording = True
        self.start_time = time.monotonic()

        # Clear previous data
        for key in self.recording_data:
            self.recording_data[key].clear()

        print("Data recording started")

    def stop_recording(self):
        """Stop data recording and save CSV files"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Get motion name
        motion_name = (
            self.motion_manager.current_motion_info.name
            if self.motion_manager.current_motion_info
            else "unknown_motion"
        )
        current_time = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        motion_name_with_time = f"{motion_name}{current_time}"

        # Create output directory
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for CSV saving
        command_time_list = self.recording_data["command_time"]
        command_val_list = self.recording_data["command_val"]

        # Create joint names
        robot_joint_names = [f"joint_{i}" for i in range(self.robot_config.hw_dof)]

        # Split data into left and right arms (simulate dual-arm data)
        # Use first half and second half of upper body joints
        upper_body_start = len(self.robot_config.leg_joints)
        upper_body_end = self.robot_config.hw_dof
        mid_point = upper_body_start + (upper_body_end - upper_body_start) // 2

        # Create simulated dual-arm data structure
        data1 = {
            "time": np.array(self.recording_data["time"]),
            "angle_rad": np.array(self.recording_data["joint_positions"])[:, upper_body_start:mid_point],
            "velocity_rads": np.array(self.recording_data["joint_velocities"])[:, upper_body_start:mid_point],
            "current": np.array(self.recording_data["joint_torques"])[:, upper_body_start:mid_point],
        }

        data2 = {
            "time": np.array(self.recording_data["time"]),
            "angle_rad": np.array(self.recording_data["joint_positions"])[:, mid_point:upper_body_end],
            "velocity_rads": np.array(self.recording_data["joint_velocities"])[:, mid_point:upper_body_end],
            "current": np.array(self.recording_data["joint_torques"])[:, mid_point:upper_body_end],
        }

        # Save CSV files
        self.save_motion_csv_files(
            output_dir=output_dir,
            motion_name=motion_name_with_time,
            command_time_list=command_time_list,
            command_val_list=command_val_list,
            robot_joint_names=robot_joint_names,
            data1=data1,
            data2=data2,
        )

        print("Data recording completed and CSV files saved")

    def run(self):
        """Main control loop"""
        print("Starting motion data collection loop...")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)

            # Record data if recording is active
            if self.is_recording:
                current_time = time.monotonic() - self.start_time
                self.recording_data["time"].append(current_time)
                self.recording_data["command_time"].append(current_time)

                # Record current joint states
                self.recording_data["joint_positions"].append(self.joint_positions.copy())
                self.recording_data["joint_velocities"].append(self.joint_velocities.copy())
                self.recording_data["joint_torques"].append(self.joint_tau.copy())
                self.recording_data["joint_temperatures"].append(self.joint_temp.copy())

                # Record IMU data
                self.recording_data["imu_data"].append(self.obs_imu.copy())
                # self.recording_data['ang_vel_data'].append(self.obs_ang_vel.copy())

                # Record command values
                target_positions = self._compute_target_positions()
                self.recording_data["command_val"].append(target_positions.copy())

            # Always compute and send target positions
            target_positions = self._compute_target_positions()
            self._send_commands(target_positions)

            # Control loop timing
            time.sleep(1.0 / self.robot_config.control_frequency)


def unitree_robot_main(robot_name, motion_file, output_dir):
    """Main entry point"""

    # Get robot configuration
    try:
        robot_config = get_robot_config(robot_name)

        # Set motion file path if provided
        if motion_file:
            robot_config.motion_file_path = motion_file
            print(f"Using motion file: {motion_file}")

        print_robot_info(robot_config)
    except ValueError as e:
        print(f"Error: {e}")
        print("Available robots:", list_available_robots())
        return

    # Initialize ROS2
    rclpy.init()

    # Create and run the node
    node = MultiRobotMotionControlNode(robot_config, output_dir)

    try:
        node.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        rclpy.shutdown()
