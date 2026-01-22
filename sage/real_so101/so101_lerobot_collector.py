# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
SO-101 motion collector using LeRobot for hardware communication.

This module provides motion playback and data collection for the SO-101 robot arm,
outputting data in SAGE-compatible format for sim-to-real analysis.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# LeRobot imports for SO-101 control
try:
    import json

    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
    from lerobot.robots.so101_follower import SO101Follower

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("[WARNING] LeRobot not installed. Install with: pip install lerobot")


# SO-101 joint names matching SAGE simulation (for output compatibility)
SAGE_JOINT_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

# LeRobot motor names (must match calibration file)
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# LeRobot motor IDs for SO-101 (Feetech STS3215 servos)
SO101_MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# Calibration cache path prefix
CALIBRATION_PATH_PREFIX = Path.home() / ".cache/huggingface/lerobot/calibration/robots/"

# Joint offset from simulation coordinates to robot coordinates (in radians)
# Set these by positioning robot at simulation's zero pose and reading actual positions
# Or leave as zeros if simulation zero matches robot calibrated center
JOINT_OFFSETS_RAD = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,  # Gripper uses 0-1 scale, not radians
}

# Joint direction inversion: True = negate radian value before conversion
# Set to True if joint moves opposite to simulation direction
JOINT_INVERTED = {
    "shoulder_pan": False,
    "shoulder_lift": False,
    "elbow_flex": False,
    "wrist_flex": False,
    "wrist_roll": False,
    "gripper": False,
}

# Packed/home position to return to after motion (radians, gripper 0-1)
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
PACKED_POSITION = np.array(
    [
        0.0,  # shoulder_pan: 0°
        -1.8,  # shoulder_lift: ~-103°
        1.7,  # elbow_flex: ~97°
        -1.7,  # wrist_flex: ~-97°
        0.0,  # wrist_roll: 0°
        0.5,  # gripper: 50%
    ]
)


def interpolate_motion(seq, original_freq, target_freq):
    """Interpolate motion sequence to target frequency."""
    n_frames, n_joints = seq.shape
    duration = (n_frames - 1) / original_freq
    t_src = np.linspace(0, duration, n_frames)
    new_n_frames = int(duration * target_freq) + 1
    t_dst = np.linspace(0, duration, new_n_frames)
    seq_interp = np.zeros((new_n_frames, n_joints))
    for j in range(n_joints):
        f = interp1d(t_src, seq[:, j], kind="linear")
        seq_interp[:, j] = f(t_dst)
    return seq_interp


def load_motion_from_txt(txt_file_path):
    """
    Load motion data from TXT/CSV file in SAGE format.

    Args:
        txt_file_path: Path to motion file (CSV with joint names header)

    Returns:
        seq: Joint angle sequence (n_frames, n_joints) in radians
        joint_names: List of joint names
        motion_freq: Motion frequency (default 50Hz)
        motion_name: Motion name from filename
    """
    data_df = pd.read_csv(txt_file_path)
    joint_names = data_df.columns.tolist()
    seq = data_df.values.astype(np.float64)
    motion_name = Path(txt_file_path).stem
    motion_freq = 50  # Default frequency

    return seq, joint_names, motion_freq, motion_name


class So101Collector:
    """
    Collector for SO-101 robot arm using LeRobot driver.

    Handles motion playback and data logging in SAGE-compatible format.
    """

    def __init__(self, port, calibration_path, baudrate=1000000):
        """
        Initialize SO-101 collector.

        Args:
            port: Serial port for Feetech bus
            calibration_path: Path to calibration JSON file
            baudrate: Communication baudrate (default 1000000)
        """
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("LeRobot is not installed. Install with: pip install lerobot")

        self.port = port
        self.baudrate = baudrate
        self.joint_names = SO101_JOINT_NAMES
        self.motor_ids = list(SO101_MOTOR_IDS.values())

        # Data storage
        self.collected_data = {
            "time": [],
            "positions": [],
            "velocities": [],
            "currents": [],
        }
        self.start_monotonic = None
        print(port)

        # Load calibration from cache
        calibration = None
        self.calib_data = None  # Store raw calibration for radian→encoder conversion
        if calibration_path is None:
            raise RuntimeError("calibration_path must be provided")
        calibration_path = Path(calibration_path)
        if calibration_path.exists():
            with open(calibration_path) as f:
                self.calib_data = json.load(f)
            calibration = {
                name: MotorCalibration(
                    id=data["id"],
                    drive_mode=data["drive_mode"],
                    homing_offset=data["homing_offset"],
                    range_min=data["range_min"],
                    range_max=data["range_max"],
                )
                for name, data in self.calib_data.items()
            }
        else:
            raise RuntimeError(f"Calibration not found at {calibration_path}")

        # Initialize motor bus with DEGREES mode for arm joints, RANGE_0_100 for gripper
        motors = {}
        for name, motor_id in SO101_MOTOR_IDS.items():
            if name == "gripper":
                motors[name] = Motor(id=motor_id, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)
            else:
                motors[name] = Motor(id=motor_id, model="sts3215", norm_mode=MotorNormMode.DEGREES)

        self.bus = FeetechMotorsBus(
            port=self.port,
            motors=motors,
            calibration=calibration,
        )
        self.bus.connect()
        self._configure_motors()

    def _configure_motors(self):
        """Configure motor PID and limits for smooth motion."""
        for name in self.joint_names:
            # Lower P gain for smoother motion (default is 32)
            self.bus.write("P_Coefficient", name, 16)
            self.bus.write("I_Coefficient", name, 0)
            self.bus.write("D_Coefficient", name, 32)

            if name == "gripper":
                # Limit gripper torque to avoid burnout
                self.bus.write("Max_Torque_Limit", name, 500)
                self.bus.write("Protection_Current", name, 250)

    def read_state(self):
        """
        Read current joint state from all motors.

        Returns:
            positions: Joint positions in radians (gripper: 0-1 normalized)
            velocities: Joint velocities (raw)
            currents: Motor currents (proxy for torque)
        """
        positions = []
        velocities = []
        currents = []

        for name in self.joint_names:
            # Read raw encoder position
            encoder_pos = self.bus.read("Present_Position", name, normalize=False)

            if name != "gripper":
                # Convert encoder to radians
                pos = self.encoder_to_rad(encoder_pos, name)
            else:
                # Gripper: encoder -> 0-1 normalized
                calib = self.calib_data[name]
                range_min = calib["range_min"]
                range_max = calib["range_max"]
                pos = (encoder_pos - range_min) / (range_max - range_min)
                pos = np.clip(pos, 0.0, 1.0)
            positions.append(pos)

            # Read velocity
            vel = self.bus.read("Present_Velocity", name)
            velocities.append(vel)

            # Read current (as torque proxy)
            cur = self.bus.read("Present_Current", name)
            currents.append(cur)

        return np.array(positions), np.array(velocities), np.array(currents)

    def rad_to_encoder(self, radian, joint_name):
        """
        Convert radian value to raw encoder position.

        Assumes radian 0 = center of calibrated range.
        STS3215: 4096 ticks = 360 degrees = 2π radians

        Args:
            radian: Joint angle in radians
            joint_name: Name of the joint

        Returns:
            Raw encoder value (int)
        """
        # Apply inversion if configured
        if JOINT_INVERTED.get(joint_name, False):
            radian = -radian

        calib = self.calib_data[joint_name]
        range_min = calib["range_min"]
        range_max = calib["range_max"]
        mid = (range_min + range_max) / 2

        # Convert: 1 radian = 4096 / (2π) encoder ticks
        TICKS_PER_RADIAN = 4095 / (2 * np.pi)
        encoder = radian * TICKS_PER_RADIAN + mid

        # Clamp to calibrated range
        encoder = np.clip(encoder, range_min, range_max)
        return int(encoder)

    def encoder_to_rad(self, encoder, joint_name):
        """
        Convert raw encoder position to radian value.

        Args:
            encoder: Raw encoder value
            joint_name: Name of the joint

        Returns:
            Joint angle in radians
        """
        calib = self.calib_data[joint_name]
        range_min = calib["range_min"]
        range_max = calib["range_max"]
        mid = (range_min + range_max) / 2

        TICKS_PER_RADIAN = 4096 / (2 * np.pi)
        radian = (encoder - mid) / TICKS_PER_RADIAN

        # Apply inversion if configured
        if JOINT_INVERTED.get(joint_name, False):
            radian = -radian

        return radian

    def write_positions(self, positions):
        """
        Write goal positions to all motors.

        Args:
            positions: Target positions in radians (6 values), gripper 0-1 normalized
        """
        for i, name in enumerate(self.joint_names):
            if name != "gripper":
                # Convert radians to raw encoder value
                encoder_val = self.rad_to_encoder(positions[i], name)
            else:
                # Gripper: 0-1 normalized -> encoder range
                calib = self.calib_data[name]
                range_min = calib["range_min"]
                range_max = calib["range_max"]
                encoder_val = int(positions[i] * (range_max - range_min) + range_min)
                encoder_val = np.clip(encoder_val, range_min, range_max)

            self.bus.write("Goal_Position", name, encoder_val, normalize=False)

    def enable_torque(self):
        """Enable torque on all motors."""
        for name in self.joint_names:
            self.bus.write("Torque_Enable", name, 1)

    def disable_torque(self):
        """Disable torque on all motors."""
        for name in self.joint_names:
            self.bus.write("Torque_Enable", name, 0)

    def move_to_position(self, target_positions, duration=3.0):
        """
        Move smoothly to target position over specified duration.

        Args:
            target_positions: Target joint positions in radians
            duration: Time to reach target in seconds
        """
        start_positions, _, _ = self.read_state()
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= duration:
                break

            alpha = elapsed / duration
            interpolated = (1 - alpha) * start_positions + alpha * target_positions
            self.write_positions(interpolated)
            time.sleep(0.01)

        self.write_positions(target_positions)

    def safety_check(self, target_positions, max_diff_deg=45.0):
        """
        Check if target position is safe to move to.

        Args:
            target_positions: Target joint positions in radians
            max_diff_deg: Maximum allowed difference in degrees before warning

        Returns:
            True if safe to proceed, False otherwise
        """
        current_pos, _, _ = self.read_state()

        # Also read raw encoder values for debugging
        raw_encoders = []
        for name in self.joint_names:
            raw = self.bus.read("Present_Position", name, normalize=False)
            raw_encoders.append(raw)

        print("\n=== SAFETY CHECK ===")
        print(f"{'Joint':<15} {'Cur(deg)':>10} {'Tgt(deg)':>10} {'Diff':>8} {'CurEnc':>8} {'TgtEnc':>8} {'Range'}")
        print("-" * 90)

        max_diff = 0.0
        for i, name in enumerate(self.joint_names):
            calib = self.calib_data[name]
            range_min = calib["range_min"]
            range_max = calib["range_max"]

            if name == "gripper":
                cur_val = current_pos[i] * 100  # 0-100 scale
                tgt_val = target_positions[i] * 100
                diff = abs(cur_val - tgt_val)
                unit = "%"
                tgt_enc = int(target_positions[i] * (range_max - range_min) + range_min)
            else:
                cur_val = np.rad2deg(current_pos[i])
                tgt_val = np.rad2deg(target_positions[i])
                diff = abs(cur_val - tgt_val)
                max_diff = max(max_diff, diff)
                unit = "°"
                tgt_enc = self.rad_to_encoder(target_positions[i], name)

            warning = " WARN" if diff > max_diff_deg else ""
            print(
                f"{name:<15} {cur_val:>9.1f}{unit} {tgt_val:>9.1f}{unit} {diff:>7.1f}{unit} {raw_encoders[i]:>8} {tgt_enc:>8} [{range_min}-{range_max}]{warning}"
            )

        print("-" * 90)

        if max_diff > max_diff_deg:
            print(f"\nWARNING: Max joint difference is {max_diff:.1f}° (threshold: {max_diff_deg}°)")
            print("The robot will move significantly. Ensure path is clear.")
            response = input("Proceed? [y/N]: ").strip().lower()
            return response == "y"
        else:
            print(f"Max joint difference: {max_diff:.1f}° - OK")
            return True

    def collect_motion(self, motion_seq, control_freq=50, slowdown_factor=2, auto_start=False):
        """
        Execute motion sequence and collect data.

        Args:
            motion_seq: Joint angle sequence (n_frames, n_joints) in radians
            control_freq: Control loop frequency in Hz
            slowdown_factor: Factor to slow down motion (1 = realtime)

        Returns:
            command_times: List of command timestamps
            command_positions: List of commanded positions
        """
        # Safety check before moving
        # if not self.safety_check(motion_seq[0]):
        #     print("Motion cancelled by user.")
        #     return [], []

        n_frames = motion_seq.shape[0]
        loop_dt = 1.0 / control_freq * slowdown_factor

        command_times = []
        command_positions = []

        # Clear previous data
        self.collected_data = {
            "time": [],
            "positions": [],
            "velocities": [],
            "currents": [],
        }

        # Move to start position
        print("Moving to start position...")
        self.move_to_position(motion_seq[0], duration=3.0)
        time.sleep(0.5)

        # Wait for user confirmation before starting motion
        print("\n=== READY TO START MOTION ===")
        print(f"Motion: {n_frames} frames at {control_freq}Hz (slowdown: {slowdown_factor}x)")
        if not auto_start:
            response = input("Press ENTER to start motion, or 'q' to quit: ").strip().lower()
            if response == "q":
                print("Motion cancelled by user.")
                return [], []
        else:
            print("Auto-start enabled, beginning motion...")

        print(f"Starting motion execution: {n_frames} frames at {control_freq}Hz")
        self.start_monotonic = time.monotonic()

        for i in range(n_frames):
            loop_start = time.monotonic()
            t = loop_start - self.start_monotonic

            # Send command
            target_pos = motion_seq[i]
            self.write_positions(target_pos)

            # Record command
            command_times.append(t)
            command_positions.append(target_pos.copy())

            # Read and store state
            positions, velocities, currents = self.read_state()
            self.collected_data["time"].append(t)
            self.collected_data["positions"].append(positions)
            self.collected_data["velocities"].append(velocities)
            self.collected_data["currents"].append(currents)

            # Timing control
            loop_elapsed = time.monotonic() - loop_start
            sleep_time = max(0, loop_dt - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Progress logging
            if i % 50 == 0:
                print(f"  Frame {i}/{n_frames} | Loop: {loop_elapsed*1000:.1f}ms | Sleep: {sleep_time*1000:.1f}ms")

        print(f"Motion completed: {n_frames} frames in {time.monotonic() - self.start_monotonic:.2f}s")

        # Return to packed/home position
        print("Returning to packed position...")
        self.move_to_position(PACKED_POSITION, duration=3.0)
        print("Returned to packed position.")

        return command_times, command_positions

    def close(self):
        """Disconnect from robot."""
        self.disable_torque()
        self.bus.disconnect()


def save_sage_format(
    output_dir,
    motion_name,
    joint_names,
    command_times,
    command_positions,
    collected_data,
):
    """
    Save collected data in SAGE-compatible format.

    Args:
        output_dir: Output directory path
        motion_name: Name of the motion
        joint_names: List of joint names
        command_times: List of command timestamps (seconds)
        command_positions: List of commanded positions
        collected_data: Dict with time, positions, velocities, currents
    """
    motion_dir = os.path.join(output_dir, motion_name)
    os.makedirs(motion_dir, exist_ok=True)

    # 1. Save joint_list.txt
    joint_list_file = os.path.join(motion_dir, "joint_list.txt")
    with open(joint_list_file, "w") as f:
        for joint_name in joint_names:
            f.write(f"{joint_name}\n")

    # 2. Save control.csv (timestamps in microseconds for real robot data)
    control_file = os.path.join(motion_dir, "control.csv")
    with open(control_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions"])
        for t, pos in zip(command_times, command_positions):
            # Convert to microseconds for real robot format
            timestamp_us = t * 1e6
            writer.writerow(["CONTROL", timestamp_us, pos.tolist()])

    # 3. Save event.csv
    event_file = os.path.join(motion_dir, "event.csv")
    with open(event_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "event"])
        if len(command_times) > 0:
            writer.writerow(["EVENT", 0, "CONTROL_MODE_ENABLE"])
            writer.writerow(["EVENT", command_times[0] * 1e6, "MOTION_START"])
            writer.writerow(["EVENT", command_times[-1] * 1e6, "DISABLE"])

    # 4. Save state_motor.csv
    state_file = os.path.join(motion_dir, "state_motor.csv")
    with open(state_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        for i, t in enumerate(collected_data["time"]):
            timestamp_us = t * 1e6
            positions = collected_data["positions"][i].tolist()
            velocities = collected_data["velocities"][i].tolist()
            torques = collected_data["currents"][i].tolist()  # Using current as torque proxy
            writer.writerow(["STATE_MOTOR", timestamp_us, positions, velocities, torques])

    print(f"Data saved to: {motion_dir}")
    print(f"  - {joint_list_file}")
    print(f"  - {control_file}")
    print(f"  - {event_file}")
    print(f"  - {state_file}")


def so101_collector_main(
    motion_file,
    output_dir,
    robot_port,
    robot_type,
    robot_id,
    control_freq=50,
    slowdown_factor=1,
    auto_start=False,
):
    """
    Main function to collect motion data from SO-101.

    Args:
        motion_file: Path to motion file (CSV format)
        output_dir: Output directory for SAGE-format data
        robot_port: Serial port for SO-101
        robot_type: Robot type for calibration path
        robot_id: Robot ID for calibration path
        control_freq: Control frequency in Hz
        slowdown_factor: Factor to slow down motion
        auto_start: If True, skip user confirmation prompts
    """
    # Construct calibration path from prefix, robot_type, and robot_id
    calibration_path = CALIBRATION_PATH_PREFIX / robot_type / f"{robot_id}.json"
    print(f"[SO-101 Collector] Using calibration: {calibration_path}")
    print(f"[SO-101 Collector] Loading motion: {motion_file}")

    # Load motion data
    seq, joint_names, motion_freq, motion_name = load_motion_from_txt(motion_file)
    print(f"  Motion: {motion_name}, {seq.shape[0]} frames, {len(joint_names)} joints")

    # Interpolate to control frequency if needed
    if motion_freq != control_freq:
        print(f"  Resampling from {motion_freq}Hz to {control_freq}Hz")
        seq = interpolate_motion(seq, motion_freq, control_freq)

    # Initialize collector
    print(f"[SO-101 Collector] Connecting to robot on {robot_port}")
    collector = So101Collector(port=robot_port, calibration_path=calibration_path)

    try:
        # Enable torque
        collector.enable_torque()
        time.sleep(0.5)

        # Execute motion and collect data
        command_times, command_positions = collector.collect_motion(
            seq, control_freq=control_freq, slowdown_factor=slowdown_factor, auto_start=auto_start
        )

        # Save data in SAGE format
        save_sage_format(
            output_dir=output_dir,
            motion_name=motion_name,
            joint_names=joint_names,
            command_times=command_times,
            command_positions=command_positions,
            collected_data=collector.collected_data,
        )

        print("[SO-101 Collector] Collection complete!")

    except Exception as e:
        print(f"[SO-101 Collector] Error: {e}")
        raise

    finally:
        collector.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SO-101 motion collector")
    parser.add_argument("--motion-file", type=str, required=True, help="Path to motion file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--robot-type", type=str, default="so101_follower", help="Robot type for calibration path")
    parser.add_argument("--robot-id", type=str, default="my_awesome_follower_arm", help="Robot ID for calibration path")
    parser.add_argument("--control-freq", type=int, default=50, help="Control frequency Hz")
    parser.add_argument("--slowdown", type=int, default=1, help="Slowdown factor")
    args = parser.parse_args()

    so101_collector_main(
        motion_file=args.motion_file,
        output_dir=args.output_dir,
        robot_port=args.robot_port,
        robot_type=args.robot_type,
        robot_id=args.robot_id,
        control_freq=args.control_freq,
        slowdown_factor=args.slowdown,
    )
