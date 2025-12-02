# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import csv
import os
from datetime import datetime
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from .assets import get_robot_cfg


def get_motion_files(motion_files_arg):
    """Process motion files argument and return list of motion files."""
    # If it's a directory, get all txt files
    if os.path.isdir(motion_files_arg):
        files = []
        for file in os.listdir(motion_files_arg):
            if file.endswith(".txt"):
                files.append(os.path.join(motion_files_arg, file))
        return sorted(files)

    # If it's multiple files (comma-separated)
    if "," in motion_files_arg:
        return [f.strip() for f in motion_files_arg.split(",")]

    # Single file
    return [motion_files_arg]


def get_motion_name(motion_file):
    """Generate motion name from file path."""
    return Path(motion_file).stem


def log_message(message):
    """Format and print log messages with timestamp."""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[Benchmark][Log][{current_time}] {message}")


class JointMotionBenchmark:
    def __init__(self, args):
        # Get repository path
        self.repo_path = os.getcwd()

        # Store command line arguments
        self.robot_name = args.robot_name.lower()
        self.motion_source = args.motion_source.lower()
        self.valid_joints_file = args.valid_joints_file
        self.output_folder = args.output_folder
        self.fix_root = args.fix_root
        self.headless = args.headless
        self.physics_freq = args.physics_freq
        self.render_freq = args.render_freq
        self.original_control_freq = args.original_control_freq
        self.solver_type = args.solver_type
        self.control_freq = args.control_freq
        self.device = args.device
        self.kp = args.kp
        self.kd = args.kd

        # Check frequency divisibility
        if self.physics_freq % self.render_freq != 0:
            raise ValueError("Physics frequency must be divisible by render frequency.")
        if self.render_freq % self.control_freq != 0:
            raise ValueError("Render frequency must be divisible by control frequency.")

        # Initialize simulation parameters
        self.physics_dt = 1.0 / self.physics_freq
        self.render_dt = 1.0 / self.render_freq
        self.control_dt = 1.0 / self.control_freq
        self.divisor = self.render_freq // self.control_freq

        # Load valid joint names from config file
        self.set_valid_joints = False
        self.valid_joint_names = []
        self._load_valid_joints()

        # Initialize simulation environment
        self._setup_simulation()

    def _load_valid_joints(self):
        """Load valid joint names from valid_joints_file"""
        if self.valid_joints_file is not None:
            config_file = self.valid_joints_file
        else:
            config_file = os.path.join(self.repo_path, f"configs/{self.robot_name}_valid_joints.txt")

        # Check if the config file can be opened
        if not os.path.isfile(config_file):
            self.set_valid_joints = False
            log_message(f"No valid joints file found: {config_file}, use all joints in motion file")
            return

        with open(config_file) as file:
            self.set_valid_joints = True
            self.valid_joint_names = [line.strip() for line in file if line.strip()]
        log_message(f"Loaded {len(self.valid_joint_names)} valid joint names from {config_file}")

    def _setup_simulation(self):
        """Set up the simulation environment."""
        # Create simulation context
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.physics_dt,
            render_interval=int(self.render_dt / self.physics_dt),
            device=self.device,
            physx=sim_utils.PhysxCfg(
                solver_type=self.solver_type,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        # Get device from simulation context
        self.device = self.sim.device

        # Ground-plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Get robot ArticulationCfg
        robot_cfg = get_robot_cfg(self.robot_name).copy()

        # Set prim path
        robot_cfg.prim_path = "/World/Robot"

        # Overwrite fix root link if specified
        if self.fix_root:
            robot_cfg.spawn.articulation_props.fix_root_link = True

        # Create robot articulation
        self.robot = Articulation(cfg=robot_cfg)

        # Reset simulation
        self.sim.reset()

        log_message(f"Loaded robot {self.robot_name}")

    def setup_motion(self, motion_file, motion_name):
        """Set up and prepare motion data for benchmark."""
        self.motion_file = motion_file
        self.motion_name = motion_name

        # Parse motion file and configure joints
        self._parse_motion_file()

        # Load and preprocess motion data (includes resampling if needed)
        log_message(f"Loading motion data from {self.motion_file}...")
        self.joint_angles = self._load_motion()

        # Initialize logging
        self._init_logger()

    def _parse_motion_file(self):
        """Parse motion file and set up joint mappings."""
        # Load joint configuration data
        with open(self.motion_file) as file:
            self._motion_lines = file.readlines()

        # Parse joint names from first line
        self._all_joint_names = [name.strip().split("/")[-1] for name in self._motion_lines[0].strip().split(",")]

        # Filter joints to only include valid ones
        if self.set_valid_joints:
            self.joint_names = [joint for joint in self._all_joint_names if joint in self.valid_joint_names]
            log_message(f"Filtered {len(self.joint_names)} valid joints from motion file")
        else:
            self.joint_names = self._all_joint_names
            log_message(f"Using all {len(self.joint_names)} joints from motion file")

        if not self.joint_names:
            raise ValueError("No valid joints found in motion file that match the config file")

        # Find indices of valid joints in the motion file
        self._valid_motion_indices = [self._all_joint_names.index(joint) for joint in self.joint_names]

        # Map joint names to robot joint indices
        self.joint_indices = []
        for joint in self.joint_names:
            if joint in self.robot.data.joint_names:
                index = self.robot.data.joint_names.index(joint)
                self.joint_indices.append(index)

    def _apply_gain_overrides(self):
        """Apply kp/kd overrides to valid joints."""
        if self.kp is None and self.kd is None:
            return

        num_joints = len(self.joint_indices)
        log_message("Applying gain overrides for valid joints")

        # Process kp
        if self.kp is not None:
            if len(self.kp) == 1:
                kp_values = torch.full((num_joints,), self.kp[0], dtype=torch.float, device=self.device)
                log_message(f"Overriding kp with single value: {self.kp[0]} for all joints")
            elif len(self.kp) == num_joints:
                kp_values = torch.tensor(self.kp, dtype=torch.float, device=self.device)
                log_message(f"Overriding kp with per-joint values: {self.kp}")
            else:
                raise ValueError(f"kp list length ({len(self.kp)}) must be 1 or match valid joints ({num_joints})")

            current_stiffness = self.robot.data.joint_stiffness[0, self.joint_indices].cpu().tolist()
            log_message(f"kp before override: {current_stiffness}")

            self.robot.write_joint_stiffness_to_sim(stiffness=kp_values, joint_ids=self.joint_indices)

            final_stiffness = self.robot.data.joint_stiffness[0, self.joint_indices].cpu().tolist()
            log_message(f"kp after override: {final_stiffness}")

        # Process kd
        if self.kd is not None:
            if len(self.kd) == 1:
                kd_values = torch.full((num_joints,), self.kd[0], dtype=torch.float, device=self.device)
                log_message(f"Overriding kd with single value: {self.kd[0]} for all joints")
            elif len(self.kd) == num_joints:
                kd_values = torch.tensor(self.kd, dtype=torch.float, device=self.device)
                log_message(f"Overriding kd with per-joint values: {self.kd}")
            else:
                raise ValueError(f"kd list length ({len(self.kd)}) must be 1 or match valid joints ({num_joints})")

            current_damping = self.robot.data.joint_damping[0, self.joint_indices].cpu().tolist()
            log_message(f"kd before override: {current_damping}")

            self.robot.write_joint_damping_to_sim(damping=kd_values, joint_ids=self.joint_indices)

            final_damping = self.robot.data.joint_damping[0, self.joint_indices].cpu().tolist()
            log_message(f"kd after override: {final_damping}")

        log_message("Gain overrides applied successfully")

    def _init_logger(self):
        """Initialize logger for joint motion benchmark"""
        # Create output directory with proper structure
        self.sim_output_folder = os.path.join(
            self.output_folder, "sim", self.robot_name, self.motion_source, self.motion_name
        )
        os.makedirs(self.sim_output_folder, exist_ok=True)

        # Setup log files with new path structure
        self.joint_list_file = os.path.join(self.sim_output_folder, "joint_list.txt")
        self.control_file = os.path.join(self.sim_output_folder, "control.csv")
        self.dof_file = os.path.join(self.sim_output_folder, "state_motor.csv")

        # Write joint list
        with open(self.joint_list_file, "w") as file:
            for joint in self.joint_names:
                file.write(f"{joint}\n")

        # Initialize control log
        with open(self.control_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "positions"])

        # Initialize state motor log
        with open(self.dof_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])

    def _load_motion(self):
        """Load motion data from file"""
        joint_angles = [[] for _ in range(len(self._valid_motion_indices))]
        for line in self._motion_lines[1:]:
            # Skip empty lines
            if not line.strip():
                continue
            # Split line and remove whitespace
            values = [val.strip() for val in line.strip().split(",")]
            # Only process lines that have the correct number of values
            if len(values) == len(self._all_joint_names):
                for i, valid_idx in enumerate(self._valid_motion_indices):
                    if values[valid_idx]:  # Only convert non-empty strings
                        joint_angles[i].append(float(values[valid_idx]))

        # Convert to tensor (num_joints, num_timesteps)
        joint_angles = torch.tensor(joint_angles, dtype=torch.float, device=self.device)

        # Resample if needed
        if self.original_control_freq is not None and self.original_control_freq != self.control_freq:
            joint_angles = self._resample_motion(joint_angles)

        return joint_angles

    def _resample_motion(self, joint_angles):
        """Resample motion data to match target control frequency"""
        log_message(f"Resampling motion from {self.original_control_freq}Hz to {self.control_freq}Hz")

        # Calculate new length to match target control freq
        old_len = joint_angles.shape[1]
        duration = old_len / self.original_control_freq
        new_len = int(round(duration * self.control_freq))

        # Linear interpolation
        # Reshape to (1, num_joints, old_len)
        joint_angles = joint_angles.unsqueeze(0)
        joint_angles = torch.nn.functional.interpolate(joint_angles, size=new_len, mode="linear", align_corners=True)
        # Back to (num_joints, new_len)
        joint_angles = joint_angles.squeeze(0)

        return joint_angles

    def _log_state(self, time, command_positions, actual_positions, actual_velocities, actual_efforts):
        """Log current robot state"""
        # Log control data (command positions)
        with open(self.control_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CONTROL", time, command_positions.tolist()])

        # Log state motor data (actual state)
        with open(self.dof_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["STATE_MOTOR", time, actual_positions.tolist(), actual_velocities.tolist(), actual_efforts.tolist()]
            )

    def run_benchmark(self):
        """Run the benchmark for the current motion file"""
        num_timesteps = self.joint_angles.shape[1]

        log_message(f"Physics dt: {self.physics_dt}, Rendering dt: {self.physics_dt * self.divisor}")
        log_message(f"Expected log interval: {self.physics_dt * self.divisor}")

        # Reset simulation
        self.sim.reset()

        # Apply kp/kd overrides if provided
        self._apply_gain_overrides()

        # Buffer to the motion start position
        BUFFER_TIME = 5.0
        buffer_control_steps = int(BUFFER_TIME / self.control_dt)

        # Get initial joint positions and motion start positions
        initial_joint_positions = self.robot.data.joint_pos[0, self.joint_indices]
        motion_start_positions = self.joint_angles[:, 0]

        # Generate interpolated positions for smooth initialization
        log_message(f"Starting initialization phase with {BUFFER_TIME}s buffer...")

        for buffer_counter in range(buffer_control_steps * self.divisor):
            control_step = int(buffer_counter / self.divisor)

            # Update joint positions at control frequency
            if buffer_counter % self.divisor == 0:
                # Linear interpolation between initial and target positions
                alpha = control_step / buffer_control_steps
                interpolated_positions = (1 - alpha) * initial_joint_positions + alpha * motion_start_positions

                # Set interpolated positions
                joint_pos_target = self.robot.data.joint_pos.clone()
                joint_pos_target[0, self.joint_indices] = interpolated_positions
                self.robot.set_joint_position_target(joint_pos_target)
                self.robot.write_data_to_sim()

            self.sim.step()
            self.robot.update(self.physics_dt)
        buffer_end_time = self.sim.current_time

        log_message(
            f"Buffer completed in {BUFFER_TIME:.2f} seconds, {buffer_counter+1} physics steps. "
            f"Joint positions: {self.robot.data.joint_pos[0, self.joint_indices].cpu().numpy().tolist()}"
        )

        log_message("Initialization complete. Starting main motion...")

        log_message(
            f"physics_dt: {self.physics_dt} render_dt: {self.render_dt} "
            f"control_dt: {self.control_dt} divisor: {self.divisor}"
        )

        # Main motion execution
        for counter in range(num_timesteps * self.divisor):
            index = int(counter / self.divisor)
            adjusted_time = self.sim.current_time - buffer_end_time

            if index >= num_timesteps:
                break

            # Set joint positions
            if counter % self.divisor == 0:
                joint_pos_target = self.robot.data.joint_pos.clone()
                target_pos = self.joint_angles[:, index]
                joint_pos_target[0, self.joint_indices] = target_pos
                self.robot.set_joint_position_target(joint_pos_target)
                self.robot.write_data_to_sim()

            # Get and log current state
            command_positions = self.joint_angles[:, index]
            actual_positions = self.robot.data.joint_pos[0, self.joint_indices]
            actual_velocities = self.robot.data.joint_vel[0, self.joint_indices]

            # Get link incoming joint forces
            # 6D, including the active and passive components of the force
            link_forces = self.robot.root_physx_view.get_link_incoming_joint_force()
            # Extract torque components (last 3 elements of 6D force) and compute magnitude
            actual_efforts = torch.norm(link_forces[0, self.joint_indices, -3:], dim=1)

            self._log_state(
                time=adjusted_time,
                command_positions=command_positions.cpu().numpy(),
                actual_positions=actual_positions.cpu().numpy(),
                actual_velocities=actual_velocities.cpu().numpy(),
                actual_efforts=actual_efforts.cpu().numpy(),
            )

            self.sim.step()
            self.robot.update(self.physics_dt)

        log_message(
            f"Motion completed in {counter+1} physics steps. "
            f"Joint positions: {self.robot.data.joint_pos[0, self.joint_indices].cpu().numpy().tolist()}"
        )

        log_message(f"Benchmark complete for {self.motion_name}. Results saved to {self.sim_output_folder}")
