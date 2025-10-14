# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import csv
import ctypes
import os
from datetime import datetime
from pathlib import Path

import carb
import cv2
import numpy as np
import omni

# Isaac Sim modules - will be imported after SimulationApp initialization
# These will be set as module-level variables by the calling script
schemas = None
World = None
Articulation = None
add_reference_to_stage = None
create_new_stage = None
capture_viewport_to_buffer = None
create_prim = None


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
        self.control_freq = args.control_freq
        self.solver_type = args.solver_type
        self.record_video = args.record_video

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

        # Initialize video recording components as None
        self.viewport_api = None
        self.video_writer = None

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
        # Set up basic robot configuration
        self.prim_path = "/World/Robot"
        if self.robot_name == "h1_2":
            self.robot_usd_path = os.path.join(self.repo_path, "assets/h1_2/h1_2.usd")
            self.robot_offset = [0.0, 0.0, 1.1]
        elif self.robot_name == "g1":
            NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
            ISAAC_NUCLEUS_DIR = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
            self.robot_usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"
            self.robot_offset = [0.0, 0.0, 0.82]
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}. Supported robots: h1_2, g1")

        # Create and configure stage
        create_new_stage()
        stage = omni.usd.get_context().get_stage()

        # 4. Simulation config
        sim_params = {
            "gravity": [0, 0, -9.81],
            "solver_type": self.solver_type,
        }

        # Initialize world with physics configuration
        self.world = World(
            physics_dt=self.physics_dt, rendering_dt=self.render_dt, stage_units_in_meters=1.0, sim_params=sim_params
        )

        # Add default ground plane
        self.world.scene.add_default_ground_plane()

        # Load robot USD with offset
        create_prim(
            prim_path=self.prim_path, prim_type="Xform", usd_path=self.robot_usd_path, translation=self.robot_offset
        )
        log_message(f"Loaded robot {self.robot_name} from {self.robot_usd_path} with offset {self.robot_offset}")

        # Configure robot articulation view
        self.robot = Articulation(prim_paths_expr=self.prim_path, name=self.robot_name)
        self.world.scene.add(self.robot)

        # Configure articulation properties
        # TODO: configurable with config file or argument
        articulation_props = schemas.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=self.fix_root,
        )
        schemas.modify_articulation_root_properties(self.prim_path, articulation_props, stage)

        # Initialize physics simulation
        self.world.reset()
        for _ in range(5):
            self.world.step(render=False)

    def set_motion(self, motion_file, motion_name):
        """Set up for a new motion file."""
        self.motion_file = motion_file
        self.motion_name = motion_name

        # Load joint configuration data
        with open(self.motion_file) as file:
            first_line = file.readline().strip()

            all_joint_names = [name.strip() for name in first_line.split(",")]

            # Filter joints to only include valid ones
            if self.set_valid_joints:
                self.joint_names = [joint for joint in all_joint_names if joint in self.valid_joint_names]
            else:
                self.joint_names = all_joint_names

            if not self.joint_names:
                raise ValueError("No valid joints found in motion file that match the config file")

            log_message(f"Filtered {len(self.joint_names)} valid joints from motion file")

        # Map joint names to indices
        self.joint_indices = []
        for joint in self.joint_names:
            key = joint.split("/")[-1]
            index = self.robot.get_dof_index(dof_name=key)
            self.joint_indices.append(index)
        self.joint_indices = np.array(self.joint_indices)

        # Config joint controller and PD
        self._config_controller()

        # Initialize logging
        self._init_logger()

    def _config_controller(self):
        """Config joint mode and PD value"""
        self.robot.set_effort_modes("force", joint_indices=self.joint_indices)
        self.robot.switch_control_mode("position", joint_indices=self.joint_indices)

        stiffnesses, dampings = self.robot.get_gains(joint_indices=self.joint_indices)
        log_message(
            f"PD before configuration. Total joints: {len(self.joint_names)}. Kp={stiffnesses[0][0]},"
            f" Kd={dampings[0][0]}"
        )

        # Note: Currently just directly set to 50, 1 to follow upper body control in real world
        self.robot.set_gains(kps=50.0, kds=1.0, joint_indices=self.joint_indices)

        stiffnesses, dampings = self.robot.get_gains(joint_indices=self.joint_indices)
        log_message(
            f"PD after configuration. Total joints configured: {len(self.joint_names)}. Kp={stiffnesses[0][0]},"
            f" Kd={dampings[0][0]}"
        )

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
        with open(self.motion_file) as file:
            lines = file.readlines()

        # Parse joint names from first line, removing any whitespace
        all_joint_names = [name.strip() for name in lines[0].strip().split(",")]

        # Find indices of valid joints in the motion file
        valid_indices = []
        for joint in self.joint_names:
            idx = all_joint_names.index(joint)
            valid_indices.append(idx)

        # Parse joint values from subsequent lines, only for valid joints
        joint_angles = [[] for _ in range(len(valid_indices))]
        for line in lines[1:]:
            # Skip empty lines
            if not line.strip():
                continue
            # Split line and remove whitespace
            values = [val.strip() for val in line.strip().split(",")]
            # Only process lines that have the correct number of values
            if len(values) == len(all_joint_names):
                for i, valid_idx in enumerate(valid_indices):
                    if values[valid_idx]:  # Only convert non-empty strings
                        joint_angles[i].append(float(values[valid_idx]))

        return joint_angles, self.joint_names

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

    def _init_video_writer(self):
        video_path = f"{self.sim_output_folder}/{self.motion_name}.avi"
        log_message(f"Initializing video writer at: {video_path}")
        viewport = omni.ui.Workspace.get_window("Viewport")
        self.viewport_api = viewport.viewport_api
        frame_width = self.viewport_api.resolution[0]
        frame_height = self.viewport_api.resolution[1]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))

    def _capture_video_fn(self, *args, **kwargs):
        capsule, data_size, width, height = args[0], args[1], args[2], args[3]

        ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, None)
        buffer = ctypes.string_at(ptr, data_size)

        bytes_per_pixel = 4
        raw_array = np.frombuffer(buffer, dtype=np.uint8)
        image_array = raw_array.reshape((height, width, bytes_per_pixel))

        rgb_image_array = image_array[:, :, :3]
        rgb_image = cv2.cvtColor(rgb_image_array, cv2.COLOR_RGB2BGR)
        self.video_writer.write(rgb_image)

    def run_benchmark(self):
        """Run the benchmark for the current motion file"""
        # Load motion data
        log_message(f"Loading motion data from {self.motion_file}...")
        joint_angles, joint_names = self._load_motion()

        log_message(f"Physics dt: {self.physics_dt}, Rendering dt: {self.physics_dt * self.divisor}")
        log_message(f"Expected log interval: {self.physics_dt * self.divisor}")

        # Reset world time and counter
        self.world.reset()
        self._config_controller()  # need to reconfig after world.reset()

        # Buffer to the motion start position
        BUFFER_TIME = 5.0
        buffer_control_steps = int(BUFFER_TIME / self.control_dt)

        # Get initial joint positions and motion start positions
        initial_joint_positions = self.robot.get_joint_positions(joint_indices=self.joint_indices)[0]
        motion_start_positions = np.array([joint_angles[j][0] for j in range(len(self.joint_names))])

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
                target_pos = np.zeros((1, self.robot.num_dof), dtype=np.float32)
                for j, idx in enumerate(self.joint_indices):
                    target_pos[0, idx] = interpolated_positions[j]
                self.robot.set_joint_position_targets(target_pos)

            self.world.step(True)
        buffer_end_time = self.world.current_time

        log_message(
            f"Buffer completed in {BUFFER_TIME:.2f} seconds, {buffer_counter+1} physics steps. "
            f"Joint positions set to (rad): {list(self.robot.get_joint_positions(joint_indices=self.joint_indices)[0])}"
        )

        log_message("Initialization complete. Starting main motion...")

        log_message(
            f"physics_dt: {self.physics_dt} render_dt: {self.render_dt} "
            f"control_dt: {self.control_dt} divisor: {self.divisor} | "
            f"Solver type is : {self.world.get_physics_context().get_solver_type()}"
        )

        if self.record_video:
            self._init_video_writer()

        # 4. Main motion execution
        for counter in range(len(joint_angles[0]) * self.divisor):
            index = int(counter / self.divisor)
            adjusted_time = self.world.current_time - buffer_end_time

            if index >= len(joint_angles[0]):
                break

            # Set joint positions
            if counter % self.divisor == 0:
                target_pos = np.zeros((1, self.robot.num_dof), dtype=np.float32)
                for j, idx in enumerate(self.joint_indices):
                    target_pos[0, idx] = joint_angles[j][index]
                self.robot.set_joint_position_targets(target_pos)

            # Get and log current state
            command_positions = np.array([joint_angles[k][index] for k in range(len(self.joint_names))])
            actual_positions = self.robot.get_joint_positions(joint_indices=self.joint_indices)[0]
            actual_velocities = self.robot.get_joint_velocities(joint_indices=self.joint_indices)[0]
            actual_efforts = self.robot.get_measured_joint_efforts(joint_indices=self.joint_indices)[0]

            self._log_state(
                time=adjusted_time,
                command_positions=command_positions,
                actual_positions=actual_positions,
                actual_velocities=actual_velocities,
                actual_efforts=actual_efforts,
            )

            if self.record_video:
                capture_viewport_to_buffer(self.viewport_api, self._capture_video_fn)

            self.world.step(True)

        if self.record_video:
            self.video_writer.release()

        log_message(
            f"Motion completed in {counter+1} physics steps. Joint positions stopped at (rad):"
            f" {list(self.robot.get_joint_positions(joint_indices=self.joint_indices)[0])}"
        )

        log_message(f"Benchmark complete for {self.motion_name}. Results saved to {self.sim_output_folder}")

    def log_joint_properties(self):
        """Print a summary of joint properties including stiffness, damping, max velocity and max effort."""
        joint_names = self.robot._dof_names
        max_velocities = self.robot.get_joint_max_velocities()[0]
        max_efforts = self.robot.get_max_efforts()[0]
        stiffnesses, dampings = self.robot.get_gains()
        stiffnesses = stiffnesses[0]
        dampings = dampings[0]

        log_message("\nJoint Properties Summary:")
        log_message("=" * 110)
        log_message(f"{'Joint Name':<50} {'Stiffness':<15} {'Damping':<15} {'Max Velocity':<15} {'Max Effort':<15}")
        log_message("-" * 110)
        for i, name in enumerate(joint_names):
            log_message(
                f"{name:<50} {stiffnesses[i]:<15.2e} "
                f"{dampings[i]:<15.2e} {max_velocities[i]:<15.2e} "
                f"{max_efforts[i]:<15.2e}"
            )
        log_message("=" * 110)
