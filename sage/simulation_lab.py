# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
SAGE Simulation with IsaacLab and GapONet support.

This module provides a benchmark class similar to JointMotionBenchmark
but uses IsaacLab API and supports GapONet actuator.
"""

import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d

import isaaclab.sim.schemas as schemas
from isaaclab.actuators import GapONetActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


def log_message(message):
    """Print formatted log message with timestamp."""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[GapONetBenchmark][{current_time}] {message}")


class GapONetBenchmark:
    """SAGE-style benchmark using IsaacLab with GapONet actuator."""
    
    def __init__(self, args):
        self.repo_path = os.getcwd()
        
        # Store arguments
        self.robot_name = args.robot_name.lower()
        self.valid_joints_file = args.valid_joints_file
        self.output_folder = args.output_folder
        self.fix_root = args.fix_root
        self.headless = args.headless
        self.physics_freq = args.physics_freq
        self.render_freq = args.render_freq
        self.control_freq = args.control_freq
        self.original_control_freq = getattr(args, 'original_control_freq', None)
        self.kp = args.kp
        self.kd = args.kd
        
        # GapONet settings
        self.use_gaponet = args.use_gaponet
        self.gaponet_model = args.gaponet_model
        self.gaponet_action_scale = args.gaponet_action_scale
        
        # Realtime sync setting
        self.enable_realtime = getattr(args, 'enable_realtime', False)
        
        # Calculate time steps
        self.physics_dt = 1.0 / self.physics_freq
        self.control_dt = 1.0 / self.control_freq
        self.divisor = self.physics_freq // self.control_freq
        self.render_interval = max(1, self.physics_freq // self.render_freq)
        
        # Load valid joints
        self.valid_joint_names = []
        self._load_valid_joints()
        
        # Setup simulation
        self._setup_simulation()
    
    def _load_valid_joints(self):
        """Load valid joint names from config file."""
        if self.valid_joints_file is not None:
            config_file = self.valid_joints_file
        else:
            config_file = os.path.join(self.repo_path, f"configs/{self.robot_name}_valid_joints.txt")
        
        if not os.path.isfile(config_file):
            log_message(f"No valid joints file found: {config_file}, will use all joints")
            return
        
        with open(config_file) as file:
            self.valid_joint_names = [line.strip() for line in file if line.strip()]
        
        log_message(f"Loaded {len(self.valid_joint_names)} valid joint names")
    
    def _setup_simulation(self):
        """Setup IsaacLab simulation environment."""
        log_message("Setting up simulation...")
        log_message(f"Physics: {self.physics_freq}Hz, Render: {self.render_freq}Hz, Control: {self.control_freq}Hz")
        
        # Create simulation configuration
        sim_cfg = SimulationCfg(
            dt=self.physics_dt,
            render_interval=self.render_interval,
            gravity=(0.0, 0.0, -9.81),
        )
        
        # Create simulation context
        self.sim = SimulationContext(sim_cfg)
        
        # Set camera if not headless
        if not self.headless:
            self.sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 1.0])
        
        # Create robot configuration
        robot_cfg = self._create_robot_cfg()
        
        # Create ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        
        # Create scene
        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            """Scene configuration."""
            pass
        
        scene_cfg = MySceneCfg(num_envs=1, env_spacing=5.0)
        setattr(scene_cfg, "robot", robot_cfg)
        
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene.articulations["robot"]
        
        log_message(f"Loaded robot: {self.robot_name}")
        
        # Reset simulation
        self.sim.reset()
        
        log_message("Simulation setup complete")
    
    def _create_robot_cfg(self):
        """Create robot articulation configuration."""
        # Robot USD paths
        robot_usd_paths = {
            "h1_2": os.path.join(self.repo_path, "assets/h1_2/h1_2.usd"),
        }
        
        robot_offsets = {
            "h1_2": (0.0, 0.0, 1.1),
        }
        
        if self.robot_name not in robot_usd_paths:
            raise ValueError(f"Unsupported robot: {self.robot_name}")
        
        usd_path = robot_usd_paths[self.robot_name]
        offset = robot_offsets[self.robot_name]
        
        # Configure actuators (both use position control)
        if self.use_gaponet:
            log_message(f"Using GapONetActuator: {self.gaponet_model}")
            log_message("  → Position control: Neural network computes torques internally")
            actuators = {
                "gaponet_all": GapONetActuatorCfg(
                    joint_names_expr=[".*"],
                    network_file=self.gaponet_model,
                    action_scale=self.gaponet_action_scale,
                    stiffness=self.kp,
                    damping=self.kd,
                )
            }
        else:
            log_message("Using ImplicitActuator (PhysX native)")
            log_message("  → Position control: PhysX C++ PD controller (highly stable)")
            actuators = {
                "implicit_all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=self.kp,
                    damping=self.kd,
                )
            }
        
        # Configure articulation properties
        articulation_props = None
        if self.fix_root:
            articulation_props = schemas.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            )
        
        robot_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=False,
                articulation_props=articulation_props,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=offset,
                joint_pos={},
            ),
            actuators=actuators,
        )
        
        return robot_cfg
    
    def set_motion(self, motion_file, motion_name):
        """Load motion file and prepare for benchmark."""
        self.motion_file = motion_file
        self.motion_name = motion_name
        
        log_message(f"Loading motion: {motion_file}")
        
        # Load motion data
        self.joint_angles, self.joint_names = self._load_motion_data(motion_file)
        
        log_message(f"Loaded {len(self.joint_angles[0])} frames, {len(self.joint_names)} joints")
        
        # Map joint names to robot indices
        # Extract joint names from full paths (e.g., "/h1_2/torso_link/left_shoulder_pitch_joint" -> "left_shoulder_pitch_joint")
        joint_keys = [name.split("/")[-1] for name in self.joint_names]
        joint_indices_result, found_joint_names = self.robot.find_joints(joint_keys, preserve_order=True)
        
        # Ensure joint_indices is a torch.Tensor
        if isinstance(joint_indices_result, list):
            self.joint_indices = torch.tensor(joint_indices_result, dtype=torch.long, device=self.robot.device)
        else:
            self.joint_indices = joint_indices_result
        
        log_message(f"Mapped {len(self.joint_indices)} joints: {joint_keys}")
        log_message(f"Joint indices: {self.joint_indices.tolist()}")
        log_message(f"Joint indices dtype: {self.joint_indices.dtype}, device: {self.joint_indices.device}")
        
        # Debug: Print all joint info
        log_message(f"Total robot joints: {self.robot.num_joints}")
        log_message(f"Robot joint names: {self.robot.joint_names}")
        
        # Initialize logger
        self._init_logger()
    
    def _load_motion_data(self, motion_file):
        """Load motion data from file."""
        with open(motion_file) as file:
            lines = file.readlines()
        
        # Parse joint names from first line, removing any whitespace
        all_joint_names = [name.strip() for name in lines[0].strip().split(",")]
        
        # Filter joints to only include valid ones
        if self.valid_joint_names:
            joint_names = [joint for joint in all_joint_names if joint in self.valid_joint_names]
        else:
            joint_names = all_joint_names
        
        if not joint_names:
            raise ValueError("No valid joints found")
        
        log_message(f"Filtered {len(joint_names)} valid joints from motion file")
        
        # Find indices of valid joints in the motion file
        valid_indices = []
        for joint in joint_names:
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
        
        # Log original frames loaded
        original_frames = len(joint_angles[0]) if joint_angles else 0
        log_message(f"Loaded {original_frames} frames from file (before resampling)")
        
        # Convert to numpy array for processing
        joint_angles = np.array(joint_angles)
        log_message(f"Motion data sample (first joint, first frame): {joint_angles[0, 0]:.6f} rad")
        
        # Resample if original frequency is different from control frequency
        if self.original_control_freq is not None and self.original_control_freq != self.control_freq:
            log_message(f"Resampling motion from {self.original_control_freq}Hz to {self.control_freq}Hz")
            # joint_angles is already a numpy array from the conversion above
            # joint_angles shape is (num_joints, num_frames)
            num_joints, old_len = joint_angles.shape
            duration = old_len / self.original_control_freq
            new_len = int(round(duration * self.control_freq))

            # Create linspace for resample
            old_times = np.linspace(0, duration, old_len, endpoint=False)
            new_times = np.linspace(0, duration, new_len, endpoint=False)
            # Cap new time and fill with last value
            new_times = new_times[new_times <= old_times[-1]]
            new_times = np.append(new_times, old_times[-1])
            new_len = len(new_times)

            # Resample
            new_angles = np.zeros((num_joints, new_len), dtype=joint_angles.dtype)
            for i in range(num_joints):
                f = interp1d(old_times, joint_angles[i, :], kind="linear")
                new_angles[i, :] = f(new_times)
            joint_angles = new_angles
        
        # Convert to list format for compatibility
        if isinstance(joint_angles, np.ndarray):
            joint_angles = joint_angles.tolist()
        
        return joint_angles, joint_names
    
    def _init_logger(self):
        """Initialize CSV loggers."""
        output_folder = os.path.join(
            self.output_folder,
            self.robot_name,
            self.motion_name
        )
        os.makedirs(output_folder, exist_ok=True)
        
        self.control_file = os.path.join(output_folder, "control.csv")
        self.state_file = os.path.join(output_folder, "state_motor.csv")
        self.joint_list_file = os.path.join(output_folder, "joint_list.txt")
        
        # Write joint list
        with open(self.joint_list_file, "w") as f:
            for joint in self.joint_names:
                f.write(f"{joint}\n")
        
        # Initialize CSV files
        with open(self.control_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "positions"])
        
        with open(self.state_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        
        log_message(f"Logging to: {output_folder}")
    
    def run_benchmark(self):
        """Run the motion tracking benchmark."""
        log_message("Starting benchmark...")
        
        # Realtime sync setup
        if self.enable_realtime:
            log_message("⏱️  Realtime sync ENABLED - simulation will match wall clock time")
            realtime_start = time.perf_counter()
        else:
            log_message("🚀 Realtime sync DISABLED - simulation will run at maximum speed")
            realtime_start = None
        
        # Reset robot to default state before starting motion (like simulation_lab.py)
        log_message("Resetting robot to default state...")
        default_joint_pos = self.robot.data.default_joint_pos.clone()
        default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        self.scene.reset()  # Clear internal buffers
        
        # Step simulation a few times to stabilize
        for _ in range(10):
            if not self.use_gaponet:
                self.robot.set_joint_position_target(default_joint_pos)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.physics_dt)
        log_message("Robot reset complete.")
        
        # Buffer to the motion start position
        BUFFER_TIME = 5.0
        buffer_control_steps = int(BUFFER_TIME / self.control_dt)
        
        # Use default_joint_pos as initial position (safe baseline)
        initial_joint_positions = default_joint_pos[0, self.joint_indices].cpu().numpy()
        motion_start_positions = np.array([self.joint_angles[j][0] for j in range(len(self.joint_names))])
        
        log_message(f"Buffering {BUFFER_TIME}s from initial to first-frame position...")
        log_message(f"Initial: {initial_joint_positions}")
        log_message(f"Target: {motion_start_positions}")
        
        # Buffer loop: gradually move from initial to start position
        # Both GapONet and ImplicitActuator use position control
        for buffer_counter in range(buffer_control_steps * self.divisor):
            control_step = int(buffer_counter / self.divisor)
            
            # Update target at control frequency
            if buffer_counter % self.divisor == 0:
                alpha = control_step / buffer_control_steps
                q_interp = (1 - alpha) * initial_joint_positions + alpha * motion_start_positions
                
                # Create full joint position target (for all DOFs)
                full_target = self.robot.data.default_joint_pos.clone()
                full_target[0, self.joint_indices] = torch.from_numpy(q_interp).float().to(self.robot.device)
                
                # For GapONet actuator: provide global joint data before setting target
                if self.use_gaponet:
                    # Get current global joint states
                    current_joint_pos = self.robot.data.joint_pos
                    current_joint_vel = self.robot.data.joint_vel
                    
                    # Call set_global_joint_data on the actuator
                    for actuator in self.robot.actuators.values():
                        if hasattr(actuator, 'set_global_joint_data'):
                            actuator.set_global_joint_data(current_joint_pos, current_joint_vel, full_target)
                
                self.robot.set_joint_position_target(full_target)
            
            # Step simulation
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.physics_dt)
        
        buffer_end_time = self.sim.current_time
        # Check actual joint positions after buffer
        actual_pos_after_buffer = self.robot.data.joint_pos[0, self.joint_indices].cpu().numpy()
        log_message(f"Buffer done. t={buffer_end_time:.3f}s")
        log_message(f"Actual positions after buffer: {actual_pos_after_buffer}")
        log_message(f"Max absolute position: {abs(actual_pos_after_buffer).max():.6f}")
        
        # Safety check: if positions exploded during buffer, abort
        if abs(actual_pos_after_buffer).max() > 10.0:
            log_message(f"ERROR: Joint positions exploded during buffer phase!")
            log_message(f"Max position: {abs(actual_pos_after_buffer).max():.2f} rad")
            log_message(f"This indicates actuator instability. Aborting...")
            return
        
        # Reset realtime clock after buffer (for accurate motion timing)
        if self.enable_realtime:
            motion_start_time = time.perf_counter()
            log_message("🎬 Starting realtime motion playback...")
        
        # Main motion loop
        log_message("Starting main motion playback...")
        num_steps = len(self.joint_angles[0])
        
        # Initialize command_pos with first frame
        command_pos = np.array([self.joint_angles[j][0] for j in range(len(self.joint_names))])
        
        # Both GapONet and ImplicitActuator use position control
        # GapONetActuator receives position targets and internally computes torques using neural network
        # ImplicitActuator uses PhysX native PD controller
        for step in range(num_steps * self.divisor):
            index = int(step / self.divisor)
            
            if index >= num_steps:
                break
            
            # Update command at control frequency
            if step % self.divisor == 0:
                command_pos = np.array([self.joint_angles[j][index] for j in range(len(self.joint_names))])
                
                # Create full joint position target (for all DOFs)
                full_target = self.robot.data.default_joint_pos.clone()
                full_target[0, self.joint_indices] = torch.from_numpy(command_pos).float().to(self.robot.device)
                
                # For GapONet actuator: provide global joint data before setting target
                if self.use_gaponet:
                    # Get current global joint states
                    current_joint_pos = self.robot.data.joint_pos
                    current_joint_vel = self.robot.data.joint_vel
                    
                    # Call set_global_joint_data on the actuator
                    for actuator in self.robot.actuators.values():
                        if hasattr(actuator, 'set_global_joint_data'):
                            actuator.set_global_joint_data(current_joint_pos, current_joint_vel, full_target)
                
                self.robot.set_joint_position_target(full_target)
            
            # Step simulation
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.physics_dt)
            
            # Realtime sync
            if self.enable_realtime:
                expected_motion_time = (step + 1) * self.physics_dt
                elapsed_motion_time = time.perf_counter() - motion_start_time
                if elapsed_motion_time < expected_motion_time:
                    time.sleep(expected_motion_time - elapsed_motion_time)
            
            # Log state
            adjusted_time = self.sim.current_time - buffer_end_time
            actual_pos = self.robot.data.joint_pos[0, self.joint_indices].cpu().numpy()
            actual_vel = self.robot.data.joint_vel[0, self.joint_indices].cpu().numpy()
            actual_eff = self.robot.data.applied_torque[0, self.joint_indices].cpu().numpy()
            self._log_state(adjusted_time, command_pos, actual_pos, actual_vel, actual_eff)
            
            if (step + 1) % 100 == 0:
                log_message(f"Progress: {step + 1}/{num_steps * self.divisor} steps")
        
        log_message(f"Motion complete! Results saved to {self.output_folder}")
    
    def _log_state(self, time, command_pos, actual_pos, actual_vel, actual_eff):
        """Log current state to CSV."""
        with open(self.control_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CONTROL", time, command_pos.tolist()])
        
        with open(self.state_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["STATE_MOTOR", time, actual_pos.tolist(), actual_vel.tolist(), actual_eff.tolist()])
    
    def log_joint_properties(self):
        """Log joint properties summary (compatible with SAGE)."""
        log_message("Joint properties summary:")
        log_message(f"  Robot: {self.robot_name}")
        log_message(f"  Control frequency: {self.control_freq} Hz")
        log_message(f"  Control mode: Position control (set_joint_position_target)")
        log_message(f"  PD gains: kp={self.kp}, kd={self.kd}")
        if self.use_gaponet:
            log_message(f"  Actuator: GapONetActuatorCfg")
            log_message(f"    - Neural network model: {self.gaponet_model}")
            log_message(f"    - Network computes torques from position targets")
        else:
            log_message(f"  Actuator: ImplicitActuatorCfg")
            log_message(f"    - PhysX native PD controller (C++, highly stable)")
    
    def cleanup(self):
        """Clean up simulation resources."""
        try:
            log_message("Cleaning up resources...")
            
            # Delete scene
            if hasattr(self, 'scene'):
                del self.scene
            
            # Stop simulation
            if hasattr(self, 'sim'):
                self.sim._timeline.stop()
                self.sim.clear_all_callbacks()
                self.sim.clear_instance()
            
            log_message("Cleanup complete")
        except Exception as e:
            log_message(f"Warning during cleanup: {e}")

