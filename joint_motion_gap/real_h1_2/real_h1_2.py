import os
import sys
import multiprocessing
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc
import csv

import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)
import time
from web_hand import *
import atexit
import socket
from dynamixel_client import DynamixelClient
import h5py
from datetime import datetime
from gamepad import Gamepad, parse_remote_data


HUMANOID_XML = 'your_path/h1_2.xml'
FILE_PATH = 'your_path/motions/'
FILE_LIST = os.listdir(FILE_PATH)
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
HW_DOF = 27

WALK_STRAIGHT = False
LOG_DATA = False
USE_GRIPPPER = False
NO_MOTOR = False

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def euler_from_quat(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]
    y = quat_angle[:,1]
    z = quat_angle[:,2]
    w = quat_angle[:,3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def isWeak(motor_index):
    return motor_index == 10 or motor_index == 11 or motor_index == 12 or motor_index == 13 or \
        motor_index == 14 or motor_index == 15 or motor_index == 16 or motor_index == 17 or \
        motor_index == 18 or motor_index == 19

def cleanup(ser_l,ser_r):
    print("closing ports")
    ser_r.flush()
    ser_r.reset_input_buffer()
    ser_r.reset_output_buffer()
    ser_l.flush()
    ser_l.reset_input_buffer()
    ser_l.reset_output_buffer()
    ser_r.close()
    ser_l.close()

class H1_2():
    def __init__(self,task='stand'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = task
        
        self.num_envs = 1 
        self.num_observations = 76  # 76 *6 = 456
        self.num_actions = 12
        self.num_privileged_obs = None
        self.obs_context_len= 6
        
        self.scale_lin_vel = 2.0
        self.scale_ang_vel = 0.25
        self.scale_orn = 1.0
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.scale_action = 0.25
        
        arm_dgain = 1
        leg_pgain = 200
        leg_dgain = 5
        knee_pgain = 200
        knee_dgain = 5

        self.p_gains = np.array([leg_pgain,leg_pgain,knee_pgain,leg_pgain,leg_pgain,knee_pgain,leg_pgain,leg_pgain,leg_pgain,0,ankle_pgain,ankle_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain],dtype=np.float64)
        self.d_gains = np.array([leg_dgain,leg_dgain,knee_dgain,leg_dgain,leg_dgain,knee_dgain,leg_dgain,leg_dgain,leg_dgain,0,ankle_dgain,ankle_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain],dtype=np.float64)
        self.joint_limit_lo = [-0.43, -3.14, -0.43, -0.26, -np.inf, -np.inf, -0.43, -3.14, -3.14, -0.24,-np.inf,-np.inf,-2.618,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558]
        self.joint_limit_hi = [0.43, 2.5, 3.14, 2.05, np.inf, np.inf, 0.43, 2.5, 0.43, 2.0, np.inf, np.inf, 2.618, 2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558]
        self.default_dof_pos_np = np.array([
                                            0.0, #left hip yaw
                                            -0.16, #left hip pitch
                                            0.0, #left hip roll
                                            0.36, #left knee
                                            -0.2, #left ankle pitch
                                            0, #left ankle roll
                                            0.0, #right hip yaw
                                            -0.16, #right hip pitch
                                            0.0, #right hip roll
                                            0.36, #right knee
                                            -0.2, #right ankle pitch
                                            0, #right ankle roll
                                            0, #waist
                                            0.0,
                                            0,
                                            0,
                                            0.0,
                                            0,
                                            0,
                                            0,
                                            0.0,
                                            0,
                                            0,
                                            0.0,
                                            0,
                                            0,
                                            0,
                                            ])
        
        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)


        # prepare osbervations buffer
        self.obs_buf = torch.zeros(1, self.num_observations, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_history_buf = torch.zeros(1, self.obs_context_len, self.num_observations, dtype=torch.float, device=self.device, requires_grad=False)


def load_reference_motions():
    """Load upper body motion data from the pickle file and return upper body actions
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (upper_body_actions, motion_data, motion_info)
    """
    global count
    
    motion_file_path = os.path.join(FILE_PATH, FILE_LIST[count]) 
    if not os.path.exists(motion_file_path):
        print(f"Warning: Motion file not found at {motion_file_path}")
        return None, None, None
    
    with open(motion_file_path, 'rb') as f:
        motion_data = joblib.load(f)

    try:    
        # Extract the DOF data from the first key
        first_key = list(motion_data.keys())[0]
        dof_data = motion_data[first_key]['dof']  # Shape: (610, 27)
        fps = motion_data[first_key]['fps']  # 30 fps
        
        # Interpolate from 30Hz to 50Hz
        target_fps = 50
        original_frames = dof_data.shape[0]
        target_frames = int(original_frames * target_fps / fps)
        
        # Create time arrays for interpolation
        original_time = np.linspace(0, (original_frames - 1) / fps, original_frames)
        target_time = np.linspace(0, (original_frames - 1) / fps, target_frames)
        
        # Interpolate each DOF dimension
        interpolated_dof_data = np.zeros((target_frames, dof_data.shape[1]), dtype=np.float32)
        for dof_idx in range(dof_data.shape[1]):
            interpolator = interp1d(original_time, dof_data[:, dof_idx], kind='linear', 
                                    bounds_error=False, fill_value='extrapolate')
            interpolated_dof_data[:, dof_idx] = interpolator(target_time)
        
        # Update motion info with new parameters
        motion_info = {
            'total_frames': target_frames,
            'fps': target_fps,
            'frame_dt': 1.0 / target_fps,
            'frame_counter': 0,
            'motion_playing': False,  # Flag to control motion playback
            'motion_finished': False,  # Flag to indicate motion completion
            'return_to_zero': False,   # Flag to control return to zero
            'return_counter': 0,       # Counter for return to zero phase
            'return_duration': 1.0,    # Duration to return to zero (seconds)
            'return_steps': 50,
            'last_motion_frame': None,  # Store the last frame position
            'file_num': count,
        }
        
        print(f"Original motion: {original_frames} frames at {fps}Hz")
        print(f"Interpolated to: {target_frames} frames at {target_fps}Hz")
        print(f"Motion duration: {(target_frames - 1) / target_fps:.3f}s")
        print(f"Return to zero duration: {motion_info['return_duration']}s ({motion_info['return_steps']} steps)")
        
        return interpolated_dof_data, motion_info

    except Exception as e:
        print(f"Error loading motion file: {e}")
        return None, None, None

def get_upper_body_actions(dof_data, motion_info, default_upper_pos, action_scale):
    """Get upper body actions for the current timestep
    
    Args:
        dof_data: Motion data array
        motion_info: Motion information dictionary
        default_upper_pos: Default positions for upper body joints
        action_scale: Action scaling factor
        
    Returns:
        numpy.ndarray: Upper body actions
    """
    if motion_info is None or dof_data is None:
        return np.zeros(15, dtype=np.float32)
    
    # If motion is not playing, return zeros
    if not motion_info['motion_playing']:
        return np.zeros(15, dtype=np.float32)
    
    # If motion is finished and we need to return to zero
    if motion_info['motion_finished'] and motion_info['return_to_zero']:
        motion_info['return_counter'] += 1
        
        if motion_info['return_counter'] >= motion_info['return_steps']:
            # Return to zero complete
            motion_info['return_to_zero'] = False
            motion_info['motion_playing'] = False
            motion_info['motion_finished'] = False
            motion_info['frame_counter'] = 0
            print("Return to zero complete, motion stopped")
            return np.zeros(15, dtype=np.float32)
        
        # Interpolate from last position to zero
        progress = motion_info['return_counter'] / motion_info['return_steps']
        if motion_info['last_motion_frame'] is not None:
            current_target = motion_info['last_motion_frame'] * (1 - progress)
            upper_body_actions = (current_target - default_upper_pos) * 0.7
            return upper_body_actions
        else:
            return np.zeros(15, dtype=np.float32)
    
    # Motion phase: play the sequence normally
    current_frame_dof = dof_data[motion_info['frame_counter']]
    
    # Increment frame counter and check if motion is complete
    motion_info['frame_counter'] += 1
    if motion_info['frame_counter'] >= motion_info['total_frames']:
        # Motion finished, start return to zero phase
        motion_info['motion_finished'] = True
        motion_info['return_to_zero'] = True
        motion_info['return_counter'] = 0
        # Store the last frame position for smooth return
        motion_info['last_motion_frame'] = current_frame_dof[12:27]
        print("Motion sequence complete, returning to zero position")
        return np.zeros(15, dtype=np.float32)
    
    # Extract upper body DOFs (indices 12-26, which correspond to the last 15 DOFs)
    upper_body_dof = current_frame_dof[12:27]  # Shape: (15,)
    # Calculate actions: (target_position - default_position) / action_scale
    upper_body_actions = (upper_body_dof - default_upper_pos) * 0.7
    return upper_body_actions

class DeployNode(Node):
    def __init__(self,task='stand'):
        super().__init__("deploy_node")  # type: ignore
        self.task = task
        if self.task not in ['stand','stand_w_waist','wb','squat']:
            self.get_logger().info("Invalid task")
            raise SystemExit
        
        # init subcribers & publishers
        self.joy_stick_sub = self.create_subscription(WirelessController, "wirelesscontroller", self.joy_stick_cb, 1)
        self.joy_stick_sub  # prevent unused variable warning
        self.lowlevel_state_sub = self.create_subscription(LowState, "lowstate", self.lowlevel_state_cb, 1)  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning

        self.low_state = LowState()
        self.joint_pos = np.zeros(HW_DOF)
        self.joint_vel = np.zeros(HW_DOF)

        self.motor_pub = self.create_publisher(LowCmd, "lowcmd", 1)
        self.motor_pub_freq = 50
        self.cmd_msg = LowCmd()

        # init motor command
        self.motor_cmd = []
        for id in range(HW_DOF):
            if isWeak(id):
                mode = 0x01
            else:
                mode = 0x0A
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=mode, reserve=[0,0,0])
            self.motor_cmd.append(cmd)
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        # init policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_policy()
        self.prev_action = np.zeros(self.env.num_actions)
        self.start_policy = False

        # init multiprocess values
        self.wrist_l = multiprocessing.Value('f', 0.)
        self.wrist_r = multiprocessing.Value('f', 0.)
        self.body_ref = multiprocessing.Array('f', [0]*24) #[0]*12+[0.3]+[0]*3+[-0.3]+[0]*7)
        self.hand_ref = multiprocessing.Array('f', [160]*6+[0]+[160]*7+[0]+[160])

        # standing up
        self.get_logger().info("Standing up")
        self.stand_up = False
        self.stand_up = True

        # start
        self.start_time = time.monotonic()
        self.get_logger().info("Press L2 to start policy")
        self.get_logger().info("Press L1 for emergent stop")
        self.init_buffer = 0
        self.foot_contact_buffer = []
        self.time_hist = []
        self.obs_time_hist = []
        self.angle_hist = []
        self.action_hist = []
        self.dof_pos_hist = []
        self.dof_vel_hist = []
        self.imu_hist = []
        self.ang_vel_hist = []
        self.foot_contact_hist = []
        self.tau_hist = []
        self.obs_hist = []

    def reindex_urdf2hw(self, vec):
        vec = np.array(vec)
        assert len(vec)==19, "wrong dim for reindex"
        hw_vec = vec[[6, 7, 8, 1, 2, 3, 10, 0, 5, 0, 4, 9, 15, 16, 17, 18, 11, 12, 13, 14]]
        hw_vec[9] = 0
        return hw_vec

    def reindex_hw2urdf(self, vec):
        vec = np.array(vec)
        assert len(vec)==20, "wrong dim for reindex"
        return vec[[7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15]]
        
    ##############################
    # subscriber callbacks
    ##############################

    def joy_stick_cb(self, msg):
        if msg.keys == 2:  # L1: emergency stop
            self.get_logger().info("Emergency stop")
            self.set_gains(np.array([0.0]*HW_DOF),self.env.d_gains)
            self.set_motor_position(q=self.env.default_dof_pos_np)
            if LOG_DATA:
                print("Saving data")
                np.savez('captured_data.npz', action=np.array(self.action_hist), dof_pos=np.array(self.dof_pos_hist),
                        dof_vel=np.array(self.dof_vel_hist),imu=np.array(self.imu_hist),ang_vel=np.array(self.ang_vel_hist),
                        tau=np.array(self.tau_hist), obs=np.array(self.obs_hist))
            raise SystemExit
        if msg.keys == 32:  # L2: start policy
            if self.stand_up:
                self.get_logger().info("Start policy")
                self.start_policy = True
                self.policy_start_time = time.monotonic()
            else:
                self.get_logger().info("Wait for standing up first")

    def lowlevel_state_cb(self, msg: LowState, output_folder):
        # imu data
        imu_data = msg.imu_state
        self.msg_tick = msg.tick/1000
        self.roll, self.pitch, self.yaw = imu_data.rpy
        self.obs_ang_vel = np.array(imu_data.gyroscope)*self.env.scale_ang_vel
        self.obs_imu = np.array([self.roll, self.pitch])*self.env.scale_orn

        # termination condition
        r_threshold = abs(self.roll) > 0.5
        p_threshold = abs(self.pitch) > 0.5
        if r_threshold or p_threshold:
            self.get_logger().warning("Roll or pitch threshold reached")

        # wireless_remote btn
        global count
        joystick_data = msg.wireless_remote
        parsed_data = parse_remote_data(joystick_data)
        self.gamepad.update(parsed_data)
        
        if self.gamepad.L1.pressed:
            print(f'Policy start!')
            self.start_policy = True
        if self.gamepad.L2.pressed:
            self.start_policy = False
            self.Emergency_stop = True
            print(f'Manual emergency stop!!!')
        if self.gamepad.R1.pressed: # R1 is pressed
            self.get_logger().info("Program exiting")
            self.stop = True
        
        if self.gamepad.A.pressed: # A is pressed
            # start upper body motion playback and data recording
            if not self.start_load_upper_body_motion:
                self.get_logger().info("Starting upper body motion playback and data recording")
                self.start_load_upper_body_motion = True
                self.upper_body_motion_start_time = time.monotonic()
                self.is_recording = True
                
                # Clear previous data
                self.time_hist.clear()
                self.action_hist.clear()
                self.dof_pos_hist.clear()
                self.dof_vel_hist.clear()
                self.imu_hist.clear()
                self.ang_vel_hist.clear()
                self.tau_hist.clear()
                self.obs_hist.clear()
                self.temp_hist.clear()
                
                if self.dof_data is None or self.motion_info is None:
                    self.dof_data, self.motion_info = load_reference_motions()
                    if self.dof_data is not None and self.motion_info is not None:
                        # Start motion playback
                        self.motion_info['motion_playing'] = True
                        self.motion_info['frame_counter'] = 0
                        print("Motion playback started!")
                    else:
                        self.get_logger().warning("Failed to load motion data")
                        self.start_load_upper_body_motion = False
                        self.is_recording = False
                else:
                    # Restart motion playback
                    self.motion_info['motion_playing'] = True
                    self.motion_info['frame_counter'] = 0
                    print("Motion playback restarted!")
            else:
                self.get_logger().info("Motion already playing or loading")
                
        if self.gamepad.B.pressed: # B is pressed
            # Save data and change to next motion file
            if self.is_recording and self.start_load_upper_body_motion:
                # Save current data before switching
                self.get_logger().info("Saving data and switching motion")
                
                motion_name = FILE_LIST[count] if self.motion_info is not None else "unknown_motion"
                motion_name = os.path.splitext(motion_name)[0]
                
                # Create output directory
                output_dir = output_folder
                motion_dir = os.path.join(output_dir, f'{motion_name}')
                os.makedirs(motion_dir, exist_ok=True)
                
                # Define joint names (assuming 27 joints based on HW_DOF)
                robot_joint_names = [
                    'left_hip_yaw', 'left_hip_pitch', 'left_hip_roll', 'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
                    'right_hip_yaw', 'right_hip_pitch', 'right_hip_roll', 'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
                    'waist', 'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow', 
                    'left_wrist_yaw', 'left_wrist_pitch', 'left_wrist_roll',
                    'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow',
                    'right_wrist_yaw', 'right_wrist_pitch', 'right_wrist_roll'
                ]
                
                # 1. Save joint_list.txt
                joint_list_file = os.path.join(motion_dir, "joint_list.txt")
                with open(joint_list_file, "w") as f:
                    for joint_name in robot_joint_names:
                        f.write(f"{joint_name}\n")
                
                # 2. Save control.csv (command data)
                control_file = os.path.join(motion_dir, "control.csv")
                with open(control_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["type", "timestamp", "positions"])
                    
                    for i, (time_val, action_val) in enumerate(zip(self.time_hist, self.action_hist)):
                        # Convert action to positions (you may need to adjust this based on your action format)
                        if len(action_val) == 27:  # Full joint positions
                            positions = action_val
                        else:
                            # If action is only part of the joints, pad with zeros or default values
                            positions = np.zeros(27)
                            positions[:len(action_val)] = action_val
                        
                        writer.writerow(["CONTROL", int(time_val * 1e5), positions.tolist()])
                
                # 3. Save event.csv
                event_file = os.path.join(motion_dir, "event.csv")
                with open(event_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["type", "timestamp", "event"])
                    
                    if len(self.time_hist) > 0:
                        writer.writerow(["EVENT", 0, "CONTROL_MODE_ENABLE"])
                        writer.writerow(["EVENT", int(self.time_hist[0] * 1e5), "MOTION_START"])
                        writer.writerow(["EVENT", int(self.time_hist[-1] * 1e5), "MOTION_END"])
                
                # 4. Save state_motor.csv (actual state data)
                state_motor_file = os.path.join(motion_dir, "state_motor.csv")
                with open(state_motor_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
                    
                    for i in range(len(self.time_hist)):
                        timestamp = self.time_hist[i]
                        
                        # Get position, velocity, and torque data
                        positions = self.dof_pos_hist[i] if i < len(self.dof_pos_hist) else np.zeros(27)
                        velocities = self.dof_vel_hist[i] if i < len(self.dof_vel_hist) else np.zeros(27)
                        torques = self.tau_hist[i] if i < len(self.tau_hist) else np.zeros(27)
                        
                        # Ensure arrays have correct length
                        if len(positions) != 27:
                            pos_temp = np.zeros(27)
                            pos_temp[:len(positions)] = positions
                            positions = pos_temp
                        
                        if len(velocities) != 27:
                            vel_temp = np.zeros(27)
                            vel_temp[:len(velocities)] = velocities
                            velocities = vel_temp
                        
                        if len(torques) != 27:
                            tau_temp = np.zeros(27)
                            tau_temp[:len(torques)] = torques
                            torques = tau_temp
                        
                        writer.writerow(["STATE_MOTOR", int(timestamp * 1e5), 
                                    positions.tolist(), velocities.tolist(), torques.tolist()])
                
                # 5. Save state_base.csv (IMU and base state data)
                state_base_file = os.path.join(motion_dir, "state_base.csv")
                with open(state_base_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["type", "timestamp", "imu_quat", "imu_gyro", "imu_accel", "imu_rpy", "foot_force"])
                    
                    for i in range(len(self.time_hist)):
                        timestamp = self.time_hist[i]
                        
                        # Get IMU data
                        imu_data = self.imu_hist[i] if i < len(self.imu_hist) else [0, 0]
                        ang_vel_data = self.ang_vel_hist[i] if i < len(self.ang_vel_hist) else [0, 0, 0]
                        
                        # Format IMU quaternion (assuming we have roll, pitch, and need to calculate quaternion)
                        if len(imu_data) >= 2:
                            roll, pitch = imu_data[0], imu_data[1]
                            yaw = imu_data[2] if len(imu_data) > 2 else 0
                            
                            # Convert RPY to quaternion (simple approximation)
                            # For more accurate conversion, you might need a proper RPY to quaternion function
                            import math
                            cy = math.cos(yaw * 0.5)
                            sy = math.sin(yaw * 0.5)
                            cp = math.cos(pitch * 0.5)  
                            sp = math.sin(pitch * 0.5)
                            cr = math.cos(roll * 0.5)
                            sr = math.sin(roll * 0.5)
                            
                            w = cr * cp * cy + sr * sp * sy
                            x = sr * cp * cy - cr * sp * sy
                            y = cr * sp * cy + sr * cp * sy
                            z = cr * cp * sy - sr * sp * cy
                            
                            imu_quat = [w, x, y, z]  # Note: format might be [w,x,y,z] or [x,y,z,w] depending on your system
                        else:
                            imu_quat = [1.0, 0.0, 0.0, 0.0]  # Default quaternion
                        
                        # Format gyroscope data
                        if len(ang_vel_data) >= 3:
                            imu_gyro = [ang_vel_data[0], ang_vel_data[1], ang_vel_data[2]]
                        else:
                            imu_gyro = [0.0, 0.0, 0.0]
                        
                        # IMU acceleration (you might need to get this from your IMU data)
                        # For now, using placeholder values - you should replace this with actual accelerometer data
                        imu_accel = [0.0, 0.0, 9.8]  # Default gravity vector
                        
                        # Format RPY and foot force as "not_valid" for now
                        # You can replace these with actual values if available
                        imu_rpy = "not_valid"
                        foot_force = "not_valid"
                        
                        # Format arrays as strings with brackets
                        imu_quat_str = f"[{','.join([f'{x:.6f}' for x in imu_quat])}]"
                        imu_gyro_str = f"[{','.join([f'{x:.6f}' for x in imu_gyro])}]"
                        imu_accel_str = f"[{','.join([f'{x:.6f}' for x in imu_accel])}]"
                        
                        writer.writerow(["StateB", int(timestamp * 1e5), imu_quat_str, imu_gyro_str, imu_accel_str, imu_rpy, foot_force])
                
                print(f"CSV files saved to directory: {motion_dir}")
                print(f"  - {joint_list_file}")
                print(f"  - {control_file}")
                print(f"  - {event_file}")
                print(f"  - {state_motor_file}")
                print(f"  - {state_base_file}")
                
                # Stop recording and motion
                self.is_recording = False
                self.start_load_upper_body_motion = False
                if self.motion_info is not None:
                    self.motion_info['motion_playing'] = False
                    
                    
            # Change to next motion file
            count = (count + 1) % len(FILE_LIST)
            
            # Load new motion data
            new_dof_data, new_motion_info = load_reference_motions()
            if new_dof_data is not None and new_motion_info is not None:
                # Update the global variables
                global global_dof_data, global_motion_info
                global_dof_data = new_dof_data
                global_motion_info = new_motion_info
                global_motion_info['file_num'] = count  # Preserve the file number
                
                # Update local variables
                self.dof_data = global_dof_data
                self.motion_info = global_motion_info
                
                print(f"Switched to motion: {FILE_LIST[count]}")
                print(f"Motion file: {FILE_LIST[count]}")
            else:
                print(f"Failed to load motion file: {FILE_LIST[count]}")
                # Revert to previous file if loading failed
                count = (count - 1) % len(FILE_LIST)
            
        if self.gamepad.R2.pressed: # R2 is pressed
            # Reset motion playback
            if self.motion_info is not None:
                self.motion_info['motion_playing'] = False
                self.motion_info['motion_finished'] = False
                self.motion_info['return_to_zero'] = False
                self.motion_info['return_counter'] = 0
                self.motion_info['frame_counter'] = 0
                self.start_load_upper_body_motion = False
                self.is_recording = False
                print("Motion reset to beginning")

        # motor data
        self.joint_tau = [msg.motor_state[i].tau_est for i in range(HW_DOF)]
        self.joint_pos = [msg.motor_state[i].q for i in range(HW_DOF)]
        self.obs_joint_pos = (np.array(self.joint_pos) - self.env.default_dof_pos_np) * self.env.scale_dof_pos
        joint_vel = [msg.motor_state[i].dq for i in range(HW_DOF)]
        self.obs_joint_vel = np.array(joint_vel) * self.env.scale_dof_vel

        
    ##############################
    # motor commands
    ##############################

    def set_gains(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd
        for i in range(HW_DOF):
            self.motor_cmd[i].kp = kp[i]  #*0.5
            self.motor_cmd[i].kd = kd[i]  #*3

    def set_motor_position(
        self,
        q: np.ndarray,
    ):
        for i in range(HW_DOF):
            self.motor_cmd[i].q = q[i]
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.cmd_msg.crc = get_crc(self.cmd_msg)

    ##############################
    # deploy policy
    ##############################
    def init_policy(self):
        self.get_logger().info("Preparing policy")
        faulthandler.enable()

        # prepare environment
        self.env = H1(task=self.task)

        # load policy
        file_pth = os.path.dirname(os.path.realpath(__file__))
        self.policy = torch.jit.load(os.path.join(file_pth, "./ckpt/policy.pt"), map_location=self.env.device)  #0253 396
        self.policy.to(self.env.device)
        actions = self.policy(self.env.obs_history_buf.detach())  # first inference takes longer time

        # init p_gains, d_gains, torque_limits
        for i in range(HW_DOF):
            self.motor_cmd[i].q = self.env.default_dof_pos[0][i].item()
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # self.env.p_gains[i]  # 30
            self.motor_cmd[i].kd = 0.0  # float(self.env.d_gains[i])  # 0.6
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.angles = self.env.default_dof_pos

    def get_retarget(self):
        target_jt = self.body_ref[:19]
        if self.task == "stand":
            reference_pose = self.reindex_urdf2hw(target_jt) + self.env.default_dof_pos_np
            reference_pose_clip = np.clip(reference_pose, self.env.joint_limit_lo, self.env.joint_limit_hi)
            reference_pose_clip[:12] = self.env.default_dof_pos_np[:12]
            return reference_pose_clip
        elif self.task == "stand_w_waist":
            target_jt[10]*=2
            pose = np.array(self.body_ref[19:])
            reference_pose = self.reindex_urdf2hw(target_jt) + self.env.default_dof_pos_np
            reference_pose[:6] = self.env.default_dof_pos_np[:6]
            reference_pose[7:12] = self.env.default_dof_pos_np[7:12]
            return reference_pose, pose  # no clip for the target
        elif self.task == "wb" or self.task == "squat":
            pose = np.array(self.body_ref[19:])
            reference_pose = self.reindex_urdf2hw(target_jt) + self.env.default_dof_pos_np
        return reference_pose, pose  # no clip for the target

    @torch.no_grad()
    def main_loop(self, output_folder):
        from functools import partial
        
        self.lowlevel_state_sub = self.create_subscription(
            LowState, "lowstate", partial(self.lowlevel_state_cb, output_folder=output_folder), 1
        ) # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning

        self.low_state = LowState()
        self.joint_pos = np.zeros(HW_DOF)
        self.joint_vel = np.zeros(HW_DOF)

        self.motor_pub = self.create_publisher(LowCmd, "lowcmd_buffer", 1)
        self.motor_pub_freq = 50
        self.dt = 1/self.motor_pub_freq

        self.cmd_msg = LowCmd()

        self.cmd_msg.mode_pr = 0
        self.cmd_msg.mode_machine = 6

        # init motor command
        self.motor_cmd = []
        for id in range(HW_DOF):
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=1, reserve=0)
            self.motor_cmd.append(cmd)
        for id in range(HW_DOF, 35):
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=0, reserve=0)
            self.motor_cmd.append(cmd)
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        # init policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_policy()
        self.prev_action = np.zeros(self.env.num_actions)
        self.start_policy = False

        # standing up
        self.get_logger().info("Standing up")
        self.stand_up = False
        self.stand_up = True

        # commands 
        self.lin_vel_deadband = 0.1
        self.ang_vel_deadband = 0.1
        self.move_by_wireless_remote = True
        self.cmd_px_range = [0.1, 2.5]
        self.cmd_nx_range = [0.1, 2]
        self.cmd_py_range = [0.1, 0.5]
        self.cmd_ny_range = [0.1, 0.5]
        self.cmd_pyaw_range = [0.2, 1.0]
        self.cmd_nyaw_range = [0.2, 1.0]
        self.commands = np.array([0, 0, 0, 0, 0.95], dtype= np.float32)
        self.commands_scale = np.array([2.0, 2.0, 0.25], dtype= np.float32)

        # Upper body motion control
        self.start_load_upper_body_motion = False
        self.upper_body_motion_start_time = None
        self.dof_data = None
        self.motion_info = None
        
        # Data recording control
        self.is_recording = False  # 控制是否正在记录数据
        
        # Initialize global motion data
        global global_dof_data, global_motion_info
        if global_dof_data is None or global_motion_info is None:
            global_dof_data, global_motion_info = load_reference_motions()
            if global_dof_data is not None and global_motion_info is not None:
                self.dof_data = global_dof_data
                self.motion_info = global_motion_info
                print(f"Initial motion loaded: {FILE_LIST[global_motion_info['file_num']]}")
            else:
                print("Failed to load initial motion data")

        # start
        self.start_time = time.monotonic()
        self.get_logger().info("Press L1 for start policy")
        self.get_logger().info("Press L2 to emergent stop")
        self.get_logger().info("Press A to start upper body motion")
        self.get_logger().info("Press B to switch motion file")
        self.get_logger().info("Press R2 to reset motion")
        if self.motion_info is not None:
            self.get_logger().info(f"Current motion: {FILE_LIST[self.motion_info['file_num']]}")
        self.init_buffer = 0
        self.foot_contact_buffer = []
        self.time_hist = []
        self.obs_time_hist = []
        self.angle_hist = []
        self.action_hist = []
        self.dof_pos_hist = []
        self.dof_vel_hist = []
        self.imu_hist = []
        self.ang_vel_hist = []
        self.foot_contact_hist = []
        self.tau_hist = []
        self.obs_hist = []
        self.temp_hist = []  # 添加温度历史记录变量

        # cmd and observation
        # self.xyyaw_command = np.array([0, 0., 0.], dtype= np.float32)
        self.xyyaw_command = np.array([0, 0., 0.], dtype= np.float32)
        # self.commands_scale = np.array([self.env.scale_lin_vel, self.env.scale_lin_vel, self.env.scale_ang_vel])
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = torch.zeros((1, 3), device= self.device, dtype= torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1

        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.phase = torch.zeros(1, device=self.device, dtype=torch.float)

        self.Emergency_stop = False
        self.stop = False


        self.gamepad = Gamepad()

        time.sleep(0.02)


def real_h1_2_main(output_folder):
    rclpy.init(args=None)
    dp_node = DeployNode("stand")
    dp_node.get_logger().info("Deploy node started")

    dp_node.main_loop(output_folder)
    rclpy.shutdown()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name: stand, stand_w_waist, wb, squat', required=False, default='stand')
    parser.add_argument('--output-folder', action='store', type=str, help='Output folder for saving data', required=False, default='data_output')
    args = parser.parse_args()
    
    rclpy.init(args=None)
    dp_node = DeployNode(args.task_name)
    dp_node.get_logger().info("Deploy node started")

    dp_node.main_loop(args)
    rclpy.shutdown()