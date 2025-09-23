import os

JOINT_DICT = {
    "a": "left_shoulder_pitch_joint",
    "b": "left_shoulder_yaw_joint",
    "c": "left_shoulder_roll_joint",
    "d": "left_elbow_joint",
    "e": "right_shoulder_pitch_joint",
    "f": "right_shoulder_roll_joint",
    "g": "right_shoulder_yaw_joint",
    "h": "right_elbow_joint",
    "i": "left_wrist_roll_joint",
    "j": "right_wrist_roll_joint",
    "all": "all",
    "a_0": "left_shoulder_pitch_joint_0kg",
    "b_0": "left_shoulder_yaw_joint_0kg",
    "c_0": "left_shoulder_roll_joint_0kg",
    "d_0": "left_elbow_joint_0kg",
    "i_0": "left_wrist_roll_joint_0kg",
    "amass": "amass_merged_data"

    }

def process_motor_delta_action(env_cfg, mode, motion_letter, agent_cfg):
    env_cfg.mode = mode
    motion_joint = JOINT_DICT[motion_letter]
    env_cfg.motion_joint = motion_joint
    env_cfg.motion_file = os.path.join(env_cfg.motion_file, f"motor_edited_extend_{motion_joint}.npz")
    agent_cfg.experiment_name = f"motor_delta_action_{motion_joint}"