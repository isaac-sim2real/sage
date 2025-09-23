
# USDS
ROBOT_BODY_JOINT_NAME_DICT = {
    # 27 dofs
    "h1_2_without_hand_links": [
        'pelvis', 'left_hip_yaw_link', 'left_hip_pitch_link', 
        'left_hip_roll_link', 'left_knee_link', 'left_ankle_pitch_link', 
        'right_hip_yaw_link', 'right_hip_pitch_link', 'right_hip_roll_link', 
        'right_knee_link', 'right_ankle_pitch_link', 'torso_link', 
        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 
        'left_elbow_link', 'torso_link', 'right_shoulder_pitch_link', 
        'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'
        ],

    # 51 dofs
    "h1_2_with_hand_links": [
        'pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 
        'torso_link', 'left_hip_pitch_link', 'right_hip_pitch_link', 
        'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_roll_link',
        'right_hip_roll_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 
        'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 
        'right_shoulder_yaw_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 
        'left_elbow_link', 'right_elbow_link', 'left_ankle_roll_link', 
        'right_ankle_roll_link', 'left_wrist_roll_link', 'right_wrist_roll_link', 
        'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 
        'right_wrist_yaw_link', 'L_index_proximal', 'L_middle_proximal', 
        'L_pinky_proximal', 'L_ring_proximal', 'L_thumb_proximal_base', 
        'R_index_proximal', 'R_middle_proximal', 'R_pinky_proximal', 
        'R_ring_proximal', 'R_thumb_proximal_base', 'L_index_intermediate', 
        'L_middle_intermediate', 'L_pinky_intermediate', 'L_ring_intermediate', 
        'L_thumb_proximal', 'R_index_intermediate', 'R_middle_intermediate', 
        'R_pinky_intermediate', 'R_ring_intermediate', 'R_thumb_proximal', 
        'L_thumb_intermediate', 'R_thumb_intermediate', 'L_thumb_distal', 
        'R_thumb_distal'
        ],

    # 27 dofs
    "h1_2_with_hand_fix_links": [
        'pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 
        'torso_link', 'left_hip_pitch_link', 'right_hip_pitch_link', 
        'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_roll_link', 
        'right_hip_roll_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 
        'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 
        'right_shoulder_yaw_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 
        'left_elbow_link', 'right_elbow_link', 'left_ankle_roll_link', 
        'right_ankle_roll_link', 'left_wrist_roll_link', 'right_wrist_roll_link', 
        'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 
        'right_wrist_yaw_link'],

    # 23 dofs
    "23dofs_joints": [
        'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
        'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
        'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
        'right_elbow_joint', 'right_wrist_roll_joint'
        ],

    # 27 dofs
    "h1_2_without_hand_joints": [
        'left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_pitch_joint', 
        'right_shoulder_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 
        'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
        'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 
        'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 
        'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 
        'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
        ],

    # 51 dofs
    "h1_2_with_hand_joints": [
        'left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_pitch_joint', 
        'right_shoulder_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 
        'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
        'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 
        'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 
        'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 
        'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint', 
        'L_index_proximal_joint', 'L_middle_proximal_joint', 'L_pinky_proximal_joint',
        'L_ring_proximal_joint', 'L_thumb_proximal_yaw_joint', 'R_index_proximal_joint', 
        'R_middle_proximal_joint', 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 
        'R_thumb_proximal_yaw_joint', 'L_index_intermediate_joint', 'L_middle_intermediate_joint', 
        'L_pinky_intermediate_joint', 'L_ring_intermediate_joint', 'L_thumb_proximal_pitch_joint', 
        'R_index_intermediate_joint', 'R_middle_intermediate_joint', 'R_pinky_intermediate_joint', 
        'R_ring_intermediate_joint', 'R_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 
        'R_thumb_intermediate_joint', 'L_thumb_distal_joint', 'R_thumb_distal_joint'
        ],

    # 27 dofs
    "h1_2_with_hand_fix_joints": [
        'left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_pitch_joint', 
        'right_shoulder_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 
        'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
        'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 
        'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 
        'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 
        'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
        ],
}

# URDF
ROBOT_JOINT_NAME_DICT_URDF = {
    "h1_2_with_hand_fix_joints": [
        'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
        'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
        'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 
        'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 
        'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 
        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
}