import os
import time
import numpy as np
import pinocchio as pin



if __name__ == "__main__":
    parts = os.path.abspath(__file__).split(os.sep)
    source_index = parts.index("source")  # 找到 "source" 的位置
    base_path = os.sep.join(parts[:source_index + 1])
    H1_2_WITH_HAND_FIX_URDF_PATH = os.path.join(base_path, "sim2real_assets/sim2real_assets/urdfs/h1_2_hand_fix/h1_2_hand_fix.urdf")

    urdf_model_path = H1_2_WITH_HAND_FIX_URDF_PATH
    meshes_dir = os.path.dirname(urdf_model_path)

    model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, package_dirs=meshes_dir)
    data = model.createData()

    q = pin.randomConfiguration(model)  # in rad for the UR5
    v = np.random.rand(model.nv, 1)  # in rad/s for the UR5
    a = np.random.rand(model.nv, 1)  # in rad/s² for the UR5

    # print("q:", q)
    # print(q.shape)

    joints = []
    for joint_id in range(1, model.njoints):  # 跳过 universe (joint_id=0)
        joint_name = model.names[joint_id]
        joint_type = model.joints[joint_id].shortname()
        
        if joint_type != "JointModelFixed":
            # print(f"{joint_id}: {joint_name}")
            joints.append(joint_name)
    # print(joints)

    joint_urdf = [
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

    joint_usd = [
        'left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 
        'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_pitch_joint', 
        'right_shoulder_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 
        'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
        'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 
        'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 
        'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 
        'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
        ]
    # Computes the inverse dynamics (RNEA) for all the joints of the robot
    joint_urdf_to_joint_usd = [joint_urdf.index(name) for name in joint_usd]  # joint_usd在joint_urdf中的索引
    joint_usd_to_joint_urdf = [joint_usd.index(name) for name in joint_urdf]  # joint_urdf在joint_usd中的索引

    q_usd = np.zeros(27)
    q_usd[2] = 1

    q_urdf = q_usd[joint_usd_to_joint_urdf]
    # print(q_urdf)

    t1 = time.time()
    for i in range(4096):
        q = pin.randomConfiguration(model)  # in rad for the UR5
        v = np.random.rand(model.nv, 1)  # in rad/s for the UR5
        a = np.random.rand(model.nv, 1)  # in rad/s² for the UR5
        tau = pin.rnea(model, data, q, v, a)
    t2 = time.time()
    # print("RNEA time:", t2 - t1)

    # 按 joint_list 顺序提取惯性信息
    for joint_name in joint_urdf:
        joint_id = model.getJointId(joint_name)
        inertia = model.inertias[joint_id]
        joint = model.joints[joint_id]
        
        # 获取关节轴（以 RevoluteJoint 为例）
        # if hasattr(joint, 'axis'):
        if joint.shortname() == "JointModelRZ":
            axis = np.array([0, 0, 1])
        elif joint.shortname() == "JointModelRY":
            axis = np.array([0, 1, 0])
        elif joint.shortname() == "JointModelRX":
            axis = np.array([1, 0, 0])
        
        axis = axis / np.linalg.norm(axis)  # 单位化
        I_axis = axis @ inertia.inertia @ axis
        print(f"{joint_name} 的转轴惯量: {I_axis:.6f} kg·m²")