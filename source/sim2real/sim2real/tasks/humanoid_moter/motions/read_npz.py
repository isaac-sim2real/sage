import os
import numpy as np

# 27 dofs
_dof_names = ['left_hip_yaw_joint', 'right_hip_yaw_joint', 'torso_joint', 
                'left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_shoulder_pitch_joint', 
                'right_shoulder_pitch_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 
                'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 
                'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 
                'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_joint', 
                'right_elbow_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 
                'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 
                'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
        

def print_all_data_name(data):
    # 列出所有数组名称
    print("数组名称列表：", data.files)

    # 查看每个数组的内容和形状
    for name in data.files:
        print(f"\n数组名称: {name}")
        print("形状:", data[name].shape)
        print("数据类型:", data[name].dtype)
    print(data["real_dof_positions_cmd"][0, 0])
    print(data["joint_sequence"][0])
    import pdb; pdb.set_trace()
        

def delete_data(data, new_file_path):
    # 新建一个字典存放处理后的数据
    new_data = {}

    # 列出所有数组名称
    print("数组名称列表：", data.files)

    # 查看每个数组的内容和形状，并处理
    for name in data.files:
        print(f"\n数组名称: {name}")
        arr = data[name]
        # 兼容list或ndarray
        arr = np.array(arr)
        print("原始形状:", arr.shape)
        print("原始数据类型:", arr.dtype)

        # 删除后三个
        arr_new = arr[:-3]
        print("处理后形状:", arr_new.shape)
        new_data[name] = arr_new

    np.savez(new_file_path, **new_data)
    print(f"\n处理后的数据已保存至: {new_file_path}")

def process_and_save_npz(data, output_file: str, dof_names: list[str]):
    """
    读取 .npz 文件，将数组扩展到 (array.shape[0], array.shape[1], 27) 并保存为新文件。

    Args:
        input_file (str): 输入的 .npz 文件路径。
        output_file (str): 输出的 .npz 文件路径。
        dof_names (list[str]): 所有关节的名称列表 (_dof_names)。
    """
    # 列出所有数组名称
    print("数组名称列表：", data.files)

    # 提取最后一个数组 (joint sequence)
    joint_sequence = data[data.files[-1]]  # 假设最后一个数组是 joint_sequence
    print("\nJoint Sequence:", joint_sequence)

    # 获取 joint_sequence 中每个 joint 在 _dof_names 中的索引
    joint_indices = []
    for joint_name in joint_sequence:
        if joint_name in dof_names:
            joint_indices.append(dof_names.index(joint_name))
        else:
            raise ValueError(f"Joint name '{joint_name}' not found in _dof_names.")

    print("\nJoint indices in _dof_names:", joint_indices)

    # 扩展所有前面的数组到 (array.shape[0], array.shape[1], len(_dof_names))
    expanded_data = {}
    dof_count = len(dof_names)

    for name in data.files[:-1]:  # 遍历除 joint_sequence 以外的数组
        array = data[name]
        print(f"\n处理数组: {name}, 原始形状: {array.shape}")

        # 初始化扩展后的数组
        expanded_array = np.zeros((array.shape[0], array.shape[1], dof_count), dtype=array.dtype)

        # 按索引填充值
        for i, joint_index in enumerate(joint_indices):
            expanded_array[:, :, joint_index] = array

        # 保存扩展后的数组
        expanded_data[name] = expanded_array
        print(f"{name} 已扩展到形状: {expanded_array.shape}")

    # 将 joint_sequence 也保存
    expanded_data["joint_sequence"] = joint_sequence

    # import pdb; pdb.set_trace()

    # 保存为新的 npz 文件
    np.savez(output_file, **expanded_data)
    print(f"\n扩展后的数据已保存到: {output_file}")

def process_all(source_path, save_path):
    file_names = os.listdir(source_path)

    for file_name in file_names:
        data = np.load(os.path.join(source_path, file_name))
        file_name = file_name.split("motor", 1)[1]
        process_and_save_npz(data, os.path.join(save_path, f"motor_edited_extend{file_name}"), _dof_names)


def merge_all(source_path):
    """
    将指定目录中的所有 .npz 文件的数据拼接起来，并保存为一个新的 .npz 文件。

    参数:
    - source_path: 包含 .npz 文件的目录路径。
    - output_file: 输出文件名，默认为 "merged_data.npz"。
    """
    # 用于存储合并后的数据
    merged_data = {}

    # 获取目录中的所有 .npz 文件
    file_names = [f for f in os.listdir(source_path) if f.endswith(".npz")]

    # 遍历每个文件
    for file_name in file_names:
        file_path = os.path.join(source_path, file_name)
        data = np.load(file_path)

        # 遍历每个数组，将数据按名称合并
        for key in data.files:
            # 如果键已存在于 merged_data，则拼接数据
            if key in merged_data:
                merged_data[key] = np.concatenate((merged_data[key], data[key]))
            else:
                # 如果键不存在，直接复制数据
                merged_data[key] = data[key]

    # 保存合并后的数据为新的 .npz 文件
    np.savez(os.path.join(source_path, 'motor_all.npz'), **merged_data)
    print(f"Merged data saved to {source_path}")


if __name__ == "__main__":
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # npz_file_path = os.path.join(script_dir, './humanoid_dance.npz')
    # npz_file_path = os.path.join(script_dir, './motor.npz')
    # npz_file_path = os.path.join(script_dir, './motor_edited.npz')
    # npz_file_path = os.path.join(script_dir, './motor_edited_extend.npz')
    # npz_file_path = os.path.join(script_dir, './motion_perjoint_all/origin/motor_left_elbow_joint.npz')
    # npz_file_path = os.path.join(script_dir, './motion_perjoint_all/motor_edited_extend.npz')
    npz_file_path = os.path.join(script_dir, './motion_perjoint_all/edited/motor_edited_extend_all.npz')
    npz_file_path = os.path.join(script_dir, './motion_perjoint_all/edited/motor_edited_extend_left_shoulder_pitch_joint_0kg.npz')
    # npz_file_path = os.path.join(script_dir, './motion_perjoint_all/origin/motor_left_shoulder_pitch_joint_0kg.npz')



    # 加载 .npz 文件
    # data = np.load(npz_file_path)

    # print_all_data_name(data)
    # delete_data(data, os.path.join(script_dir, 'motor_edited.npz'))
    # process_and_save_npz(data, os.path.join(script_dir, "./motion_perjoint_all/edited/motor_edited_extend_left_shoulder_pitch_joint_0kg.npz"), _dof_names)

    process_all(os.path.join(script_dir, 'motion_perjoint_all/origin/'), os.path.join(script_dir, 'motion_perjoint_all/edited/'))
    # merge_all(os.path.join(script_dir, 'motion_perjoint_all/edited/'))