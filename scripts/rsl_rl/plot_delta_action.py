import os
import re
import sys
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt


def plot_joint_positions(data, save_path):
    robot_dof_positions = data[:, 0, 0]
    dof_target_pos = data[:, 0, 2]
    robot_sim_dof_positions = data[:, 0, 3]
    isaaclab_apply_action = data[:, 0, 7]
    real_dof_positions = data[:, 0, 5]

    # import pdb; pdb.set_trace()
    
    plt.figure(figsize=(10, 6))
    plt.plot(robot_dof_positions, label='delta action dof positions')
    plt.plot(real_dof_positions, label='real dof positions')
    plt.plot(robot_sim_dof_positions, label='isaacsim dof positions')
    plt.plot(dof_target_pos, label='Target dof positions')
    # plt.plot(isaaclab_apply_action, label='isaaclab apply action')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Joint positions & target')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, save_path, 'joint_positions.png'))

def plot_joint_velocity(data, save_path):
    robot_dof_velocities = data[:, 0, 1]
    robot_sim_dof_velocities = data[:, 0, 4]
    real_dof_velocities = data[:, 0, 6]


    plt.figure(figsize=(10, 6))
    plt.plot(robot_dof_velocities, label='delta action dof velocities')
    plt.plot(real_dof_velocities, label='real dof velocities')
    plt.plot(robot_sim_dof_velocities, label='isaacsim dof velocities')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Joint velocities & target')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, save_path, 'joint_velocities.png'))


def plot_joint_positions_only_real(data, save_path, motion_frequency, motion_num):
    robot_dof_positions = data[:, 0, 0][2:]
    dof_target_pos = data[:, 0, 2][2:]
    real_dof_positions = data[:, 0, 3][2:]

    group_size = motion_frequency - 1
    num_groups = motion_num

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # 2行3列，共6个子图
    axs = axs.flatten()  # 方便用1维索引

    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size
        steps = range(0, group_size)
        
        axs[i].plot(steps, robot_dof_positions[start:end], label='delta action dof positions')
        axs[i].plot(steps, real_dof_positions[start:end], label='real dof positions')
        axs[i].plot(steps, dof_target_pos[start:end], label='target dof positions')
        axs[i].set_title(f'Motion {i + 1}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Joint Positions & Target', fontsize=18)
    plt.savefig(os.path.join(script_dir, save_path, 'joint_position.png'))


def plot_joint_velocity_only_real(data, save_path, motion_frequency, motion_num):
    robot_dof_velocities = data[:, 0, 1][2:]
    real_dof_velocities = data[:, 0, 4][2:]

    group_size = motion_frequency - 1
    num_groups = motion_num

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # 2行3列，共6个子图
    axs = axs.flatten()  # 方便用1维索引

    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size
        steps = range(0, group_size)
        
        axs[i].plot(steps, robot_dof_velocities[start:end], label='delta action dof velocity')
        axs[i].plot(steps, real_dof_velocities[start:end], label='real dof velocity')
        axs[i].set_title(f'Motion {i + 1}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Joint velocities', fontsize=18)
    plt.savefig(os.path.join(script_dir, save_path, 'joint_velocity.png'))

def plot_joint_torque_only_real(data, save_path, motion_frequency, motion_num):
    robot_dof_torques = data[:, 0, 5][2:]
    real_dof_torques = data[:, 0, 6][2:]

    tau_external = data[:, 0, 7][2:]

    group_size = motion_frequency - 1
    num_groups = motion_num

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # 2行3列，共6个子图
    axs = axs.flatten()  # 方便用1维索引

    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size
        steps = range(0, group_size)
        
        axs[i].plot(steps, robot_dof_torques[start:end], label='delta action dof torque')
        axs[i].plot(steps, real_dof_torques[start:end], label='real dof torque')
        axs[i].plot(steps, tau_external[start:end], label='tau_external')
        
        axs[i].set_title(f'Motion {i + 1}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Joint velocities', fontsize=18)
    plt.savefig(os.path.join(script_dir, save_path, 'joint_torque.png'))

def plot_joint_acc_only_real(data, save_path, motion_frequency, motion_num):
    dof_acc = data[:, 0, 8][2:]

    group_size = motion_frequency - 1
    num_groups = motion_num

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))  # 2行3列，共6个子图
    axs = axs.flatten()  # 方便用1维索引

    for i in range(num_groups):
        start = i * group_size
        end = (i + 1) * group_size
        steps = range(0, group_size)
        
        axs[i].plot(steps, dof_acc[start:end], label='accleration dof')
        
        axs[i].set_title(f'Motion {i + 1}')
        axs[i].set_xlabel('Time Step')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Joint velocities', fontsize=18)
    plt.savefig(os.path.join(script_dir, save_path, 'joint_acc.png'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot delta action.")
    parser.add_argument("--frequency", type=int, default=50, help="Action frequency.")
    parser.add_argument("--motion_num", type=int, default=6, help="Interval between video recordings (in steps).")
    parser.add_argument("--letter", type=str, default="a", help="The first letter represents joint.")
    args = parser.parse_args()

    from process_env import JOINT_DICT
    motion_joint = JOINT_DICT[args.letter]

    script_dir = os.path.dirname(os.path.abspath(__file__))

    play_dir = os.path.join(script_dir, f'../../source/sim2real/sim2real/tasks/humanoid_moter/logs/plays/{motion_joint}')
    # 获取目录中的所有文件和目录
    file_names = os.listdir(play_dir)

    # 获取绝对路径，并过滤出仅为子目录的路径
    subdirs = [os.path.join(play_dir, name) for name in file_names if os.path.isdir(os.path.join(play_dir, name))]

    # 按子目录的创建时间排序
    file_name = sorted(subdirs, key=os.path.getctime)[-1].split('/')[-1]

    file_path = os.path.join(play_dir, f'{file_name}/play-{file_name}.npy')
    save_path = os.path.join(play_dir, f'{file_name}/')

    data = np.load(file_path)  # shape: [T, 1, obs_dim]


    frequency = args.frequency
    motion_num = args.motion_num


    # plot_joint_positions(data, save_path)
    # plot_joint_velocity(data, save_path)
    plot_joint_positions_only_real(data, save_path, frequency, motion_num)
    plot_joint_velocity_only_real(data, save_path, frequency, motion_num)
    plot_joint_torque_only_real(data, save_path, frequency, motion_num)
    plot_joint_acc_only_real(data, save_path, frequency, motion_num)