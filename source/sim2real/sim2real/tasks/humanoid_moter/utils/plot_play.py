import os
import numpy as np
import matplotlib.pyplot as plt

def plot_joint_positions(data, save_path, motion_frequency, motion_num):
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
    plt.savefig(os.path.join(save_path, 'joint_position.png'))


def plot_joint_velocity(data, save_path, motion_frequency, motion_num):
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
    plt.savefig(os.path.join(save_path, 'joint_velocity.png'))

def plot_joint_torque(data, save_path, motion_frequency, motion_num):
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
    plt.savefig(os.path.join(save_path, 'joint_torque.png'))

def plot_joint_acc(data, save_path, motion_frequency, motion_num):
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
    plt.savefig(os.path.join(save_path, 'joint_acc.png'))
