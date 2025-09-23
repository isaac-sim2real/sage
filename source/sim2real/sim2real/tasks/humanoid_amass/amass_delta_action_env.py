# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import os
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

import pinocchio as pin


from .amass_delta_action_env_cfg import HumanoidMotorAmassEnvCfg
from .motions import MotionLoaderMotor


class HumanoidMotorAmassEnv(DirectRLEnv):
    cfg: HumanoidMotorAmassEnvCfg

    def __init__(self, cfg: HumanoidMotorAmassEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]

        
        self.mode = self.cfg.mode

        # import pdb; pdb.set_trace()   # self.robot.data.joint_names

        # load motion
        self._motion_loader = MotionLoaderMotor(motion_file=self.cfg.motion_file, device=self.device, mode=self.mode, robot_name=self.cfg.robot_name)  # type: ignore
        self.num_dofs = self._motion_loader.num_dofs

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)

        # 初始化环境的 motion 和 time 索引
        self.motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))

        # shape: (num_envs, history_obs_len==1, obs_len==3)
        self.amp_observation_buffer = torch.zeros((self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device)

        self.apply_action = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        if self.mode == "play":
            self.obs_history = []
            self.reset_nums = 0

        if self.cfg.if_torque_input:
            # pinocchio模型，用于计算等效力矩
            self.pin_model, _, _ = pin.buildModelsFromUrdf(self.cfg.urdf_model_path, package_dirs=self.cfg.package_dirs)
            self.pin_data = self.pin_model.createData()

            self.joint_urdf_to_joint_usd = [self.cfg.urdf_joint_name.index(name) for name in self._motion_loader.dof_names]  # joint_usd在joint_urdf中的索引
            self.joint_usd_to_joint_urdf = [self._motion_loader.dof_names.index(name) for name in self.cfg.urdf_joint_name]  # joint_urdf在joint_usd中的索引

            inertias = []
            for joint_name in self.cfg.urdf_joint_name:
                joint_id = self.pin_model.getJointId(joint_name)
                inertia = self.pin_model.inertias[joint_id]
                joint = self.pin_model.joints[joint_id]
                
                # 获取关节轴
                if joint.shortname() == "JointModelRZ":
                    axis = np.array([0, 0, 1])
                elif joint.shortname() == "JointModelRY":
                    axis = np.array([0, 1, 0])
                elif joint.shortname() == "JointModelRX":
                    axis = np.array([1, 0, 0])
                
                axis = axis / np.linalg.norm(axis)  # 单位化
                I_axis = axis @ inertia.inertia @ axis
                inertias.append(I_axis)
            self.inertias = np.array(inertias)



    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor): 
        self.actions = actions.clone()    # shape: (num_envs, 10)

    def _apply_action(self):
        delta_action = torch.zeros((self.num_envs, self.num_dofs), device=self.device)  # shape: (num_envs, num_dofs)
        joint_index = self._motion_loader.joint_sequence_index     # shape: (10,)

        # delta_action扩展到（4096, 27）
        delta_action[:, joint_index] = self.actions

        # 根据当前 time_indices 和 motion_indices 获取目标位置
        dof_target_pos = self._motion_loader.dof_target_pos[self.motion_indices, self.time_indices]

        self.apply_action = (dof_target_pos + delta_action).clone()
        
        # TODO
        # self.apply_action = dof_target_pos.clone()

        self.robot.set_joint_position_target(self.apply_action)



    def _get_observations(self) -> dict:
        # 当前机器人状态, s_t
        robot_dof_positions = self.robot.data.joint_pos       # shape: (num_envs, num_dofs)
        robot_dof_velocities = self.robot.data.joint_vel      # shape: (num_envs, num_dofs)
        robot_dof_accelerations = self.robot.data.joint_acc   # shape: (num_envs, num_dofs)

        # robot_dof_apply_torque = self.robot.data.computed_torque
        robot_dof_apply_torque = self.robot.data.applied_torque     # shape: (num_envs, num_dofs)

        tau_ext = self.cal_equivalent_torque(robot_dof_positions, robot_dof_velocities, robot_dof_accelerations)    # shape: (num_envs, num_dofs)
        

        # 从 MotionLoader 中采样的状态, a_t
        dof_target_pos = self._motion_loader.dof_target_pos[self.motion_indices, self.time_indices].clone()   # shape: (num_envs, num_dofs)

        # import pdb; pdb.set_trace()
        joint_index = self._motion_loader.joint_sequence_index   # shape: (10,)

        # shape: (num_envs, 50)
        obs = torch.cat([
            robot_dof_positions[:, joint_index],    # shape: (num_envs, 10)
            robot_dof_velocities[:, joint_index], 
            dof_target_pos[:, joint_index],
            robot_dof_accelerations[:, joint_index],
            tau_ext[:, joint_index],
            ], dim=-1)

        if self.mode == "play":
            real_dof_positions = self._motion_loader.dof_positions[self.motion_indices, self.time_indices].clone()
            real_dof_velocities = self._motion_loader.dof_velocities[self.motion_indices, self.time_indices].clone()
            real_dof_torque = self._motion_loader.dof_torque[self.motion_indices, self.time_indices].clone()
            
            # shape: (num_envs==1, num_data_type, 10)
            save_obs = torch.cat([
                robot_dof_positions[:, joint_index].unsqueeze(1),       # shape: (num_envs==1, 1, 10)
                robot_dof_velocities[:, joint_index].unsqueeze(1),
                dof_target_pos[:, joint_index].unsqueeze(1),
                real_dof_positions[:, joint_index].unsqueeze(1),
                real_dof_velocities[:, joint_index].unsqueeze(1),
                robot_dof_apply_torque[:, joint_index].unsqueeze(1),
                real_dof_torque[:, joint_index].unsqueeze(1),
                tau_ext[:, joint_index].unsqueeze(1),
                robot_dof_accelerations[:, joint_index].unsqueeze(1),
            ], dim=-2)
            
            # import pdb; pdb.set_trace()
            
            # shape: (time_length, num_envs==1, num_data_type, 10)
            self.obs_history.append(save_obs.cpu().numpy()) 

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # buffer shape: (num_envs, num_amp_observations, 50)
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": self.amp_observation_buffer.clone().view(self.num_envs, -1)}   # shape: (num_envs, 50*5)


    def _get_rewards(self) -> torch.Tensor:
        position_diff = self._reward_tracking()
        pos_smoothness, vel_smoothness = self._reward_smoothness()


        reward = -10 * position_diff - 10 * pos_smoothness - 1e-3 * vel_smoothness
        
        # reward = -10 * position_diff - 3e-3 * pos_smoothness - 2e-3 * vel_smoothness   # 这个最好，一定要带速度
        
        # import pdb; pdb.set_trace()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)

        # 更新 time_indices
        self.time_indices += 1

        # 检查是否需要重置（time_indices 超过 motion_len）
        motion_done = self.time_indices >= self._motion_loader.motion_len - 1

        # return shape: (num_envs, )
        return died, time_out | motion_done

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        env_ids: The environment IDs to reset. If None, reset all environments.
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES  # type: ignore

        # 随机重置 motion_indices 和 time_indices, shape: [(len(env_ids), ), (len(env_ids), )]
        self.motion_indices[env_ids], self.time_indices[env_ids] = self._motion_loader.sample_indices(len(env_ids))   

       
        # 获取对应的目标状态
        # dof_target_pos = self._motion_loader.dof_target_pos[self.motion_indices[env_ids], self.time_indices[env_ids]]
        # 和实际值相同
        dof_positions = self._motion_loader.dof_positions[self.motion_indices[env_ids], self.time_indices[env_ids]]  # shape: (len(env_ids), num_dofs)
        

        # 重置机器人状态
        self.robot.reset(env_ids)  # type: ignore
        super()._reset_idx(env_ids) # type: ignore


        # 将机器人关节位置重置为目标位置, shape: (len(env_ids), num_dofs)
        joint_pos = dof_positions.clone()  # 和real值相同
        joint_vel = torch.zeros_like(joint_pos, device=self.device)  # reset velocity to 0
        joint_acc = torch.zeros_like(joint_pos, device=self.device)  # reset acceleration to 0
        joint_equivalent_torque = torch.zeros_like(joint_pos, device=self.device)  # reset equivalent torque to 0

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)   # type: ignore

        joint_index = self._motion_loader.joint_sequence_index   # shape: (10,)
        
        # shape: (len(env_ids), 5*num_dofs==50)
        amp_observations = torch.cat([
            joint_pos[:, joint_index], 
            joint_vel[:, joint_index], 
            dof_positions[:, joint_index], 
            joint_acc[:, joint_index],
            joint_equivalent_torque[:, joint_index],
            ], dim=-1)
        # shape: (len(env_ids), num_amp_observations, 50)
        amp_observations = amp_observations.unsqueeze(1).repeat(1, self.cfg.num_amp_observations, 1)  

        # shape: (num_env, num_amp_observations, 50)
        self.amp_observation_buffer[env_ids] = amp_observations

        if self.mode == "play":
            # 记录重置次数
            self.reset_nums += 1
            if self.reset_nums == 6:
                self.close()
                exit()


    def close(self):
        super().close()  # 先执行父类的清理逻辑
        
        if self.mode == "play":
            print("save obs_history")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs(os.path.join(script_dir, f'logs/plays/{self.cfg.motion_joint}/{timestamp}'), exist_ok=True)

            npy_file_path = os.path.join(script_dir, f'logs/plays/{self.cfg.motion_joint}/{timestamp}/play-{timestamp}')

            np.save(npy_file_path, np.stack(self.obs_history))
            
            
            # plot and save
            from .utils.plot_play import plot_joint_positions, plot_joint_velocity, plot_joint_torque, plot_joint_acc
            
            file_path = f"{npy_file_path}.npy"
            save_path = os.path.dirname(file_path)

            data = np.load(file_path)   # shape: (time_length, num_envs==1, num_data_type, 10)

            frequency = 50
            motion_num = 1
            
            plot_joint_positions(data, save_path, frequency, motion_num, self._motion_loader.joint_sequence)
            # plot_joint_velocity(data, save_path, frequency, motion_num, self._motion_loader.joint_sequence)
            # plot_joint_torque(data, save_path, frequency, motion_num, self._motion_loader.joint_sequence)
            # plot_joint_acc(data, save_path, frequency, motion_num, self._motion_loader.joint_sequence)
            


    def cal_equivalent_torque(self, robot_dof_positions, robot_dof_velocities, robot_dof_accelerations):
        # shape: (num_envs, num_dofs), type: ndarray
        q = robot_dof_positions.clone().detach().cpu().numpy()[:, self.joint_usd_to_joint_urdf]
        v = robot_dof_velocities.clone().detach().cpu().numpy()[:, self.joint_usd_to_joint_urdf]
        a = robot_dof_accelerations.clone().detach().cpu().numpy()[:, self.joint_usd_to_joint_urdf]

        # 计算等效力矩
        tau = np.zeros_like(q)
        # 循环num_envs次
        for i in range(q.shape[0]):
            tau[i] = pin.rnea(self.pin_model, self.pin_data, q[i], v[i], a[i])   # type:ignore
        
        # 等效外力矩
        tau_ext = tau - self.inertias * a

        # shape: (num_envs, num_dofs), type: tensor
        return torch.tensor(tau_ext[:, self.joint_urdf_to_joint_usd], device=self.device)  # 返回对应的关节力矩
            




    # ------------------------------------ reward functions ------------------------------------
    def _reward_tracking(self):
        # 机器人当前状态
        robot_dof_positions = self.robot.data.joint_pos     # shape: (num_envs, num_dofs)
        robot_dof_velocities = self.robot.data.joint_vel    # shape: (num_envs, num_dofs)

        # 从 MotionLoader 中采样的状态
        dof_positions = self._motion_loader.dof_positions[self.motion_indices, self.time_indices]    # shape: (num_envs, num_dofs)
        dof_velocities = self._motion_loader.dof_velocities[self.motion_indices, self.time_indices]  # shape: (num_envs, num_dofs)
        
        joint_index = self._motion_loader.joint_sequence_index   # shape: (10,)

        # 计算奖励
        position_diff = (robot_dof_positions - dof_positions)[:, joint_index] ** 2     # shape: (num_envs, 10)
        velocity_diff = (robot_dof_velocities - dof_velocities)[:, joint_index] ** 2   # shape: (num_envs, 10)
        
        position_diff = torch.mean(position_diff, dim=1)   # shape: (num_envs, )
        velocity_diff = torch.mean(velocity_diff, dim=1)   # shape: (num_envs, )

        return position_diff
    
    def _reward_smoothness(self):
        # 提取 joint_positions 和 joint_velocities
        joint_positions = self.amp_observation_buffer[:, :, 0:10]      # shape: (num_envs, history_obs_len, 10)
        joint_velocities = self.amp_observation_buffer[:, :, 10:20]    # shape: (num_envs, history_obs_len, 10)

        # 计算 joint_positions 的平滑性
        # 差分：计算相邻时间步的变化量
        pos_diff = joint_positions[:, 1:] - joint_positions[:, :-1]        # shape: (num_envs, history_obs_len - 1, 10)
        # 差分平方的均值作为平滑性衡量
        pos_smoothness = torch.mean(pos_diff**2, dim=1)       # shape: (num_envs, 10)

        # 计算 joint_velocities 的平滑性
        vel_diff = joint_velocities[:, 1:] - joint_velocities[:, :-1]      # shape: (num_envs, history_obs_len - 1, 10)
        vel_smoothness = torch.mean(vel_diff**2, dim=1)       # shape: (num_envs, 10)

        pos_smoothness = torch.mean(pos_smoothness, dim=1)   # shape: (num_envs, )
        vel_smoothness = torch.mean(vel_smoothness, dim=1)   # shape: (num_envs, )
        
        # 返回结果，形状为 (num_envs, 1)
        return pos_smoothness, vel_smoothness
    
