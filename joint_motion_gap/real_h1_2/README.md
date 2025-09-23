# Real Data Collection

Hardware codebase is based on [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) and [humanplus](https://github.com/MarkFzp/humanplus)

#### Installation

install [unitree_sdk](https://github.com/unitreerobotics/unitree_sdk2)

install [unitree_ros2](https://support.unitree.com/home/en/developer/ROS2_service)

    conda create -n datacollect python=3.8
    conda activate datacollect

install [nvidia-jetpack](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-461/install-jetpack/index.html)

install torch==1.11.0 and torchvision==0.12.0:  
please refer to the following links:   
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html


#### Saved Structure

```
with h5py.File(save_path, 'w') as f:
    f.create_dataset('command_time_list', data=np.array(self.time_hist))
    f.create_dataset('command_val_list', data=np.array(self.action_hist))
    f.create_dataset('motion_name', data=motion_name.encode('utf-8'))
    f.create_dataset('current_time', data=current_time.encode('utf-8'))
    
    g = f.create_group('robot')
    g.create_dataset('joint_time_list', data=np.array(self.time_hist))
    g.create_dataset('joint_angle_list', data=np.array(self.dof_pos_hist))
    g.create_dataset('joint_velocity_list', data=np.array(self.dof_vel_hist))
    g.create_dataset('joint_current_list', data=np.array(self.tau_hist))
    g.create_dataset('joint_temperature_list', data=np.array(self.temp_hist))
    
    g.create_dataset('imu_list', data=np.array(self.imu_hist))
    g.create_dataset('ang_vel_list', data=np.array(self.ang_vel_hist))
    g.create_dataset('obs_list', data=np.array(self.obs_hist))
```

#### Example Usages
Put your trained policy in the `hardware-script/ckpt` folder and rename it to `policy.pt`
Put your target motions in the `motions` folder, and make sure the motion has standard structure as `motions/example.pkl`

    conda activate datacollect
    cd hardware-script
    python hardware_whole_body.py