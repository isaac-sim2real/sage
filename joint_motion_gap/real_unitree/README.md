# Unitree Robot Motion Data Collection

This project provides tools for collecting and processing motion data from Unitree humanoid robots (G1 and H1-2). It supports loading motion sequences from TXT files and collecting real-time robot state data during motion execution using ROS2.

## Installation

1. **Create and activate a Conda environment:**
    ```bash
    conda create -n unitree python=3.9
    conda activate unitree
    ```

2. **Install ROS2 Foxy:**
   Follow the official [ROS2 Foxy installation guide](https://docs.ros.org/en/foxy/Installation.html)

3. **Install Unitree SDK:**
   Refer to [Unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) for detailed installation instructions

4. **Install required Python packages:**
    ```bash
    pip install numpy scipy joblib
    ```

## Usage

### Execute Motion and Collect Data

For G1 robot:
```bash
python hardware_data_collect.py --robot-name g1 --motion-file path/to/motion.txt --output-dir ./output
```

For H1-2 robot:
```bash
python hardware_data_collect.py --robot-name h12 --motion-file path/to/motion.txt --output-dir ./output
```

Or use the main function:
```python
from hardware_data_collect import unitree_robot_main

unitree_robot_main(
    robot_name='g1',  # or 'h12'
    motion_file='path/to/motion.txt',
    output_dir='./output'
)
```

## Input Format

### TXT Motion Files
Motion files should be in CSV format:
```
joint1,joint2,joint3,joint4,joint5,joint6,...
0.1,0.2,0.3,0.4,0.5,0.6,...
0.2,0.3,0.4,0.5,0.6,0.7,...
...
```

- **First row**: Joint names (varies by robot type)
- **Subsequent rows**: Joint angles in radians for each motion frame
- **Frequency**: 50Hz

## Output

The system generates CSV files for each motion:

```
output/
└── motion_name/
    ├── joint_list.txt      # List of joint names
    ├── control.csv         # Command data sent to robot
    ├── event.csv          # Motion events and timestamps
    └── state_motor.csv    # Actual robot state data
```

## Robot Configuration

### Supported Robots:
- **G1**: Unitree G1 humanoid robot
- **H1-2**: Unitree H1-2 humanoid robot

### Default Settings:
- **Control Frequency**: 200Hz
- **Motion Frequency**: 50Hz
- **ROS2 Topics**: `/lowcmd` (command), `/lowstate` (feedback)
